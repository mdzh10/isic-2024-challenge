import os
import gc
import time
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from tqdm import tqdm
from torcheval.metrics.functional import binary_auroc
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from utils import set_seed, print_trainable_parameters
from datasets import prepare_loaders
from models import ISICModel, ISICModelEdgnet, setup_model, ISICModelEdgnetSegL, ISICModelSegL
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torchvision.utils as vutils


def fetch_scheduler(optimizer, CONFIG):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler


def custom_metric_raw(y_hat, y_true):
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)
    
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])
    
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return partial_auc

def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def bce_dice_loss(pred, target, bce_weight=0.5):
    pred = torch.sigmoid(pred)  # ✅ Make sure predictions are bounded
    bce = F.binary_cross_entropy(pred, target)
    d = dice_loss(pred, target)
    return bce_weight * bce + (1 - bce_weight) * d

def criterion(outputs, targets):
    return nn.BCELoss()(outputs, targets)

def combined_criterion(cls_logits, cls_targets, seg_preds, seg_targets, lambda_cls=1.0, lambda_seg=1.0):
    # If cls_targets is [B] and cls_logits is [B, 1], reshape targets
    if cls_targets.dim() == 1:
        cls_targets = cls_targets.unsqueeze(1)
    cls_loss = F.binary_cross_entropy(cls_logits, cls_targets)
    # print(cls_loss.item())
    # print("Seg pred min/max:", seg_preds.min().item(), seg_preds.max().item())
    # print("Seg target min/max:", seg_targets.min().item(), seg_targets.max().item())
    # print("Seg target dtype:", seg_targets.dtype)

    seg_loss = bce_dice_loss(seg_preds, seg_targets)
    # print("Seg loss:", seg_loss.item())

    return lambda_cls * cls_loss + lambda_seg * seg_loss

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, CONFIG, criterion=criterion, metric_function=binary_auroc, 
                num_classes=1, max_train_steps=None):
    
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc  = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        if max_train_steps is not None and step >= max_train_steps:
            break
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        masks = data['mask'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        if isinstance(model, (ISICModel, ISICModelEdgnet)):
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            loss = loss / CONFIG['n_accumulate']
        else:
            cls_out, seg_out = model(images, seg=True)
            masks = data['mask'].to(device, dtype=torch.float)
            loss = combined_criterion(cls_out, targets, seg_out, masks, lambda_seg=1.0)
            outputs = cls_out.squeeze()

        loss.backward()
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        if num_classes > 1:
            auroc = metric_function(input=outputs.squeeze(), target=torch.argmax(targets, axis=-1), num_classes=num_classes).item()
        else:
            auroc = metric_function(input=outputs.squeeze(), target=targets).item()
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Auroc=epoch_auroc,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss, epoch_auroc


# @torch.inference_mode()
# def valid_one_epoch(model, dataloader, device, epoch, optimizer, criterion=criterion, use_custom_score=True, metric_function=binary_auroc,
#                        num_classes=1, return_preds=False):
#     model.eval()
    
#     dataset_size = 0
#     running_loss = 0.0
#     running_auroc = 0.0
    
#     bar = tqdm(enumerate(dataloader), total=len(dataloader))
#     predictions_all = []
#     targets_all = []
#     for step, data in bar:        
#         images = data['image'].to(device, dtype=torch.float)
#         targets = data['target'].to(device, dtype=torch.float)
        
#         batch_size = images.size(0)

#         outputs = model(images).squeeze()
#         loss = criterion(outputs, targets)

#         predictions_all.append(outputs.cpu().numpy())
#         targets_all.append(targets.cpu().numpy())

#         if num_classes > 1:
#             auroc = metric_function(input=outputs.squeeze(), target=torch.argmax(targets, axis=-1), num_classes=num_classes).item()
#         else:
#             auroc = metric_function(input=outputs.squeeze(), target=targets).item()
#         running_loss += (loss.item() * batch_size)
#         running_auroc  += (auroc * batch_size)
#         dataset_size += batch_size
        
#         epoch_loss = running_loss / dataset_size
#         epoch_auroc = running_auroc / dataset_size
        
#         bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Auroc=epoch_auroc, LR=optimizer.param_groups[0]['lr'])   
    
#     gc.collect()

#     targets_all = np.concatenate(targets_all)
#     predictions_all = np.concatenate(predictions_all)
    
#     epoch_custom_metric = None
#     if use_custom_score:
#         epoch_custom_metric = custom_metric_raw(predictions_all, targets_all)

#     if return_preds:
#         return epoch_loss, epoch_auroc, epoch_custom_metric, predictions_all, targets_all
#     return epoch_loss, epoch_auroc, epoch_custom_metric

@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch, optimizer, criterion=combined_criterion, use_custom_score=True, metric_function=binary_auroc,
                    num_classes=1, return_preds=False, max_val_steps=None):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    predictions_all = []
    targets_all = []

    for step, data in bar:
        if max_val_steps is not None and step >= max_val_steps:
            break
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        masks = data['mask'].to(device, dtype=torch.float)

        batch_size = images.size(0)

        cls_out, seg_out = model(images, seg=True)
        loss = combined_criterion(cls_out, targets, seg_out, masks)

        if step == 0 and (epoch % 10 == 0 or epoch == 1):  # save every 10th epoch or first
            os.makedirs("../debug/edgenext_sac_unet/masks", exist_ok=True)

            # Normalize for visualization
            input_vis = images.clone().detach()
            input_vis = (input_vis - input_vis.min()) / (input_vis.max() - input_vis.min())

            # Get first N examples
            N = min(4, input_vis.shape[0])
            preds = seg_out[:N].cpu()  # (B, 1, H, W)
            masks_gt = masks[:N].cpu()

            # Save predictions, ground truths, and inputs
            vutils.save_image(preds, f"../debug/edgenext_sac_unet/masks/epoch{epoch}_preds.png", normalize=True)
            vutils.save_image(masks_gt, f"../debug/edgenext_sac_unet/masks/epoch{epoch}_masks_gt.png", normalize=True)
            vutils.save_image(input_vis[:N], f"../debug/edgenext_sac_unet/masks/epoch{epoch}_inputs.png", normalize=True)

        predictions_all.append(cls_out.detach().cpu().numpy())
        targets_all.append(targets.cpu().numpy())

        if num_classes > 1:
            auroc = metric_function(input=cls_out.squeeze(), target=torch.argmax(targets, axis=-1), num_classes=num_classes).item()
        else:
            auroc = metric_function(input=cls_out.squeeze(), target=targets).item()

        running_loss += (loss.item() * batch_size)
        running_auroc += (auroc * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_AUROC=epoch_auroc, LR=optimizer.param_groups[0]['lr'])

    gc.collect()
    targets_all = np.concatenate(targets_all)
    predictions_all = np.concatenate(predictions_all)

    epoch_custom_metric = custom_metric_raw(predictions_all, targets_all) if use_custom_score else None

    if return_preds:
        return epoch_loss, epoch_auroc, epoch_custom_metric, predictions_all, targets_all
    return epoch_loss, epoch_auroc, epoch_custom_metric



def get_nth_test_step(epoch):
    if epoch < 6:
        return 5
    if epoch < 10:
        return 4
    if epoch < 15:
        return 3
    if epoch < 20:
        return 2
    return 1

def run_training(
        train_loader, valid_loader, model, optimizer, scheduler, device, num_epochs, CONFIG, 
        model_folder=None, model_name="", seed=42, tolerance_max=15, criterion=criterion, 
        test_every_nth_step=get_nth_test_step, 
        num_classes=1, best_epoch_score_def=-np.inf, max_train_steps=None, max_val_steps=None):
    set_seed(seed)
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_score = best_epoch_score_def
    history = defaultdict(list)
    tolerance = 0

    for epoch in range(1, num_epochs + 1): 
        test_every_nth_step_upd = test_every_nth_step(epoch)
       
        if tolerance > tolerance_max:
            break
        gc.collect()
        train_epoch_loss, train_epoch_auroc = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], 
                                           CONFIG=CONFIG, epoch=epoch, criterion=criterion,
                                        metric_function = binary_auroc, num_classes=num_classes, max_train_steps=max_train_steps)

        if epoch % test_every_nth_step_upd == 0:
            val_epoch_loss, val_epoch_auroc, val_epoch_custom_metric = valid_one_epoch(
                model, valid_loader, device=CONFIG['device'], epoch=epoch, optimizer=optimizer, criterion=criterion,
                metric_function=binary_auroc, num_classes=num_classes, max_val_steps=max_val_steps)
        
            history['Train Loss'].append(train_epoch_loss)
            history['Valid Loss'].append(val_epoch_loss)
            history['Train AUROC'].append(train_epoch_auroc)
            history['Valid AUROC'].append(val_epoch_auroc)
            history['Valid Kaggle metric'].append(val_epoch_custom_metric)
            history['lr'].append( scheduler.get_lr()[0] )
            
            if best_epoch_score <= val_epoch_custom_metric:
                tolerance = 0
                print(f"Validation AUROC Improved ({best_epoch_score} ---> {val_epoch_custom_metric})")
                best_epoch_score = val_epoch_custom_metric
                best_model_wts = copy.deepcopy(model.state_dict())
                if model_folder is not None:
                    torch.save(model.state_dict(), os.path.join(model_folder, model_name))
            else:
                tolerance += 1
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best AUROC: {:.4f}".format(best_epoch_score))    
    model.load_state_dict(best_model_wts)
    return model, history


# def get_metrics(drop_path_rate, drop_rate, models_folder, model_maker, CONFIG):
#     tsp = StratifiedGroupKFold(5, shuffle=True, random_state=CONFIG['seed'])
#     results_list = []
#     fold_df_valid_list = []
#     for fold_n, (train_index, val_index) in enumerate(tsp.split(df_train, y=df_train.target, groups=df_train[CONFIG["group_col"]])):
#         fold_df_train = df_train.iloc[train_index].reset_index(drop=True)
#         fold_df_valid = df_train.iloc[val_index].reset_index(drop=True)
#         set_seed(CONFIG['seed'])
#         model = setup_model(model_name, drop_path_rate=drop_path_rate, drop_rate=drop_rate, model_maker=model_maker)
#         print_trainable_parameters(model)

#         train_loader, valid_loader = prepare_loaders(fold_df_train, fold_df_valid)
    
#         optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
#                            weight_decay=CONFIG['weight_decay'])
#         scheduler = fetch_scheduler(optimizer)
    
#         model, history = run_training(
#             train_loader, valid_loader,
#             model, optimizer, scheduler,
#             device=CONFIG['device'],
#             num_epochs=CONFIG['epochs'],
#             CONFIG=CONFIG,
#             tolerance_max=20,
#             test_every_nth_step=lambda x: 5,
#             seed=CONFIG['seed'])
#         torch.save(model.state_dict(), os.path.join(models_folder, f"model__{fold_n}"))
#         results_list.append(np.max(history['Valid Kaggle metric']))

#         val_epoch_loss, val_epoch_auroc, val_epoch_custom_metric, tmp_predictions_all, tmp_targets_all = valid_one_epoch(
#             model, 
#             valid_loader, 
#             device=CONFIG['device'], 
#             epoch=1, 
#             optimizer=optimizer, 
#             criterion=criterion, 
#             use_custom_score=True,
#             metric_function=binary_auroc, 
#             num_classes=1,
#             return_preds=True)

#         fold_df_valid['tmp_targets_all'] = tmp_targets_all
#         fold_df_valid['tmp_predictions_all'] = tmp_predictions_all
#         fold_df_valid['fold_n'] = fold_n
#         fold_df_valid_list.append(fold_df_valid)
#     fold_df_valid_list = pd.concat(fold_df_valid_list).reset_index(drop=True)
#     return results_list, fold_df_valid_list
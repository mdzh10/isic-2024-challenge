import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import os

class ISICDatasetSamplerW(Dataset):
    def __init__(self, meta_df, transforms=None, process_target: bool=False, n_classes:int=3, weight_adg = 1, do_augmentations: bool=True, *args, **kwargs):
        self.df_positive = meta_df[meta_df["target"] == 1].reset_index()
        self.df_negative = meta_df[meta_df["target"] == 0].reset_index()
        self.file_names_positive = self.df_positive['path'].values
        self.file_names_negative = self.df_negative['path'].values
        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values
        self.negative_weights = self.df_negative['weight'].values
        self.negative_ind = np.arange(0, self.negative_weights.shape[0])
        self.weight_adg = weight_adg
        self.transforms = transforms
        self.n_classes = n_classes
        self.process_target = process_target
        self.do_augmentations = do_augmentations
        
    def __len__(self):
        return len(self.df_positive) * 2
    
    def __getitem__(self, index):
        if random.random() >= 0.5:
            df = self.df_positive
            file_names = self.file_names_positive
            targets = self.targets_positive
            index = index % df.shape[0]
        else:
            df = self.df_negative
            file_names = self.file_names_negative
            targets = self.targets_negative
            index = random.choices(self.negative_ind, weights=self.negative_weights ** self.weight_adg, k=1)[0]
        
        
        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = targets[index]

        if self.transforms and self.do_augmentations: 
            img = self.transforms(image=img)["image"]
            
        if self.process_target:
            target_pr = np.zeros(shape=(self.n_classes,))
            target_pr[int(target)] += 1.0
            target = target_pr
            
        return {
            'image': img,
            'target': target
        }


class ISICDatasetSampler(Dataset):
    def __init__(self, meta_df, transforms=None, process_target: bool = False, n_classes: int = 3, do_augmentations: bool = True):
        self.df_positive = meta_df[meta_df["target"] == 1].reset_index(drop=True)
        self.df_negative = meta_df[meta_df["target"] == 0].reset_index(drop=True)
        self.file_names_positive = self.df_positive['path'].values
        self.file_names_negative = self.df_negative['path'].values
        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values
        self.transforms = transforms
        self.n_classes = n_classes
        self.process_target = process_target
        self.do_augmentations = do_augmentations
        
    def __len__(self):
        return len(self.df_positive) * 2
    
    def __getitem__(self, index):
        # Randomly choose positive or negative
        if random.random() >= 0.5:
            file_names = self.file_names_positive
            targets = self.targets_positive
        else:
            file_names = self.file_names_negative
            targets = self.targets_negative
        index = index % len(file_names)
        
        img_path = file_names[index]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = targets[index]
        
        # Load mask from corresponding path
        base_filename = os.path.basename(img_path)
        mask_filename = base_filename.replace(".jpg", ".png")
        mask_dir = "../data/original/train-image/masks_pred/"
        mask_path = os.path.join(mask_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # If mask not found, create a zero mask with the same size as image.
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            # Ensure mask is exactly the same size as image.
            if mask.shape != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Convert mask to [0, 1] float
        mask = mask.astype(np.float32) / 255.0
        mask = np.clip(mask, 0, 1)
        print("Loaded mask unique values:", np.unique(mask))  # Debug line
        
        # Apply transforms if provided
        if self.transforms and self.do_augmentations:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask'].unsqueeze(0)  # (1, H, W)
        else:
            # Fallback conversion using torch.from_numpy to guarantee contiguous, resizable storage.
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
        # Optionally process target: convert to one-hot if needed.
        if self.process_target:
            target_arr = np.zeros(self.n_classes, dtype=np.float32)
            target_arr[int(target)] = 1.0
            target = target_arr
        else:
            target = float(target)  # make it a float scalar

        # return {
        #     'image': img,
        #     'target': target
        # }
        return {
            'image': img,
            'target': torch.tensor(target, dtype=torch.float32),
            'mask': mask.to(torch.float32)
        }
        
class ISICDatasetSimple(Dataset):
    def __init__(self, meta_df, transforms=None, process_target: bool = False, n_classes: int = 3, do_augmentations: bool = True):
        self.meta_df = meta_df
        self.transforms = transforms
        self.n_classes = n_classes
        self.process_target = process_target
        self.do_augmentations = do_augmentations
        

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        path = row.path
        target = row.target

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        base_filename = os.path.basename(path)
        mask_filename = base_filename.replace(".jpg", ".png")
        mask_dir = "../data/original/train-image/masks_pred/"
        mask_path = os.path.join(mask_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            if mask.shape != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.float32) / 255.0
        mask = np.clip(mask, 0, 1)
        
        if self.transforms and self.do_augmentations:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask'].unsqueeze(0)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)
        
        if self.process_target:
            target_arr = np.zeros(self.n_classes, dtype=np.float32)
            target_arr[int(target)] = 1.0
            target = target_arr
        else:
            target = float(target)

            
        # target = self.targets[idx]    

        # return {
        #     'image': img,
        #     'target': target
        # }

        return {
            'image': img,
            'target': torch.tensor(target, dtype=torch.float32),
            'mask': mask.to(torch.float32)
        }
        


class ISICDatasetSamplerMulticlass(Dataset):
    def __init__(self, meta_df, transforms=None, process_target: bool=False, n_classes:int=3):
        self.df_2 = meta_df[meta_df["target"] == 2].reset_index()
        self.df_1 = meta_df[meta_df["target"] == 1].reset_index()
        self.df_0 = meta_df[meta_df["target"] == 0].reset_index()
        self.file_names_2 = self.df_2['path'].values
        self.file_names_1 = self.df_1['path'].values
        self.file_names_0 = self.df_0['path'].values
        self.targets_2 = self.df_2['target'].values
        self.targets_1 = self.df_1['target'].values
        self.targets_0 = self.df_0['target'].values
        self.transforms = transforms
        self.n_classes = n_classes
        self.process_target = process_target
        
    def __len__(self):
        return len(self.df_2) * 3
    
    def __getitem__(self, index):
        target_p = random.choices([0,1,2], weights=[1,1,1],k=1)[0]
        if target_p == 1:
            df = self.df_1
            file_names = self.file_names_1
            targets = self.targets_1
        elif target_p == 2:
            df = self.df_2
            file_names = self.file_names_2
            targets = self.targets_2
        else:
            df = self.df_0
            file_names = self.file_names_0
            targets = self.targets_0
            
        index = index % df.shape[0]
        
        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = targets[index]

        if self.process_target:
            target_pr = np.zeros(shape=(self.n_classes,))
            target_pr[int(target)] += 1.0
            target = target_pr
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }



def prepare_loaders(df_train, df_valid, CONFIG, data_transforms, data_loader_base=ISICDatasetSampler, weight_adg=1, num_workers=10):
    
    # train_dataset = data_loader_base(df_train, transforms=data_transforms["train"], weight_adg=weight_adg)
    train_dataset = data_loader_base(df_train, transforms=data_transforms["train"])

    valid_dataset = ISICDatasetSimple(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=num_workers, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader
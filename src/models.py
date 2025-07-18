import timm
import torch
from torch import nn
import torch.nn.functional as F

class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, drop_path_rate=0, drop_rate=0, pretrained=True, checkpoint_path=None):
        super(ISICModel, self).__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            heckpoint_path=checkpoint_path,
            drop_rate=drop_rate, 
            drop_path_rate=drop_path_rate)

        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid() if num_classes == 1 else nn.Softmax()

    def forward(self, images):
        return self.sigmoid(self.model(images))


class ISICModelEdgnet(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None, *args, **kwargs):
        super(ISICModelEdgnet, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, global_pool='avg')
        self.sigmoid = nn.Sigmoid()
    def forward(self, images):
        return self.sigmoid(self.model(images))


class ISICModelSegL(nn.Module):
    def __init__(self, model_name, num_classes=1, drop_path_rate=0, drop_rate=0, 
                 pretrained=True, checkpoint_path=None):
        super(ISICModelSegL, self).__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            features_only=True  # <-- get intermediate features
        )

        # Get final feature shape from the encoder's last feature map.
        num_features = self.encoder.feature_info[-1]['num_chs']

        # Classification head: global pool + linear.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(num_features, num_classes)

        # Segmentation head: a single 1x1 convolution that produces raw logits.
        self.seg_head = nn.Conv2d(num_features, 1, kernel_size=1)

        # Activation for classification branch.
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        # x: expected input shape (B, C, H, W)
        feat = self.encoder(x)[-1]  # Last feature map, e.g. shape (B, num_features, h, w)
        logits = self.cls_head(self.pool(feat).squeeze(-1).squeeze(-1))  # Classification logits: (B, num_classes)
        
        seg_logits = self.seg_head(feat)  # Raw segmentation logits: (B, 1, h, w)
        # Upsample segmentation logits to the input resolution (matching ground truth mask size)
        seg_logits_up = F.interpolate(seg_logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return self.activation(logits), seg_logits_up


class ISICModelEdgnetSegL(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None, *args, **kwargs):
        super(ISICModelEdgnet, self).__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            num_classes=0,  # remove head
            global_pool=None,
            features_only=True,
            out_indices=(0,1,2,3),
        )

        feature_dim = self.encoder.feature_info[-1]['num_chs']

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(feature_dim, num_classes)

        self.seg_head = nn.Sequential(
            nn.Conv2d(feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        feat = self.encoder(x)[-1]
        cls_out = self.cls_head(self.pool(feat).squeeze(-1).squeeze(-1))
        seg_out = self.seg_head(feat)
        return self.activation(cls_out), seg_out


def setup_model(model_name, checkpoint_path=None, num_classes=1, drop_path_rate=0, drop_rate=0, device: str = 'cuda', model_maker=ISICModel, seg=False):
    model = model_maker(model_name, pretrained=True, num_classes=num_classes, drop_path_rate=drop_path_rate, drop_rate=drop_rate)

    return model.to(device)


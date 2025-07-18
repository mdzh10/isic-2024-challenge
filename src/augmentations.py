import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentations(CONFIG):
    # Transformations that affect both image and mask
    positional_transforms = [
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
    ]

    # Only for image (pixel-level transforms)
    pixel_transforms = [
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),
        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.0),
            A.ElasticTransform(alpha=3),
        ], p=0.7),
        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.CoarseDropout(
            max_holes=1,
            max_height=int(CONFIG['img_size'] * 0.375),
            max_width=int(CONFIG['img_size'] * 0.375),
            num_holes_range=(1, 1), p=0.7),
    ]

    normalization = [
        A.Normalize(
            mean=[0.4815, 0.4578, 0.4082],
            std=[0.2686, 0.2613, 0.2758],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()
    ]

    data_transforms = {
        "train": A.Compose(positional_transforms + pixel_transforms + normalization),
        "valid": A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            *normalization
        ])
    }

    return data_transforms

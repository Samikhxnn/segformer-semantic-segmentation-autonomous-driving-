import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(size):

    return A.Compose(
        [
            A.Resize(size,size),

            A.HorizontalFlip(p=0.5),

            A.RandomBrightnessContrast(p=0.2),

            A.Normalize(),

            ToTensorV2()
        ]
    )


def get_val_transform(size):

    return A.Compose(
        [
            A.Resize(size,size),

            A.Normalize(),

            ToTensorV2()
        ]
    )
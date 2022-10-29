from torchvision import transforms as T

data_augmentation = T.Compose(
    [
        T.ToPILImage(),
        T.RandomHorizontalFlip(p=0.3),
        T.RandomVerticalFlip(p=0.3),
        T.ToTensor(),
        T.GaussianBlur(3),
        T.RandomErasing(p=0.3),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
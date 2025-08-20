import os
from typing import Tuple, Optional
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def build_transforms(img_size: int):
    train_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_t, test_t

def get_loaders(root: str, img_size: int, batch_size: int, num_workers: int,
                val_split_from_train: float = 0.0):
    train_t, test_t = build_transforms(img_size)
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")

    if os.path.isdir(val_dir) and len(os.listdir(val_dir)) > 0:
        train_ds = datasets.ImageFolder(train_dir, transform=train_t)
        val_ds = datasets.ImageFolder(val_dir, transform=test_t)
    else:
        full_train = datasets.ImageFolder(train_dir, transform=train_t)
        if val_split_from_train and val_split_from_train > 0:
            val_len = int(len(full_train) * val_split_from_train)
            train_len = len(full_train) - val_len
            train_ds, val_ds = random_split(full_train, [train_len, val_len])
        else:
            train_ds = full_train
            val_ds = None

    test_ds = datasets.ImageFolder(test_dir, transform=test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

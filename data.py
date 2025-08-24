from collections import Counter
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def build_transforms(img_size: int, strong_augment: bool = False):
    if strong_augment:
        train_t = transforms.Compose([
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    else:
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
                val_split_from_train: float = 0.0, strong_augment: bool = False):
    train_t, test_t = build_transforms(img_size, strong_augment)
    
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")
    
    print("Using train split for reproducible validation...")
    full_train = datasets.ImageFolder(train_dir, transform=train_t)
    
    if val_split_from_train and val_split_from_train > 0:
        val_len = int(len(full_train) * val_split_from_train)
        train_len = len(full_train) - val_len
        
        generator = torch.Generator().manual_seed(42)
        train_ds, val_split_ds = random_split(full_train, [train_len, val_len], 
                                            generator=generator)
        
        # Create separate dataset for validation with test transforms
        val_ds = datasets.ImageFolder(train_dir, transform=test_t)
        # Use the same indices from the split
        val_ds = torch.utils.data.Subset(val_ds, val_split_ds.indices)
        
        print(f"Dataset split:")
        print(f"  - Training: {train_len:,} samples")
        print(f"  - Validation: {val_len:,} samples")
        print(f"  - Test: Loading...")
        
        # Verify split is reasonable
        if val_len < 100:
            print(f"WARNING: Validation set very small ({val_len} samples)")
            print("   Consider increasing val_split_from_train")
    else:
        train_ds = full_train
        val_ds = None
        print(f"No validation set. Training on {len(train_ds):,} samples")
    
    test_ds = datasets.ImageFolder(test_dir, transform=test_t)
    print(f"  - Test: {len(test_ds):,} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers)
    
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers)
        print(f"Validation batches: {len(val_loader)} (should be >10 for stability)")
        
        if len(val_loader) < 10:
            print("WARNING: Too few validation batches! Results will be noisy.")
            print("   Recommendation: Increase val_split_from_train or batch_size")
    
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def print_detailed_class_analysis(train_ds, val_ds, test_ds):
    """Print comprehensive class analysis"""
    
    datasets = [
        (train_ds, "Training"),
        (val_ds, "Validation"), 
        (test_ds, "Test")
    ]
    
    print("\nCOMPREHENSIVE CLASS ANALYSIS:")
    print("="*50)
    
    for ds, name in datasets:
        if hasattr(ds, 'targets'):
            targets = ds.targets
        elif hasattr(ds, 'dataset') and hasattr(ds, 'indices'):
            targets = [ds.dataset.targets[i] for i in ds.indices]
        else:
            continue
            
        counter = Counter(targets)
        total = len(targets)
        
        normal_count = counter.get(0, 0)
        pneumonia_count = counter.get(1, 0)
        ratio = pneumonia_count / normal_count if normal_count > 0 else float('inf')
        
        print(f"{name}:")
        print(f"  Normal: {normal_count:,} ({normal_count/total*100:.1f}%)")
        print(f"  Pneumonia: {pneumonia_count:,} ({pneumonia_count/total*100:.1f}%)")
        print(f"  Ratio (P:N): {ratio:.2f}:1")
        print()
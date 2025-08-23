# השוואת אסטרטגיות validation
def compare_validation_strategies():
    """השווה בין original val dir לtrain split"""
    
    print("🔍 Validation Strategy Comparison")
    print("="*50)
    
    # Strategy 1: Original val dir
    print("📁 Strategy 1: Original val/ directory")
    print("   Pros:")
    print("   ✅ Truly independent validation set")
    print("   ✅ No data leakage from train")
    print("   ✅ Designed by dataset creators")
    print("   ✅ Standard practice in many competitions")
    
    print("   Cons:")
    print("   ❌ Usually very small (16-50 images)")
    print("   ❌ High variance in metrics")
    print("   ❌ Possibly imbalanced classes")
    print("   ❌ Noisy validation curves")
    
    # Strategy 2: Train split
    print("\n📊 Strategy 2: Train split (25%)")
    print("   Pros:")
    print("   ✅ Large validation set (~1,300 images)")
    print("   ✅ Low variance in metrics")
    print("   ✅ Smooth validation curves")
    print("   ✅ Maintains class balance")
    print("   ✅ More statistical power")
    
    print("   Cons:")
    print("   ❌ Less training data (75% vs 100%)")
    print("   ❌ Potential overfitting to val split")
    print("   ❌ Not truly 'independent'")
    
    print("\n🎯 When to use each:")
    print("Original val_dir:")
    print("  - Competition/benchmark datasets")
    print("  - When val_dir is reasonably sized (>200 images)")
    print("  - When you want maximum training data")
    
    print("Train split:")
    print("  - Research projects")
    print("  - When val_dir is too small")
    print("  - When you need stable validation metrics")
    print("  - Model development and hyperparameter tuning")

# Modified data.py function to support both strategies
def get_loaders_flexible(root: str, img_size: int, batch_size: int, num_workers: int,
                        val_strategy: str = "auto", val_split_from_train: float = 0.25):
    """
    Flexible validation strategy
    val_strategy: "original", "train_split", or "auto"
    """
    
    train_t, test_t = build_transforms(img_size)
    
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")
    
    # Analyze original val directory
    val_exists = os.path.isdir(val_dir)
    val_count = 0
    if val_exists:
        val_count = sum(len(files) for _, _, files in os.walk(val_dir))
    
    # Decide strategy
    if val_strategy == "auto":
        if val_count >= 200:  # Good size threshold
            strategy = "original"
            print(f"🎯 Auto-selected: Original val_dir ({val_count} images)")
        else:
            strategy = "train_split"
            print(f"🎯 Auto-selected: Train split (val_dir too small: {val_count} images)")
    else:
        strategy = val_strategy
        print(f"🎯 Manual selection: {strategy}")
    
    # Apply strategy
    if strategy == "original" and val_exists and val_count > 0:
        print(f"📁 Using original val_dir: {val_count} images")
        train_ds = datasets.ImageFolder(train_dir, transform=train_t)
        val_ds = datasets.ImageFolder(val_dir, transform=test_t)
        
        # Check if it will be stable enough
        val_batches = val_count // batch_size
        if val_batches < 5:
            print(f"⚠️  Warning: Only {val_batches} validation batches - expect high variance!")
        
    else:
        print(f"📊 Using train split: {val_split_from_train*100}% for validation")
        full_train = datasets.ImageFolder(train_dir, transform=train_t)
        
        val_len = int(len(full_train) * val_split_from_train)
        train_len = len(full_train) - val_len
        
        # Fixed seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_ds, val_split_ds = random_split(full_train, [train_len, val_len], 
                                            generator=generator)
        
        # Validation set with test transforms
        val_ds = datasets.ImageFolder(train_dir, transform=test_t)
        val_ds = torch.utils.data.Subset(val_ds, val_split_ds.indices)
        
        print(f"  Training: {train_len:,} samples")
        print(f"  Validation: {val_len:,} samples")
    
    test_ds = datasets.ImageFolder(test_dir, transform=test_t)
    
    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    compare_validation_strategies()
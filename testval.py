# בדיקה של val_dir המקורי
import os
from collections import Counter

def analyze_original_val_dir(root="data/chest_xray"):
    val_dir = os.path.join(root, "val")
    
    if not os.path.exists(val_dir):
        print("❌ val directory doesn't exist!")
        return
    
    print("📊 Original val/ directory analysis:")
    print("="*50)
    
    total_files = 0
    class_distribution = {}
    
    for class_name in os.listdir(val_dir):
        class_path = os.path.join(val_dir, class_name)
        if os.path.isdir(class_path):
            files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            count = len(files)
            class_distribution[class_name] = count
            total_files += count
            print(f"  {class_name}: {count} images")
    
    print(f"\nTotal validation images: {total_files}")
    
    # Calculate batches
    batch_sizes = [16, 32, 64]
    print(f"\nValidation batches by batch_size:")
    for bs in batch_sizes:
        batches = total_files // bs
        remainder = total_files % bs
        print(f"  batch_size {bs}: {batches} full batches" + 
              (f" + {remainder} remainder" if remainder > 0 else ""))
    
    # Check balance
    if len(class_distribution) == 2:
        values = list(class_distribution.values())
        ratio = max(values) / min(values) if min(values) > 0 else float('inf')
        print(f"\nClass balance ratio: {ratio:.1f}:1")
        if ratio > 2.0:
            print("⚠️  Significant class imbalance!")
        else:
            print("✅ Reasonable class balance")
    
    # Stability assessment
    print(f"\n📊 Validation stability assessment:")
    if total_files < 50:
        print("🔴 Very small - expect high variance")
    elif total_files < 200:
        print("🟡 Small - moderate stability")
    elif total_files < 500:
        print("🟢 Medium - good stability")
    else:
        print("🟢 Large - excellent stability")
    
    return total_files, class_distribution

# Run the analysis
if __name__ == "__main__":
    analyze_original_val_dir()
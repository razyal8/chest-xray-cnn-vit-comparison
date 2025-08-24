import os, time, json, math
from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
from utils.metrics import compute_metrics
from utils.plotting import plot_curves
from collections import Counter
import torch.nn as nn

def make_optimizer(params, name, lr, weight_decay):
    name = name.lower()
    lr = float(lr); weight_decay = float(weight_decay)
    if name == "adam":  return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw": return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":   return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name}")

def make_scheduler(optimizer, name, epochs):
    name = name.lower()
    if name == "none": return None
    if name == "cosine": return CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step": return StepLR(optimizer, step_size=max(1, epochs//3), gamma=0.1)
    raise ValueError(f"Unknown scheduler {name}")

def train_one_epoch(model, loader, criterion, optimizer, device, amp=False, grad_clip=None):
    model.train()
    losses, correct, total = 0.0, 0, 0
    use_cuda_amp = amp and device.type == "cuda"

    if use_cuda_amp:
        scaler = torch.amp.GradScaler('cuda', enabled=True)
    else:
        scaler = torch.amp.GradScaler('cpu', enabled=False)

    for images, targets in tqdm(loader, desc="Train", leave=True, ncols=100, position=0):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_cuda_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)
    return losses/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device , desc="Eval"):
    model.eval()
    losses, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    for images, targets in tqdm(loader, desc, leave=True, ncols=100, position=0):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        losses += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
    avg_loss = losses/total
    acc = correct/total
    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = float(avg_loss)
    metrics["accuracy_epoch"] = float(acc)
    return metrics

def create_balanced_criterion(train_loader, device):
    """Create balanced CrossEntropyLoss based on class distribution"""
    
    # ספור classes בtraining set
    class_counts = Counter()
    total_samples = 0
    
    for images, targets in train_loader:
        for target in targets:
            class_counts[target.item()] += 1
            total_samples += 1
    
    # חשב class weights (inverse frequency)
    num_classes = len(class_counts)
    class_weights = []
    
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        weight = total_samples / (num_classes * count)
        class_weights.append(weight)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print(f"📊 Training class distribution:")
    for class_id in sorted(class_counts.keys()):
        class_name = "NORMAL" if class_id == 0 else "PNEUMONIA"
        count = class_counts[class_id]
        weight = class_weights[class_id].item()
        pct = count / total_samples * 100
        print(f"   {class_name}: {count:,} samples ({pct:.1f}%) → weight: {weight:.3f}")
    
    return nn.CrossEntropyLoss(weight=class_weights)

def run_training(model, train_loader, val_loader, test_loader, config, device, out_dir):
    """
    Complete training function with balanced loss, early stopping, and comprehensive logging
    """
    
    epochs = config["train"]["epochs"]
    
    print("🚀 Initializing training...")
    print(f"📊 Training for {epochs} epochs")
    print(f"🎯 Device: {device}")
    print(f"📁 Output directory: {out_dir}")
    
    # Create balanced criterion instead of regular CrossEntropyLoss
    criterion = create_balanced_criterion(train_loader, device)
    
    # Create optimizer and scheduler
    optimizer = make_optimizer(
        model.parameters(), 
        config["train"]["optimizer"], 
        config["train"]["lr"], 
        config["train"]["weight_decay"]
    )
    
    scheduler = make_scheduler(optimizer, config["train"]["scheduler"], epochs)
    
    # Training configuration
    amp_enabled = bool(config["train"]["amp"]) and device.type == "cuda"
    grad_clip = config["train"].get("grad_clip", None)
    
    print(f"⚙️  Training configuration:")
    print(f"   Optimizer: {config['train']['optimizer']} (lr={config['train']['lr']})")
    print(f"   Scheduler: {config['train']['scheduler']}")
    print(f"   Weight decay: {config['train']['weight_decay']}")
    print(f"   AMP enabled: {amp_enabled}")
    print(f"   Gradient clipping: {grad_clip}")
    
    # Early stopping parameters
    best_val_f1 = -1
    patience = 7
    patience_counter = 0
    best_path = None
    
    # History tracking
    history = defaultdict(list)
    
    print("\n🎯 Starting training...")
    
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 50)
        
        # Training phase
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            amp=amp_enabled, 
            grad_clip=grad_clip
        )
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)

        # Validation phase
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device, desc="Validation")
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            
            current_f1 = val_metrics["f1"]
            
            print(f"📊 Epoch {epoch} Results:")
            print(f"   Train: loss={tr_loss:.4f} acc={tr_acc:.4f}")
            print(f"   Val:   loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
                  f"precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} f1={current_f1:.4f}")
            
            # Early stopping logic
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                patience_counter = 0
                
                # Save best model
                best_path = os.path.join(out_dir, "best.pt")
                torch.save({
                    "model": model.state_dict(), 
                    "epoch": epoch, 
                    "val_metrics": val_metrics,
                    "train_loss": tr_loss,
                    "train_acc": tr_acc
                }, best_path)
                print(f"   ✅ New best F1: {current_f1:.4f} (saved to best.pt)")
            else:
                patience_counter += 1
                print(f"   📉 No improvement ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"   🛑 Early stopping! No improvement for {patience} epochs")
                    print(f"   🏆 Best validation F1: {best_val_f1:.4f}")
                    break
        else:
            print(f"📊 Epoch {epoch} Results:")
            print(f"   Train: loss={tr_loss:.4f} acc={tr_acc:.4f}")

        # Learning rate scheduling
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if abs(old_lr - new_lr) > 1e-8:  # Only print if LR actually changed
                print(f"   📈 Learning rate: {old_lr:.6f} → {new_lr:.6f}")

        # Periodic checkpoint saving
        checkpoint_every = config["train"].get("checkpoint_every", 0)
        if checkpoint_every and epoch % checkpoint_every == 0:
            checkpoint_path = os.path.join(out_dir, f"epoch_{epoch}.pt")
            torch.save({
                "model": model.state_dict(), 
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved: epoch_{epoch}.pt")

    # Training completed
    print(f"\n🎉 Training completed!")
    if val_loader is not None:
        print(f"🏆 Best validation F1: {best_val_f1:.4f}")
    
    # Plot training curves
    print("📈 Creating training curves...")
    plot_curves(history, os.path.join(out_dir, "curves.png"))

    # Load best model for final evaluation
    if best_path and os.path.exists(best_path):
        print(f"📥 Loading best model for final evaluation...")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"   Loaded model from epoch {ckpt['epoch']} (F1={ckpt['val_metrics']['f1']:.4f})")

    # Final test evaluation
    print("🧪 Final test evaluation...")
    test_metrics = evaluate(model, test_loader, criterion, device, desc="Test")
    
    # Save final results
    results_path = os.path.join(out_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    # Print final results
    print(f"\n🎯 FINAL TEST RESULTS:")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall: {test_metrics['recall']:.4f}")
    print(f"   F1 Score: {test_metrics['f1']:.4f}")
    
    # Clinical analysis
    cm = test_metrics['confusion_matrix']
    if len(cm) == 2 and len(cm[0]) == 2:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n🏥 Clinical Performance:")
        print(f"   Sensitivity (Recall): {sensitivity:.4f} ({tp}/{tp+fn})")
        print(f"   Specificity: {specificity:.4f} ({tn}/{tn+fp})")
        print(f"   False Positives: {fp} (Normal → Pneumonia)")
        print(f"   False Negatives: {fn} (Pneumonia → Normal)")
        
        if fn <= 20:
            print("   ✅ Excellent: Very few missed pneumonia cases")
        elif fn <= 50:
            print("   🟡 Good: Acceptable number of missed cases")
        else:
            print("   🔴 Concerning: Too many missed pneumonia cases")
    
    print(f"📁 All results saved to: {out_dir}")
    
    return history, test_metrics

import os, time, json, math
from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
from utils.metrics import compute_metrics
from utils.plotting import plot_curves

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

    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

    for images, targets in tqdm(loader, desc="Train", leave=False):
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
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    for images, targets in tqdm(loader, desc="Eval", leave=False):
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

def run_training(model, train_loader, val_loader, test_loader, config, device, out_dir):
    epochs = config["train"]["epochs"]
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model.parameters(), config["train"]["optimizer"], config["train"]["lr"], config["train"]["weight_decay"])
    scheduler = make_scheduler(optimizer, config["train"]["scheduler"], epochs)
    scaler_amp = bool(config["train"]["amp"]) and device.type == "cuda"
    grad_clip = config["train"]["grad_clip"] or None

    history = defaultdict(list)
    best_val_f1, best_path = -1, None

    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, amp=scaler_amp, grad_clip=grad_clip)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            print(f"  Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | Val: loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f}")
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_path = os.path.join(out_dir, "best.pt")
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, best_path)
        else:
            print(f"  Train: loss={tr_loss:.4f} acc={tr_acc:.4f}")

        if scheduler is not None:
            scheduler.step()

        if (config["train"]["checkpoint_every"] and epoch % int(config["train"]["checkpoint_every"]) == 0):
            torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(out_dir, f"epoch_{epoch}.pt"))

    plot_curves(history, os.path.join(out_dir, "curves.png"))

    if best_path and os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    test_metrics = evaluate(model, test_loader, criterion, device)
    with open(os.path.join(out_dir, "report_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("Test metrics:", test_metrics)
    return history, test_metrics

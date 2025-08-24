import argparse, os, json, random, numpy as np, yaml, time
import torch
from data import get_loaders
from models.cnn import build_cnn
from models.vit import build_vit
from train import run_training

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    device = pick_device()
    print(f"Using device: {device}")

    out_dir = cfg["log"]["out_dir"]
    run_name = cfg["log"]["run_name"] or time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # data
    train_loader, val_loader, test_loader = get_loaders(
        cfg["data"]["root"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        val_split_from_train=cfg["data"]["val_split_from_train"],
    )

    # model
    if cfg["model"]["type"] == "cnn":
        mcfg = cfg["model"]["cnn"]
        model = build_cnn(name=mcfg["name"], num_classes=cfg["model"]["num_classes"],
                          pretrained=bool(mcfg["pretrained"]))
    elif cfg["model"]["type"] == "vit":
        mcfg = cfg["model"]["vit"]
        model = build_vit(img_size=cfg["data"]["img_size"], patch_size=mcfg["patch_size"], dim=mcfg["dim"],
                          depth=mcfg["depth"], heads=mcfg["heads"], mlp_ratio=mcfg["mlp_ratio"],
                          drop_rate=mcfg["drop_rate"], num_classes=cfg["model"]["num_classes"])
    else:
        raise ValueError("Unknown model type")

    model.to(device)

    # אם רצים על MPS – נבטל AMP (CUDA AMP לא נתמך שם)
    if device.type == "mps":
        cfg["train"]["amp"] = False

    history, test_metrics = run_training(model, train_loader, val_loader, test_loader, cfg, device, out_dir)

    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(json.dumps(test_metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)

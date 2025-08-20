# run_benchmark.py
import os, time, copy, json, yaml, random, numpy as np
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

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def with_overrides(cfg, **kv):
    """ shallow+dict-merge helper for top-level keys """
    new = copy.deepcopy(cfg)
    for k, v in kv.items():
        if isinstance(v, dict) and isinstance(new.get(k, {}), dict):
            new[k] = {**new.get(k, {}), **v}
        else:
            new[k] = v
    return new

def build_model_from_cfg(cfg):
    if cfg["model"]["type"] == "cnn":
        mcfg = cfg["model"]["cnn"]
        model = build_cnn(
            name=mcfg["name"],
            num_classes=cfg["model"]["num_classes"],
            pretrained=bool(mcfg.get("pretrained", False)),
            freeze_backbone=bool(mcfg.get("freeze_backbone", False)),
        )
    elif cfg["model"]["type"] == "vit":
        mcfg = cfg["model"]["vit"]
        model = build_vit(
            img_size=cfg["data"]["img_size"],
            patch_size=mcfg["patch_size"],
            dim=mcfg["dim"],
            depth=mcfg["depth"],
            heads=mcfg["heads"],
            mlp_ratio=mcfg["mlp_ratio"],
            drop_rate=mcfg["drop_rate"],
            num_classes=cfg["model"]["num_classes"],
        )
    else:
        raise ValueError("Unknown model type")
    return model

def train_one_cfg(cfg, device, run_name=None):
    # AMP רק ב-CUDA
    if device.type != "cuda":
        cfg = copy.deepcopy(cfg)
        cfg["train"]["amp"] = False

    # יציאת ריצה
    tag_time = time.strftime("%Y%m%d_%H%M%S")
    actual_run_name = run_name or cfg["log"].get("run_name") or f"{cfg['model']['type']}_{tag_time}"
    out_dir = os.path.join(cfg["log"]["out_dir"], actual_run_name)
    os.makedirs(out_dir, exist_ok=True)

    # DataLoaders (כיבוי pin_memory ב-MPS/CPU)
    pin_memory = torch.cuda.is_available()
    train_loader, val_loader, test_loader = get_loaders(
        cfg["data"]["root"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        val_split_from_train=cfg["data"]["val_split_from_train"],
        # אם הוספת פרמטר pin_memory ל-get_loaders, בטל את הקומנט הבא:
        # pin_memory=pin_memory,
    )

    # מודל
    model = build_model_from_cfg(cfg)
    model.to(device)
    n_params = count_params(model)

    # אימון + הערכה
    start = time.time()
    history, test_metrics = run_training(model, train_loader, val_loader, test_loader, cfg, device, out_dir)
    wall_time_sec = time.time() - start

    # שמירת קונפיג
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    # metadata להשוואה
    test_metrics.update({
        "num_params": int(n_params),
        "device": device.type,
        "wall_time_sec": round(wall_time_sec, 2),
        "out_dir": out_dir,
    })
    return test_metrics

def main():
    cfg = load_cfg("config.yaml")
    set_seed(cfg.get("seed", 42))
    device = pick_device()
    print(f"Using device: {device}")

    # שלוש תוכניות ריצה:
    plans = [
        ("cnn_custom", with_overrides(cfg, model={"type": "cnn", "cnn": {"name": "custom", "pretrained": False}}, log={"run_name": "cnn_custom"})),
        ("cnn_pretrained", with_overrides(cfg, model={"type": "cnn", "cnn": {"name": "resnet18", "pretrained": True}}, log={"run_name": "cnn_pretrained"})),
        ("vit", with_overrides(cfg, model={"type": "vit"}, log={"run_name": "vit"})),
    ]

    results = []
    for tag, cfg_i in plans:
        print(f"\n=== Running {tag.upper()} ===")
        res = train_one_cfg(cfg_i, device, run_name=cfg_i["log"]["run_name"])
        res["model_type"] = tag
        print(f"{tag.upper()} TEST metrics: {json.dumps(res, indent=2)}")
        results.append(res)

    # כתיבת קובץ השוואה
    os.makedirs(cfg["log"]["out_dir"], exist_ok=True)
    compare_path = os.path.join(cfg["log"]["out_dir"], "comparison.json")
    with open(compare_path, "w") as f:
        json.dump(results, f, indent=2)

    # הדפסה קצרה למסך
    headers = ["model","acc","prec","recall","f1","params","time_s","out_dir"]
    def row(r): return [r["model_type"], r.get("accuracy"), r.get("precision"), r.get("recall"), r.get("f1"), r["num_params"], r["wall_time_sec"], r["out_dir"]]
    widths = [max(len(str(x)) for x in [h]+[row(r)[i] for r in results]) for i,h in enumerate(headers)]
    print("\n=== COMPARISON ===")
    print(" | ".join(h.ljust(widths[i]) for i,h in enumerate(headers)))
    for r in results:
        vals = row(r)
        print(" | ".join(str(vals[i]).ljust(widths[i]) for i in range(len(headers))))
    print(f"\nSaved summary: {compare_path}")

if __name__ == "__main__":
    main()

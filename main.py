# run_benchmark.py
import os
import time
import copy
import json
import yaml
import random
from pathlib import Path

import numpy as np
import torch

from data import get_loaders
from models.cnn import build_cnn
from models.vit import build_vit
from train import run_training
from utils.advanced_analysis import analyze_training_behavior, compare_models_analysis, generate_analysis_report, plot_detailed_curves


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_model_params(model):
    """Count total number of model parameters."""
    return sum(p.numel() for p in model.parameters())


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config, output_path):
    """Save configuration to YAML file."""
    with open(output_path, "w") as f:
        yaml.safe_dump(config, f)


def update_config(base_config, **updates):
    """Update configuration with new values."""
    new_config = copy.deepcopy(base_config)
    
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(new_config.get(key, {}), dict):
            new_config[key] = {**new_config.get(key, {}), **value}
        else:
            new_config[key] = value
    
    return new_config


def build_model(config):
    """Build model based on configuration."""
    model_type = config["model"]["type"]
    num_classes = config["model"]["num_classes"]
    
    if model_type == "cnn":
        cnn_config = config["model"]["cnn"]
        return build_cnn(
            name=cnn_config["name"],
            num_classes=num_classes,
            pretrained=cnn_config.get("pretrained", False),
        )
    
    elif model_type == "vit":
        vit_config = config["model"]["vit"]
        return build_vit(
            img_size=config["data"]["img_size"],
            patch_size=vit_config["patch_size"],
            dim=vit_config["dim"],
            depth=vit_config["depth"],
            heads=vit_config["heads"],
            mlp_ratio=vit_config["mlp_ratio"],
            drop_rate=vit_config["drop_rate"],
            num_classes=num_classes,
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_output_directory(base_dir, run_name):
    """Create output directory for experiment."""
    output_dir = Path(base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def get_data_loaders(config):
    """Get data loaders based on configuration."""
    data_config = config["data"]
    
    
    return get_loaders(
        root=data_config["root"],
        img_size=data_config["img_size"],
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        val_split_from_train=data_config["val_split_from_train"],
    )


def train_single_model(config, device, run_name=None):
    """Train a single model with given configuration."""
    
    # Disable AMP for non-CUDA devices
    if device.type != "cuda":
        config = copy.deepcopy(config)
        config["train"]["amp"] = False
    
    # Setup run name and output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if not run_name:
        run_name = config["log"].get("run_name", f"{config['model']['type']}_{timestamp}")
    
    output_dir = create_output_directory(config["log"]["out_dir"], run_name)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # Build and prepare model
    model = build_model(config)
    model.to(device)
    num_params = count_model_params(model)
    
    # Train model
    start_time = time.time()
    history, test_metrics = run_training(
        model, train_loader, val_loader, test_loader, 
        config, device, output_dir
    )
    training_time = time.time() - start_time
    
    # Save configuration used
    save_config(config, os.path.join(output_dir, "config_used.yaml"))
    
    # Analyze training behavior
    training_analysis = analyze_training_behavior(history, output_dir)
    
    # Create detailed plots
    model_name = config["model"]["type"] + (f"_{config['model']['cnn']['name']}" if config["model"]["type"] == "cnn" else "")
    plot_detailed_curves(history, output_dir, model_name)
    
    # Add metadata to results
    test_metrics.update({
        "num_params": int(num_params),
        "device": device.type,
        "wall_time_sec": round(training_time, 2),
        "out_dir": output_dir,
        "training_analysis": training_analysis  
    })
    
    return test_metrics


def create_experiment_plans(base_config):
    plans = [
        {
            "name": "custom_cnn",
            "config": update_config(
                base_config,
                model={"type": "cnn", "cnn": {"name": "custom"}},
                log={"run_name": "custom_cnn"}
            )
        },
        {
            "name": "resnet18_pretrained", 
            "config": update_config(
                base_config,
                model={"type": "cnn", "cnn": {"name": "resnet18", "pretrained": True}},
                log={"run_name": "resnet18_pretrained"}
            )
        },
        {
            "name": "vit_optimized",
            "config": update_config(
                base_config,
                model={
                    "type": "vit", 
                    "vit": {
                        "patch_size": 14,
                        "dim": 384,
                        "depth": 6,
                        "heads": 6,
                        "mlp_ratio": 4,     
                        "drop_rate": 0.15
                    }
                },
                log={"run_name": "vit_optimized"}
            )
        },
    ]
    return plans


def run_experiments(plans, device):
    """Run all experiments and collect results."""
    results = []
    
    for plan in plans:
        name = plan["name"]
        config = plan["config"]
        
        print(f"\n{'='*20} Running {name.upper()} {'='*20}")
        
        result = train_single_model(
            config, 
            device, 
            run_name=config["log"]["run_name"]
        )
        result["model_type"] = name
        
        print(f"\n{name.upper()} Test Results:")
        print(json.dumps(result, indent=2))
        
        results.append(result)
    
    return results


def save_comparison(results, output_dir):
    """Save comparison results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    comparison_path = os.path.join(output_dir, "comparison.json")
    
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return comparison_path


def print_comparison_table(results):
    """Print formatted comparison table."""
    # Define table columns
    columns = [
        ("Model", "model_type"),
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Params", "num_params"),
        ("Time (s)", "wall_time_sec"),
        ("Output Dir", "out_dir")
    ]
    
    # Calculate column widths
    widths = []
    for header, key in columns:
        values = [header] + [str(r.get(key, "N/A")) for r in results]
        widths.append(max(len(v) for v in values))
    
    # Print header
    print("\n" + "="*50 + " COMPARISON " + "="*50)
    headers = [h.ljust(w) for (h, _), w in zip(columns, widths)]
    print(" | ".join(headers))
    print("-" * (sum(widths) + 3 * (len(columns) - 1)))
    
    # Print rows
    for result in results:
        row = []
        for (_, key), width in zip(columns, widths):
            value = str(result.get(key, "N/A"))
            row.append(value.ljust(width))
        print(" | ".join(row))


def main():
    """Main benchmark runner."""
    print("="*60)
    print(" "*20 + "BENCHMARK RUNNER")
    print("="*60)
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config("config.yaml")
    
    # Set random seed
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Random seed set to: {seed}")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create experiment plans
    plans = create_experiment_plans(config)
    print(f"Running {len(plans)} experiment(s)")
    
    # Run experiments
    results = run_experiments(plans, device)
    
    # Enhanced model comparison
    compare_models_analysis(results, config["log"]["out_dir"])
    
    # Generate analysis report
    generate_analysis_report(results, config["log"]["out_dir"])
    
    # Save comparison
    comparison_path = save_comparison(results, config["log"]["out_dir"])
    print(f"\nSaved comparison to: {comparison_path}")
    
    # Print comparison table
    print_comparison_table(results)
    
    # Print analysis summary
    print("\n" + "="*50 + " ANALYSIS SUMMARY " + "="*50)
    print("Enhanced analysis files created:")
    print(f"- Detailed plots: {config['log']['out_dir']}/model_comparison.png")
    print(f"- Analysis report: {config['log']['out_dir']}/analysis_report.md")
    print("- Individual model analyses in each output directory")
    
    print("\nBenchmark completed successfully!")



if __name__ == "__main__":
    main()
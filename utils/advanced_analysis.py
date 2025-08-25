import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def analyze_training_behavior(history, out_dir):
    """Comprehensive training analysis"""
    
    # 1. Convergence analysis
    train_loss = history['train_loss']
    val_loss = history.get('val_loss', [])
    
    results = {
        'converged': False,
        'best_epoch': 0,
        'overfitting_detected': False,
        'generalization_gap': 0
    }
    
    if val_loss:
        best_epoch = np.argmin(val_loss)
        results['best_epoch'] = best_epoch
        
        # Check convergence
        stable_window = max(3, len(val_loss) // 5)
        recent_std = np.std(val_loss[-stable_window:])
        results['converged'] = recent_std < 0.01
        
        # Overfitting detection
        if best_epoch < len(val_loss) * 0.8: 
            results['overfitting_detected'] = True
        
        # Generalization gap
        final_train_loss = train_loss[-1]
        final_val_loss = val_loss[-1]
        results['generalization_gap'] = final_val_loss - final_train_loss
    
    # Convert numpy types to Python types for JSON serialization
    for key, value in results.items():
        if hasattr(value, 'item'): 
            results[key] = value.item()
        elif isinstance(value, np.bool_):
            results[key] = bool(value)
        elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            results[key] = int(value)
        elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
            results[key] = float(value)
    
    # Save analysis
    import json
    with open(f"{out_dir}/training_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_detailed_curves(history, out_dir, model_name):
    """Enhanced plotting with analysis - 3 plots in one row"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{model_name} - Training Analysis', fontsize=16)

    # ----- Loss curves -----
    ax = axes[0]
    ax.plot(history['train_loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        val_loss = history['val_loss']
        ax.plot(val_loss, label='Validation Loss', linewidth=2)

        # Best epoch
        best_epoch = np.argmin(val_loss)
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
        ax.scatter([best_epoch], [val_loss[best_epoch]], color='red', s=100, zorder=5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Loss Curves')
    ax.grid(True, alpha=0.3)

    # ----- Accuracy curves -----
    ax = axes[1]
    ax.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    if 'val_acc' in history and history['val_acc']:
        ax.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.set_title('Accuracy Curves')
    ax.grid(True, alpha=0.3)

    # ----- Overfitting analysis -----
    ax = axes[2]
    if 'val_loss' in history and history['val_loss']:
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        gap = [v - t for v, t in zip(val_loss, train_loss)]
        ax.plot(gap, label='Generalization Gap (Val - Train)', linewidth=2, color='orange')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.fill_between(range(len(gap)), gap, alpha=0.3, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Difference')
    ax.legend()
    ax.set_title('Overfitting Analysis')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/detailed_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def compare_models_analysis(results, out_dir):
    """Comprehensive model comparison"""
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Comparison Analysis', fontsize=16)
    
    models = [r['model_type'] for r in results]
    
    # Performance metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_data = {metric: [r.get(metric, 0) for r in results] for metric in metrics}
    
    ax = axes[0, 0]
    x = np.arange(len(models))
    width = 0.2
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, metric_data[metric], width, label=metric.capitalize())
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Model complexity
    ax = axes[0, 1]
    params = [r.get('num_params', 0) / 1e6 for r in results] 
    colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(models)]
    bars = ax.bar(models, params, color=colors)
    ax.set_ylabel('Parameters (Millions)')
    ax.set_title('Model Complexity')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:.1f}M', ha='center', va='bottom')
    
    # Training time
    ax = axes[0, 2]
    times = [r.get('wall_time_sec', 0) / 60 for r in results]  # In minutes
    bars = ax.bar(models, times, color=colors)
    ax.set_ylabel('Training Time (Minutes)')
    ax.set_title('Training Efficiency')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}m', ha='center', va='bottom')
    
    # Efficiency analysis (F1 vs Parameters)
    ax = axes[1, 0]
    f1_scores = [r.get('f1', 0) for r in results]
    param_counts = [r.get('num_params', 0) / 1e6 for r in results]
    
    scatter = ax.scatter(param_counts, f1_scores, s=100, c=colors, alpha=0.7)
    for i, model in enumerate(models):
        ax.annotate(model, (param_counts[i], f1_scores[i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Parameters (Millions)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Efficiency: F1 vs Model Size')
    ax.grid(True, alpha=0.3)
    
    # Confusion matrices comparison
    ax = axes[1, 1]
    if all('confusion_matrix' in r for r in results):
        # Show confusion matrix for best performing model
        best_model_idx = np.argmax([r.get('f1', 0) for r in results])
        best_cm = results[best_model_idx]['confusion_matrix']
        
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Best Model CM: {models[best_model_idx]}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Speed vs Accuracy trade-off
    ax = axes[1, 2]
    accuracy_scores = [r.get('accuracy', 0) for r in results]
    training_times = [r.get('wall_time_sec', 0) for r in results]
    
    scatter = ax.scatter(training_times, accuracy_scores, s=100, c=colors, alpha=0.7)
    for i, model in enumerate(models):
        ax.annotate(model, (training_times[i], accuracy_scores[i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Training Time (Seconds)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Speed vs Accuracy Trade-off')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_analysis_report(results, out_dir):
    """Generate comprehensive text analysis"""
    
    report = []
    report.append("# Model Comparison Analysis Report\n")
    
    # Performance summary
    report.append("## Performance Summary")
    best_acc_idx = np.argmax([r.get('accuracy', 0) for r in results])
    best_f1_idx = np.argmax([r.get('f1', 0) for r in results])
    
    report.append(f"**Best Accuracy:** {results[best_acc_idx]['model_type']} ({results[best_acc_idx]['accuracy']:.4f})")
    report.append(f"**Best F1-Score:** {results[best_f1_idx]['model_type']} ({results[best_f1_idx]['f1']:.4f})")
    
    # Efficiency analysis
    report.append("\n## Efficiency Analysis")
    for result in results:
        model_name = result['model_type']
        params = result.get('num_params', 0) / 1e6
        time = result.get('wall_time_sec', 0) / 60
        f1 = result.get('f1', 0)
        
        efficiency = f1 / params if params > 0 else 0
        report.append(f"**{model_name}:** {params:.1f}M params, {time:.1f}min training, F1/Param ratio: {efficiency:.6f}")
    
    # Save report
    with open(f"{out_dir}/analysis_report.md", "w") as f:
        f.write("\n".join(report))
    
    return report
# Pneumonia Classification – Final Project (DL Course 2025-B)

This repository implements and compares deep learning models for **pneumonia detection** from chest X-ray images:

1. **Custom CNN** – a lightweight 4-block convolutional neural network.
2. **ResNet18 (Pretrained)** – transfer learning using torchvision’s ResNet18 pretrained on ImageNet.
3. **Vision Transformer (ViT)** – implemented from scratch, configurable (patch size, depth, heads, etc.).

The goal is to evaluate trade-offs between CNNs and Transformers in terms of accuracy, precision/recall balance, and efficiency (training time, number of parameters).

---

## Dataset

The project uses the [Kaggle Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

Expected directory structure:

```
data/
     chest_xray/
       train/
         NORMAL/
         PNEUMONIA/
       val/
         NORMAL/
         PNEUMONIA/
       test/
         NORMAL/
         PNEUMONIA/

```

- **Total images**: 5,863 (pediatric CXRs, ages 1–5).
- **Classes**: NORMAL vs. PNEUMONIA.
- Labels were verified by multiple physicians.
- Original validation set (16 images) is replaced by a 25% train split (~1,304 images) for stable evaluation.

---

## Installation

Requirements: **Python 3.10+**, PyTorch, torchvision.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quickstart

1. **Edit `config.yaml`** if needed (dataset paths, model choice, hyperparams).

2. **Run all models (CNN + ResNet18 + ViT) and compare**:

   ```bash
   python main.py
   ```

3. **Outputs** (in `outputs/`): model checkpoints, training curves, confusion matrices, and a `report_metrics.json` for the final test evaluation.

---

## Outputs

After training, results are saved under `outputs/`:

- **Model checkpoints** (`.pt`)
- **Training curves** (`curves.png`, `detailed_analysis.png`)
- **Confusion matrices**
- **Comparison report** (`comparison.json`, `analysis_report.md`)

All results are saved under `outputs/` (example of my run at outputs-run).

- **Per-model directories** (`custom_cnn/`, `resnet18_pretrained/`, `vit_optimized/`):

  - `best.pt` → best checkpoint by validation F1
  - `epoch_X.pt` → checkpoints saved every few epochs
  - `config_used.yaml` → exact configuration used for the run
  - `curves_acc.png` / `curves_loss.png` → accuracy & loss curves
  - `detailed_analysis.png` → combined training analysis (loss, accuracy, generalization gap)
  - `test_results.json` → final evaluation metrics on the test set
  - `training_analysis.json` → convergence and overfitting analysis

- **Global outputs** (in `outputs/` root):
  - `comparison.json` → summary of all models (accuracy, precision, recall, F1, params, time)
  - `model_comparison.png` → comparison plots (metrics, complexity, efficiency, confusion matrix)
  - `analysis_report.md` → text summary of results and efficiency analysis

Example final test results:

| Model               | Accuracy  | Precision | Recall    | F1        | Params | Time (min) |
| ------------------- | --------- | --------- | --------- | --------- | ------ | ---------- |
| Custom CNN          | 0.792     | 0.752     | **0.995** | 0.857     | 0.6M   | 27.2       |
| ResNet18 Pretrained | 0.817     | 0.775     | **0.997** | **0.872** | 11.2M  | 26.0       |
| ViT Optimized       | **0.830** | **0.832** | 0.913     | 0.870     | 11.0M  | 44.0       |

---

## Project Structure

.
├── data/ # dataset (Chest X-Ray Pneumonia, from Kaggle)
│ └── chest_xray/ # contains train/ val/ test/ subfolders
├── models/ # model definitions (Custom CNN, ResNet18, ViT)
├── utils/ # helper functions: metrics, plotting, analysis
├── outputs-run/ # experiment outputs (checkpoints, plots, reports)
├── config.yaml # experiment configuration
├── main.py # run all models (CNN, ResNet18, ViT) for comparison
├── train.py # training + evaluation logic
├── requirements.txt # dependencies
└── README.md # project documentation

---

## 👨‍💻 Author

**Razy Alshekh**
Deep Learning Course – Reichman University (RUNI), Semester B, 2025

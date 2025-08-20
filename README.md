# Pneumonia Classification (Guided Project)

This repo trains two image classifiers on the Chest X-Ray Pneumonia dataset:
1) A CNN baseline (transfer learning on torchvision backbones).
2) A Vision Transformer (ViT) implemented from scratch, configurable (patch size, depth, heads, etc.).

## Quickstart

1. **Download dataset** from Kaggle: `paultimothymooney/chest-xray-pneumonia`  
   Directory structure should be:
   ```
   data/
     chest_xray/
       train/
         NORMAL/
         PNEUMONIA/
       val/        # you may create this by splitting train
         NORMAL/
         PNEUMONIA/
       test/
         NORMAL/
         PNEUMONIA/
   ```

2. **Install deps** (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

3. **Edit `config.yaml`** if needed (dataset paths, model choice, hyperparams).

4. **Train**:
   ```bash
   python main.py --config config.yaml
   ```

5. **Outputs** (in `outputs/`): model checkpoints, training curves, confusion matrices, and a `report_metrics.json` for the final test evaluation.

### Re-run with ViT
Edit `model.type: vit` in `config.yaml` (and adjust ViT hyperparameters).

## Notes
- Test set is never used for training/selection. Train & validate; use test only once for final report.
- Mixed precision (AMP) is supported.
- Reproducibility: seeds are set; full determinism is not guaranteed across HW/CUDA.

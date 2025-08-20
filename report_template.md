# Pneumonia Classification: CNN vs. ViT

**Authors:** <Your Names>  
**Course:** Deep Learning, Reichman University, Semester B 2025

## 1. Introduction
- Task: Binary classification of chest X-rays: NORMAL vs. PNEUMONIA.
- Motivation: clinical utility and benchmarking CNN vs. ViT.
- Contributions: (a) clean baseline with transfer learning; (b) ViT from scratch with ablations (patch size, depth); (c) analysis.

## 2. Background
- Brief overview of CNNs and inductive biases for vision.
- Vision Transformer (ViT) core ideas (patch embedding, MHSA, class token, position embeddings). Cite Dosovitskiy et al. (2020).
- Related work comparing CNNs & ViTs; mention training data scale, augmentation, optimization.

## 3. Data
- Dataset source (Kaggle). Class distribution. Any train/val split decisions.
- Preprocessing and augmentations.
- Leakage safeguards: test untouched.

## 4. Methods
### 4.1 CNN
- Backbone (e.g., ResNet18) + fine-tuning policy. Parameters & #params.

### 4.2 ViT
- Architecture details (patch size, dim, depth, heads, mlp-ratio). Initialization, dropout.

### 4.3 Training
- Optimizer, LR schedule, epochs, batch size, AMP, regularization. Early stopping/checkpoints.

## 5. Experiments
- Metrics: Accuracy, Precision, Recall, F1. Confusion matrices.
- Convergence curves. Compute time per epoch, total training time.
- Ablations: (e.g., data aug on/off, patch_size ∈ {8,16}).

## 6. Results
- Table comparing CNN vs. ViT on validation and on **test** (final).
- Qualitative error analysis: sample false positives/negatives (if permitted).

## 7. Discussion
- Over/underfitting signs, generalization behavior.
- Strengths & weaknesses for this task.
- Practical considerations (data size, compute).

## 8. Conclusion
- Final takeaways and future work (self-supervision, better regularization, larger ViT).

## References
- Dosovitskiy et al., 2020: "An Image is Worth 16x16 Words..."
- Touvron et al., 2021; "When ViTs outperform ResNets..." etc.

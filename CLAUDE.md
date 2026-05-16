# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Swin-Base (ImageNet-22K pretrained) fine-tuned for 3-class chest X-ray classification (COVID19 / NORMAL / PNEUMONIA). Full training pipeline with modern techniques: mixed precision (AMP), exponential moving average (EMA), Mixup/CutMix augmentation, cosine annealing LR schedule, and gradient clipping.

## Required Environment

- **Python**: 3.10 (conda env: `vit` at `D:\anaconda\envs\vit`)
- **PyTorch**: 2.6.0+cu124 (CUDA 12.4, NVIDIA GeForce GTX 1650)
- **Key packages**: `timm` (1.0.26), `torchvision`, `matplotlib`, `scikit-learn`
- **Large files tracked via Git LFS**: `*.pth`

## Commands

```bash
# Activate environment
conda activate vit

# Train the model
python train.py

# Run inference on test images
python test.py

# Compute dataset mean/std
python mean_std.py

# Split raw data into train/val
python makedata.py
```

## Key Training Parameters (in `train.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | `swin_base_patch4_window7_224_in22k` | Swin-Base with 224px window |
| Batch size | 8 | Per GPU |
| Epochs | 400 | Total training epochs |
| Learning rate | 1e-4 | AdamW optimizer |
| AMP | Enabled | torch.cuda.amp |
| EMA | Enabled (decay=0.999, every 20 epochs) | Exponential Moving Average |
| Mixup | alpha=0.8, cutmix=1.0 | Label smoothing=0.1 |
| Input size | 224×224 | Resize + Normalize |

## Code Architecture

### Main Scripts
- **`train.py`** — Entry point. Defines model, data loaders, optimizer, training/validation loops, and generates loss/accuracy plots.
- **`test.py`** — Inference on `test/` images using `checkpoints/Swin/best.pth`. Uses CPU/CUDA automatically.
- **`makedata.py`** — Splits raw images from `COVID-TEST/` into `data/train/` and `data/val/` (80/20 stratified split).
- **`mean_std.py`** — Computes per-channel mean and std of training dataset for normalization.
- **`classification_report.py`** — (empty) intended for sklearn `classification_report` output.

### Modules
- **`ema.py`** — EMA helper class: `register()`, `update()`, `apply_shadow()`, `restore()`.
- **`models/`** — Alternative model architectures (RepVGG, RepVGGPlus, SE blocks), not used in current train.py config.

### Data Layout
```
data/
  train/    (training set)
    COVID19/    (428 images)
    NORMAL/     (1257 images)
    PNEUMONIA/  (3439 images)
  val/      (validation set)
    COVID19/    (138 images)
    NORMAL/     (318 images)
    PNEUMONIA/  (826 images)
```

### Output Structure
Training saves checkpoints and logs to `checkpoints/Swin/`:
- `best.pth` / `model_{epoch}_{acc}.pth` — model checkpoints (saved when val accuracy improves)
- `result.json` — per-epoch training/validation loss and accuracy
- `loss.png` / `acc.png` — training curves

## Notes

- `test.py` has a hardcoded 4-class tuple `('COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia')` that does not match the actual 3-class data. Update before running inference.
- The pretrained Swin-Base weight (`swin_base_patch4_window7_224_22k.pth`) is ~418MB and stored via Git LFS. `timm` auto-downloads it if missing when `pretrained=True`.
- Training removes and recreates `checkpoints/Swin/` on each run.

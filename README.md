# InceptionNet v1 - CIFAR-10

A from-scratch PyTorch reimplementation of **GoogLeNet / Inception v1** (Szegedy et al., 2014), with stem modifications to make it work on CIFAR-10's 32×32 images instead of the original ImageNet 224×224 input.

Achieved **86.45% validation accuracy** on CIFAR-10 after 35 epochs.

---

## Architecture

The original GoogLeNet stem (7×7 conv, stride 2 → MaxPool → ...) aggressively downsamples the spatial dimensions and is unsuitable for 32×32 inputs. The stem was redesigned as follows:

| Stage | Original (ImageNet) | This Repo (CIFAR-10) |
|---|---|---|
| Conv1 | 7×7, stride 2, pad 3 | 3×3, stride 1, pad 1 |
| Pool1 | MaxPool 3×3, stride 2 | MaxPool 2×2, stride 2 |
| Pool2 | MaxPool 3×3, stride 2 | MaxPool 2×2, stride 2 |
| Aux pool | AvgPool 5×5, stride 3 | AvgPool 2×2, stride 2 |

Everything after the stem - the 9 Inception blocks (3a/3b, 4a–4e, 5a/5b), two auxiliary classifiers, and the final FC head follows the original paper exactly.

### Inception Block

Each block runs four parallel branches and concatenates their outputs along the channel dimension:

```
Input
 ├─ 1×1 conv
 ├─ 1×1 conv  →  3×3 conv
 ├─ 1×1 conv  →  5×5 conv
 └─ 3×3 MaxPool  →  1×1 conv
              ↓
         Concat (dim=1)
```

### Auxiliary Classifiers

Two auxiliary classifiers are attached at `inception4a` (512 ch) and `inception4d` (528 ch) during training. Their loss is weighted at **0.3** and added to the main loss. They are disabled at inference.

---

## Results

| Metric | Value |
|---|---|
| Val Accuracy | **86.45%** |
| Val Loss | 0.4122 |
| Epochs | 35 |
| Params | ~6.8M |

---

## Project Structure

```
inceptionnet-from-scratch/
├── configs/
│   └── configs.yaml          # Hyperparameters and dataset paths
├── data/
│   ├── dataset/              # CIFAR-10 data (auto-downloaded by torchvision)
│   │   └── cifar-10-batches-py/
│   └── dataset.py            # Dataset / DataLoader construction
├── inference/
│   └── predict.py            # Load a checkpoint and run inference
├── models/                   # Saved model checkpoints (.pth)
├── notebooks/
│   └── InceptionNet.ipynb    # End-to-end training & evaluation notebook
├── tests/
│   └── test.py               # Unit tests for model components
├── training/
│   └── train.py              # Training loop (standalone script)
├── utils/                    # Shared helpers (logging, metrics, etc.)
├── writeups/
│   └── writings.md           # Notes, experiments, observations
├── main.py                   # Entry point — calls train or predict
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```
---

## Quickstart

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train (via notebook)

Open `notebooks/InceptionNet.ipynb` and run all cells. CIFAR-10 will be downloaded automatically to `data/dataset/` on the first run.

### Train (via script)

```bash
python main.py --mode train
```

### Inference

```bash
python main.py --mode predict --checkpoint models/inceptionnet_best.pth --image path/to/image.png
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Weight decay | 1e-4 |
| LR schedule | StepLR (step=8, γ=0.96) |
| Batch size | 64 |
| Epochs | 35 |
| Dropout (main) | 0.4 |
| Dropout (aux) | 0.7 |
| Aux loss weight | 0.3 |

### Data augmentation (train)

- RandomCrop 32×32 with padding 4
- RandomHorizontalFlip
- ColorJitter (brightness, contrast, saturation ±0.2)
- Normalize — mean `[0.4914, 0.4822, 0.4465]`, std `[0.2023, 0.1994, 0.2010]`

---

## Requirements

```
torch
torchvision
tensorflow        # used only for keras.utils.Progbar progress bar
pyyaml
```

> **Note:** `tensorflow` is imported solely for `tf.keras.utils.Progbar`. If you want to remove the TensorFlow dependency, it can be swapped for `tqdm` with minimal code change.

---

## Reference

> Szegedy, C., Liu, W., Jia, Y., et al. (2014). **Going Deeper with Convolutions.** *CVPR 2015.* [arXiv:1409.4842](https://arxiv.org/abs/1409.4842)

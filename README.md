# LoCA: Spatially-Aware Low-Rank Convolutional Adaptation

[Paper](https://arxiv.org/abs/2607.06918) · [PDF](https://arxiv.org/pdf/2607.06918)

This repository provides the PyTorch implementation and classification training code for **LoCA: Spatially-Aware Low-Rank Convolutional Adaptation of Vision Foundation Models**, accepted to **ECCV 2026**.

LoCA is a convolution-aware parameter-efficient fine-tuning method for vision foundation models. It decouples channel and spatial adaptation for convolutional kernels, preserving spatial priors while training a small number of adapter parameters.

The classification scaffold follows the organization of [Conv-Adapter](https://github.com/Hhhhhhao/Conv-Adapter): dataset builders, model builders, `engine.py`, `main.py`, and `utils.py`. The adapter implementation and LoCA-specific training paths are maintained here.

## Repository Contents

- `loca.py`: standalone LoCA module wrappers for linear and convolutional layers.
- `models/`: backbone, head, and tuning-module implementations.
- `datasets/`: FGVC, VTAB-1K, and ImageNet-style dataset builders.
- `configs/`: optional per-model optimizer defaults loaded by `main.py`.
- `scripts/train.sh`: configurable LoCA classification training entrypoint.
- `scripts/eval.sh`: configurable evaluation entrypoint.

## Supported Backbones

- ConvNeXt: `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`, `convnext_base_22k`
- ResNet: `resnet50`, `resnet101`, `resnet152`
- Swin Transformer: `swin_base_patch4_window7_224`, `swin_base_patch4_window7_224_in22k`
- DINOv3/timm wrappers: `dinov3_vit_base`, `dinov3_convnext_base`
- Mamba models: `mambaout_base`, `mambavision_tiny`, `mambavision_small`, `mambavision_base`

## Installation

```bash
pip install -r requirements.txt
```

MambaVision requires `mamba-ssm` and its CUDA-compatible dependencies. Install the version that matches your PyTorch/CUDA environment.

## Training

The default script runs LoCA on FGVC classification with ConvNeXt-base.

```bash
bash scripts/train.sh
```

Common overrides:

```bash
MODEL=mambavision_base DATA=fgvc DATASET=cars DATA_PATH=/home/dataset/fgvc/cars bash scripts/train.sh
MODEL=convnext_base DATA=vtab-1k DATASET=svhn DATA_PATH=/home/dataset/vtab-1k/svhn bash scripts/train.sh
MODEL=swin_base_patch4_window7_224_in22k GPUS=0,1 NUM_GPUS=2 bash scripts/train.sh
```

Important options:

- `MODEL`: backbone name.
- `DATA`: dataset family, such as `fgvc`, `vtab-1k`, or `imagenet-1k`.
- `DATASET`: dataset name inside the selected family.
- `DATA_PATH`: dataset root.
- `LR`, `WEIGHT_DECAY`, `EPOCHS`, `BATCH_SIZE`: optimizer and training controls.
- `LORA_TARGET`: JSON list or space-separated module names to adapt.
- `SV_LAYERS`: JSON list or single module path for singular-value logging.
- `SV_INTERVAL`: epoch interval for singular-value logging.

## Evaluation

```bash
RESUME=outputs/convnext_base/fgvc/cars/checkpoint-best.pth bash scripts/eval.sh
```

The evaluation script uses the same `MODEL`, `DATA`, `DATASET`, `DATA_PATH`, `LORA_TARGET`, and LoCA rank settings as training.

## Citation

```bibtex
@article{an2026loca,
  title={LoCA: Spatially-Aware Low-Rank Convolutional Adaptation of Vision Foundation Models},
  author={An, Sojung and Lee, Junha and You, Sujeong and Cho, Nam Ik and Kim, Donghyun},
  journal={arXiv preprint arXiv:2607.06918},
  year={2026}
}
```

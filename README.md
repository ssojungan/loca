<div align="center">

# LoCA

### Spatially-Aware Low-Rank Convolutional Adaptation of Vision Foundation Models

Sojung An · Junha Lee · Sujeong You · Nam Ik Cho · Donghyun Kim

**Accepted to ECCV 2026**

[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-365B9A)](https://link.springer.com/chapter/10.1007/978-3-032-37464-6_27)
[![Paper](https://img.shields.io/badge/Paper-Springer-00599C)](https://link.springer.com/chapter/10.1007/978-3-032-37464-6_27)
[![arXiv](https://img.shields.io/badge/arXiv-2607.06918-B31B1B)](https://arxiv.org/abs/2607.06918)

[Overview](#overview) · [Get started](#get-started) · [Backbones](#supported-backbones) · [Training](#training) · [Evaluation](#evaluation) · [Citation](#citation)

</div>

## Overview

**LoCA** is a parameter-efficient fine-tuning method for vision foundation models. It adapts convolutional kernels through two complementary paths: a low-rank update for cross-channel mixing and refinement of spatial bases derived from the pretrained kernels by singular value decomposition (SVD). This separation preserves spatial information that can be lost when a convolutional kernel is flattened into a single matrix for standard LoRA.

| Channel adaptation | Spatial adaptation | Practical use |
| :--- | :--- | :--- |
| Low-rank cross-channel mixing | SVD-derived spatial basis refinement | Fine-tune vision backbones with a small set of adapter parameters |

This repository provides the PyTorch LoCA implementation and classification training and evaluation scripts. The [ECCV 2026 paper](https://link.springer.com/chapter/10.1007/978-3-032-37464-6_27) also evaluates LoCA on domain-generalized semantic segmentation and generative benchmarks.

## Repository map

| Path | Purpose |
| :--- | :--- |
| [`loca.py`](loca.py) | Standalone LoCA wrappers for linear and convolutional layers |
| [`models/`](models) | Backbone, head, and tuning-module implementations |
| [`datasets/`](datasets) | FGVC, VTAB-1K, and ImageNet-style dataset builders |
| [`configs/`](configs) | Optional model-specific optimizer defaults |
| [`scripts/train.sh`](scripts/train.sh) | Classification training entry point |
| [`scripts/eval.sh`](scripts/eval.sh) | Checkpoint evaluation entry point |

## Supported backbones

| Family | Model names |
| :--- | :--- |
| ConvNeXt | `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`, `convnext_base_22k` |
| ResNet | `resnet50`, `resnet101`, `resnet152` |
| Swin Transformer | `swin_base_patch4_window7_224`, `swin_base_patch4_window7_224_in22k` |
| DINOv3 / timm | `dinov3_vit_base`, `dinov3_convnext_base` |
| Mamba | `mambaout_base`, `mambavision_tiny`, `mambavision_small`, `mambavision_base` |

## Get started

```bash
git clone https://github.com/ssojungan/loca.git
cd loca
python -m pip install -r requirements.txt
```

MambaVision models also require `mamba-ssm` and dependencies compatible with your PyTorch and CUDA installation.

## Training

The training script defaults to LoCA with `convnext_base` on the FGVC Cars dataset. Point `DATA_PATH` to your local dataset before running it:

```bash
DATA_PATH=/path/to/fgvc/cars bash scripts/train.sh
```

Select another supported model or dataset through environment variables:

```bash
MODEL=convnext_base DATA=vtab-1k DATASET=svhn \
  DATA_PATH=/path/to/vtab-1k/svhn bash scripts/train.sh

MODEL=mambavision_base DATA=fgvc DATASET=cars \
  DATA_PATH=/path/to/fgvc/cars bash scripts/train.sh
```

| Variable | Purpose | Script default |
| :--- | :--- | :--- |
| `MODEL` | Backbone name | `convnext_base` |
| `DATA`, `DATASET`, `DATA_PATH` | Dataset family, name, and location | `fgvc`, `cars`, local path |
| `LORA_RANK`, `LORA_ALPHA` | Adapter rank and scaling | `16`, `16.0` |
| `LORA_TARGET` | Modules to adapt; chosen by model family when unset | Model-specific |
| `EPOCHS`, `BATCH_SIZE`, `LR` | Training schedule | `100`, `64`, `0.001` |
| `OUTPUT_DIR` | Checkpoint and log directory | `outputs/$MODEL/$DATA/$DATASET` |
| `GPUS`, `NUM_GPUS` | Visible GPU IDs and process count | `0`, `1` |

For multi-GPU training, set `GPUS` and `NUM_GPUS` together. Optional singular-value logging is controlled by `SV_LAYERS` and `SV_INTERVAL` in [`scripts/train.sh`](scripts/train.sh).

## Evaluation

Use the same model, dataset, and adapter settings as training. `RESUME` points to the checkpoint to evaluate:

```bash
MODEL=convnext_base DATA=fgvc DATASET=cars \
  DATA_PATH=/path/to/fgvc/cars \
  RESUME=outputs/convnext_base/fgvc/cars/checkpoint-best.pth \
  bash scripts/eval.sh
```

## Citation

If you use LoCA in your research, please cite the ECCV 2026 paper:

```bibtex
@InProceedings{10.1007/978-3-032-37464-6_27,
  author    = {An, Sojung and Lee, Junha and You, Sujeong and Cho, Nam Ik and Kim, Donghyun},
  editor    = {Favaro, Paolo and Kukelova, Zuzana and Maki, Atsuto and Rohrbach, Anna and Schindler, Konrad and Tombari, Federico},
  title     = {LoCA: Spatially-Aware Low-Rank Convolutional Adaptation of Vision Foundation Models},
  booktitle = {Computer Vision -- ECCV 2026},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {500--519},
  isbn      = {978-3-032-37464-6},
  doi       = {10.1007/978-3-032-37464-6_27}
}
```

## Acknowledgments

The classification training scaffold follows [Conv-Adapter](https://github.com/Hhhhhhao/Conv-Adapter). The LoCA adapter implementation and LoCA-specific training paths are maintained in this repository.

## 💻 Environment

- **GPU:** NVIDIA H100 (80GB)
- All experiments and evaluations were conducted using AICA's NVIDIA H100 GPUs.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# LoCA

LoCA is a PyTorch implementation of Low-rank Convolutional Adaptation for adding lightweight trainable adapters to existing neural network modules.

This repository currently includes a standalone implementation in `loca.py`.

## Features

- LoCA configuration with `LoCAConfig`
- Model wrapper with `LoCAModel`
- LoRA-style adapters for `nn.Linear`
- LoCA adapters for `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`, and `nn.ConvTranspose2d`
- Optional SVD-based initialization
- Optional dynamic rank selection
- Target and exclude module filtering

## Requirements

- Python 3.9+
- PyTorch
- PEFT

Install the required packages in your own environment:

```bash
pip install torch peft
```

## Usage

```python
import torch.nn as nn

from loca import LoCAConfig, LoCAModel


model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
)

config = LoCAConfig(
    lora_rank=16,
    lora_alpha=16.0,
    target_modules=None,
    init_lora_weights="zero",
)

loca_model = LoCAModel(config, model)
```

Only LoCA/LoRA adapter parameters are marked trainable by default.

## Repository Status

This repository is starting with the core `loca.py` implementation.

More code coming soon.

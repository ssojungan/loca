from typing import Optional, Tuple, Union, List
from dataclasses import asdict, dataclass, field
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.utils.integrations import gather_params_ctx


@dataclass
class LoraconvConfig:
    type: str = field(default="loraco", metadata={"help": "Type of Low-rank Adaptation."})
    r: int = field(default=16, metadata={"help": "Lora rank"})
    lora_alpha: float = field(default=16.0, metadata={"help": "Lora alpha"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to replace with Lora."},
    )
    merge_weights: bool = field(default=False, metadata={"help": "Merge weights of the original model and the Lora model"})
    start_lora_idx: int = field(default=0, metadata={"help": "Index of the block from which to start applying LoRA"})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    channel_wise: bool = field(default=False, metadata={"help": "Use stage-specific rank and alpha values"})

    def __post_init__(self):
        self.peft_type = "LoRAConv"
        self.lora_rank = self.r


class LoraconvModel(nn.Module):
    def __init__(self, config: LoraconvConfig, model: nn.Module):
        super().__init__()
        self.peft_config = config
        self.model = model
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self._find_and_replace()
        self.mark_only_lora_as_trainable()
        self.forward = self.model.forward

    def _is_target_module(self, name: str) -> bool:
        if self.peft_config.target_modules is None:
            return True
        if isinstance(self.peft_config.target_modules, str):
            return bool(re.fullmatch(self.peft_config.target_modules, name))
        matched = any(name.endswith("." + t) or name.endswith(t) or (t in name) for t in self.peft_config.target_modules)
        # 매칭 안 돼도 이름에 "dw"가 있으면 학습 (dwconv 등)
        if not matched and "dw" in name:
            return True
        return matched

    def _extract_block_index(self, module_name: str) -> Optional[int]:
        match = re.search(r'blocks\.(\d+)\.', module_name)
        if match:
            return int(match.group(1))
        return None

    def _get_stage_rank_alpha(self, module_name: str) -> Tuple[int, float]:
        """
        Determine rank and alpha based on module stage when channel_wise is True.
        ConvNeXt: stem (downsample_layers.0) = 4; stages.0/1 = 8; stages.2 = 16; stages.3 = 8.
        ResNet: layer1/2 = 8; layer3 = 16; layer4 = 8 (same mapping).
        """
        if not self.peft_config.channel_wise:
            return self.peft_config.r, self.peft_config.lora_alpha
        
        name_lower = module_name.lower()
        
        # Stem / first conv
        if 'downsample_layers.0' in name_lower or 'stem' in name_lower:
            return 4, 4.0
        # ConvNeXt: stages.0, stages.1, ...
        if 'stages.0' in name_lower:
            return 16, 16.0
        elif 'stages.1' in name_lower:
            return 16, 16.0
        elif 'stages.2' in name_lower:
            return 16, 16.0
        elif 'stages.3' in name_lower:
            return 16, 16.0
        
        # Default to config values if stage is not identified
        return self.peft_config.r, self.peft_config.lora_alpha

    def _find_and_replace(self):
        modules_to_replace = []
        for name, module in self.model.named_modules():
            if not self._is_target_module(name):
                continue

            parent, target_name = self._get_parent_module(name)
            new_module = None
            
            # Get stage-specific rank and alpha
            lora_rank, lora_alpha = self._get_stage_rank_alpha(name)

            if isinstance(module, nn.Linear):
                new_module = LoRALinear(
                    module,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    merge_weights=self.peft_config.merge_weights,
                )
            elif isinstance(module, nn.Conv2d):
                new_module = LoRAConv2d(
                    module,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    merge_weights=self.peft_config.merge_weights,
                )
            elif isinstance(module, nn.ConvTranspose2d):
                new_module = LoRAConvTranspose2d(
                    module,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    merge_weights=self.peft_config.merge_weights,
                )

            if new_module is not None:
                modules_to_replace.append((name, parent, target_name, new_module, module))

        for idx, (name, parent, target_name, new_module, old_module) in enumerate(modules_to_replace):
            self._replace_module(parent, target_name, new_module, old_module)
            with gather_params_ctx(new_module.weight):
                new_module.init()

    def _get_parent_module(self, module_name: str) -> Tuple[nn.Module, str]:
        names = module_name.split(".")
        parent = self.model
        for n in names[:-1]:
            parent = getattr(parent, n)
        return parent, names[-1]

    def _replace_module(self, parent_module: nn.Module, child_name: str, new_module: nn.Module, old_module: nn.Module):
        setattr(parent_module, child_name, new_module)

        if hasattr(old_module, "weight") and old_module.weight is not None:
            old_w = old_module.weight.detach()
            tgt = new_module.weight
            if old_w.shape == tgt.shape:
                tgt.data.copy_(old_w.to(device=tgt.device, dtype=tgt.dtype))
            elif old_w.dim() == 4 and tgt.dim() == 2:
                tgt.data.copy_(old_w.reshape(old_w.shape[0], -1).to(device=tgt.device, dtype=tgt.dtype))
            elif old_w.dim() == 2 and tgt.dim() == 4:
                tgt.data.copy_(old_w.reshape(tgt.shape).to(device=tgt.device, dtype=tgt.dtype))
            else:
                raise RuntimeError(f"weight shape mismatch: old={tuple(old_w.shape)} new={tuple(tgt.shape)}")

        if hasattr(old_module, "bias") and old_module.bias is not None:
            if hasattr(new_module, "bias") and new_module.bias is not None:
                new_module.bias.data.copy_(old_module.bias.detach().to(device=new_module.bias.device, dtype=new_module.bias.dtype))

    def mark_only_lora_as_trainable(self):
        trainable_keywords = ['lora_', 's_basis', 'u_dw', 'ch_scaler', 'mask_token', 'head']

        for n, p in self.model.named_parameters():
            if any(k in n for k in trainable_keywords):
                p.requires_grad = True
            else:
                p.requires_grad = False

        if self.peft_config.bias == "all":
            for n, p in self.model.named_parameters():
                block_index = self._extract_block_index(n)
                if block_index is not None and block_index >= self.peft_config.start_lora_idx:
                    if any(bias + "." in n for bias in self.peft_config.target_modules) and "bias" in n:
                        p.requires_grad = True
                    if "q_bias" in n or "v_bias" in n:
                        p.requires_grad = True
        elif self.peft_config.bias == "lora_only":
            for m in self.model.modules():
                if isinstance(m, (LoRALinear, LoRAConv2d, LoRAConvTranspose2d)):
                    if hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        cfg = {k: v for k, v in asdict(self.peft_config).items()}
        if inference:
            cfg["inference_mode"] = True
        return cfg


class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        merge_weights: bool = False,
    ):
        super().__init__()

        device = linear.weight.device
        dtype = linear.weight.dtype

        W = linear.weight.detach().clone().to(device=device, dtype=dtype)
        self.weight = nn.Parameter(W, requires_grad=False)
        self.bias = (
            nn.Parameter(linear.bias.detach().clone().to(device=device, dtype=dtype), requires_grad=False)
            if linear.bias is not None
            else None
        )

        out_f, in_f = W.shape
        r = min(int(lora_rank), int(min(in_f, out_f)))
        self.r = r
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.r) if self.r > 0 else 1.0

        self.lora_A = None
        self.lora_B = None

        if self.r > 0:
            self.lora_A = nn.Linear(in_f, self.r, bias=False, device=device, dtype=dtype)
            self.lora_B = nn.Linear(self.r, out_f, bias=False, device=device, dtype=dtype)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

        self.merge_weights = bool(merge_weights)
        self.merged = False

    @torch.no_grad()
    def init(self):
        pass

    def _delta_weight(self):
        if self.r <= 0:
            return 0
        return ((self.lora_B.weight @ self.lora_A.weight) * self.scaling).contiguous()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.merge_weights:
            if not mode and not self.merged:
                with torch.no_grad():
                    self.weight.add_(self._delta_weight())
                self.merged = True
            elif mode and self.merged:
                with torch.no_grad():
                    self.weight.sub_(self._delta_weight())
                self.merged = False
        return self

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            with torch.no_grad():
                self.weight.add_(self._delta_weight())
            self.merged = True
        return self

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.r > 0 and not (self.merge_weights and self.merged):
            z = x @ self.lora_A.weight.t()
            out = out + (z @ self.lora_B.weight.t()) * self.scaling
        return out


def init_spatial_basis_from_svd(W: torch.Tensor):
    out_c, in_c_g, kh, kw = W.shape
    spatial_dim = kh * kw
    device = W.device
    dtype = W.dtype

    W32 = W.detach().to(dtype=torch.float32)
    if torch.cuda.is_available():
        W32 = W32.to(device="cuda")

    W_flat = W32.reshape(-1, spatial_dim).contiguous()

    mean = W_flat.mean(dim=1, keepdim=True)
    # unbiased=False avoids "degrees of freedom <= 0" when spatial_dim==1 (e.g. 1x1 conv)
    std = W_flat.std(dim=1, keepdim=True, unbiased=False) + 1e-6
    W_norm = (W_flat - mean) / std

    C_sp = W_norm.T @ W_norm
    U, _, _ = torch.linalg.svd(C_sp, full_matrices=False)

    return U.T.reshape(spatial_dim, kh, kw).contiguous().to(device=device, dtype=dtype)


class LoRAConv2d(nn.Module):
    def __init__(
        self,
        conv: nn.Conv2d,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        merge_weights: bool = False,
    ):
        super().__init__()

        device = conv.weight.device
        dtype = conv.weight.dtype

        W = conv.weight.detach().clone().to(device=device, dtype=dtype)
        out_c, in_c, kh, kw = W.shape
        Wf = W.view(out_c, -1).contiguous()

        self.weight = nn.Parameter(Wf.clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(conv.bias.detach().clone().to(device=device, dtype=dtype), requires_grad=False)
            if conv.bias is not None
            else None
        )

        self.r = min(lora_rank, Wf.shape[0])
        self.scaling = (lora_alpha / self.r) if self.r > 0 else 1.0

        self.lora_a = nn.Parameter(torch.zeros(self.r, Wf.shape[1], device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(Wf.shape[0], self.r, device=device, dtype=dtype))
        
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

        self.weight_shape = (out_c, in_c, kh, kw)
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.merge_weights = merge_weights
        self.merged = False

        self.spatial_rank = kh * kw

        self.s_basis = nn.Parameter(
            init_spatial_basis_from_svd(W).to(device=device, dtype=dtype)
        )
        
        self.diag = min(out_c, in_c)
        self.u_dw = nn.Parameter(torch.zeros(self.diag, self.spatial_rank, device=device, dtype=dtype))

        self.use_weight_std = conv.__class__.__name__ == "StdConv2d"

    @torch.no_grad()
    def init(self):
        pass

    def _get_delta_sp_diag(self):
        return (self.u_dw @ self.s_basis.view(self.spatial_rank, -1)).contiguous()

    def _delta_weight(self):
        dW = ((self.lora_b @ self.lora_a) * self.scaling).contiguous()
        
        out_c, in_c, kh, kw = self.weight_shape
        delta_sp_flat = self._get_delta_sp_diag()

        dW_view = dW.view(out_c, in_c, kh * kw)
        dW_view.diagonal(dim1=0, dim2=1).add_(delta_sp_flat.T)
        
        return dW.contiguous()

    def _weight_eff_flat(self):
        W = self.weight
        if not (self.merge_weights and self.merged):
            delta_ch = ((self.lora_b @ self.lora_a) * self.scaling)
            delta_sp = self._get_delta_sp_diag()
            
            W = W + delta_ch
            
            out_c, in_c, kh, kw = self.weight_shape
            W_view = W.view(out_c, in_c, kh * kw)
            
            W_view.diagonal(dim1=0, dim2=1).add_(delta_sp.T)
            W = W_view.reshape(out_c, -1)
            
        return W.contiguous()

    def train(self, mode=True):
        super().train(mode)
        if self.merge_weights:
            if not mode and not self.merged:
                self.weight.add_(self._delta_weight())
                self.merged = True
            elif mode and self.merged:
                self.weight.sub_(self._delta_weight())
                self.merged = False
        return self

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            self.weight.add_(self._delta_weight())
            self.merged = True
        return self

    def forward(self, x):
        W = self._weight_eff_flat().reshape(self.weight_shape).contiguous()
        if self.use_weight_std:
            v, m = torch.var_mean(W, dim=[1, 2, 3], keepdim=True, unbiased=False)
            W = (W - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(
            x, W, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class LoRAConvTranspose2d(nn.Module):
    def __init__(
        self,
        conv: nn.ConvTranspose2d,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        merge_weights: bool = False,
    ):
        super().__init__()

        device = conv.weight.device
        dtype = conv.weight.dtype

        W = conv.weight.detach().clone().to(device=device, dtype=dtype)
        cin, cout_g, kh, kw = W.shape
        Wf = W.view(cin, -1).contiguous()

        self.weight = nn.Parameter(Wf.clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(conv.bias.detach().clone().to(device=device, dtype=dtype), requires_grad=False)
            if conv.bias is not None
            else None
        )

        self.r = min(lora_rank, Wf.shape[0])
        self.scaling = (lora_alpha / self.r) if self.r > 0 else 1.0

        self.lora_a = nn.Parameter(torch.zeros(self.r, Wf.shape[1], device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(Wf.shape[0], self.r, device=device, dtype=dtype))
        
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

        self.weight_shape = W.shape
        self.stride = conv.stride
        self.padding = conv.padding
        self.output_padding = conv.output_padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.merge_weights = merge_weights
        self.merged = False

        self.spatial_rank = kh * kw

        self.s_basis = nn.Parameter(
            init_spatial_basis_from_svd(W).to(device=device, dtype=dtype)
        )
        
        self.diag = min(cin, cout_g)
        self.u_dw = nn.Parameter(torch.zeros(self.diag, self.spatial_rank, device=device, dtype=dtype))

    @torch.no_grad()
    def init(self):
        pass

    def _get_delta_sp_diag(self):
        return (self.u_dw @ self.s_basis.view(self.spatial_rank, -1)).contiguous()

    def _delta_weight(self):
        dW = ((self.lora_b @ self.lora_a) * self.scaling).contiguous()
        
        cin, cout_g, kh, kw = self.weight_shape
        delta_sp_flat = self._get_delta_sp_diag()

        dW_view = dW.view(cin, cout_g, kh * kw)
        dW_view.diagonal(dim1=0, dim2=1).add_(delta_sp_flat.T)
        
        return dW.contiguous()

    def _weight_eff_flat(self):
        W = self.weight
        if not (self.merge_weights and self.merged):
            delta_ch = ((self.lora_b @ self.lora_a) * self.scaling)
            delta_sp = self._get_delta_sp_diag()
            
            W = W + delta_ch
            
            cin, cout_g, kh, kw = self.weight_shape
            W_view = W.view(cin, cout_g, kh * kw)
            
            W_view.diagonal(dim1=0, dim2=1).add_(delta_sp.T)
            W = W_view.reshape(cin, -1)
            
        return W.contiguous()

    def train(self, mode=True):
        super().train(mode)
        if self.merge_weights:
            if not mode and not self.merged:
                self.weight.add_(self._delta_weight())
                self.merged = True
            elif mode and self.merged:
                self.weight.sub_(self._delta_weight())
                self.merged = False
        return self

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            self.weight.add_(self._delta_weight())
            self.merged = True
        return self

    def forward(self, x):
        W = self._weight_eff_flat().reshape(self.weight_shape).contiguous()
        return F.conv_transpose2d(
            x, W, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )
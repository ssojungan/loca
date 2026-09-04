from typing import Optional, Tuple, Union, List
from dataclasses import asdict, dataclass, field
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.utils.integrations import gather_params_ctx


@dataclass
class LoCAConfig:
    type: str = field(default="loca", metadata={"help": "Low-rank Convolutional Adaptation."})
    lora_rank: int = field(default=16, metadata={"help": "Lora rank"})
    lora_alpha: float = field(default=16.0, metadata={"help": "Lora alpha"})
    linear_lora_rank: Optional[int] = field(default=None, metadata={"help": "Optional LoRA rank override for linear layers."})
    linear_lora_alpha: Optional[float] = field(default=None, metadata={"help": "Optional LoRA alpha override for linear layers."})
    conv_lora_rank: Optional[int] = field(default=None, metadata={"help": "Optional LoRA rank override for convolution layers."})
    conv_lora_alpha: Optional[float] = field(default=None, metadata={"help": "Optional LoRA alpha override for convolution layers."})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to replace with Lora."},
    )
    exclude_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from LoCA."},
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={"help": "Layer indexes to transform; only applies when target_modules is a list."},
    )
    layers_pattern: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "Layer pattern name(s) used with layers_to_transform."},
    )
    merge_weights: bool = field(default=False, metadata={"help": "Merge weights of the original model and the Lora model"})
    start_lora_idx: int = field(default=0, metadata={"help": "Index of the block from which to start applying LoRA"})
    bias: Optional[str] = field(default=None, metadata={"help": "Bias type for Lora. Can be None, 'all' or 'lora_only'"})
    init_lora_weights: str = field(default="zero", metadata={"help": "Weight initialization method. Can be 'svd' or 'normal'"})
    dynamic_rank: bool = field(default=False, metadata={"help": "Use dynamic rank"})

    def __post_init__(self):
        self.peft_type = "LoCA"


class LoCAModel(nn.Module):
    def __init__(self, config: LoCAConfig, model: nn.Module):
        super().__init__()
        self.peft_config = config
        self.model = model

        self._find_and_replace()
        self.mark_only_lora_as_trainable()
        self.forward = self.model.forward

    def save_pretrained(self, *args, **kwargs):
        return self.model.save_pretrained(*args, **kwargs)

    def _is_target_module(self, name: str) -> bool:
        if self.peft_config.exclude_modules:
            if isinstance(self.peft_config.exclude_modules, str):
                if re.fullmatch(self.peft_config.exclude_modules, name):
                    return False
            elif name in self.peft_config.exclude_modules:
                return False
            elif any(name.endswith(f".{exclude_key}") for exclude_key in self.peft_config.exclude_modules):
                return False

        if self.peft_config.target_modules is None:
            return True

        if isinstance(self.peft_config.target_modules, str):
            return bool(re.fullmatch(self.peft_config.target_modules, name))

        if name in self.peft_config.target_modules:
            target_module_found = True
        else:
            target_module_found = any(name.endswith(f".{target_key}") for target_key in self.peft_config.target_modules)

        layer_indexes = getattr(self.peft_config, "layers_to_transform", None)
        layers_pattern = getattr(self.peft_config, "layers_pattern", None)
        is_using_layer_indexes = layer_indexes is not None and (
            len(layer_indexes) != 0 if isinstance(layer_indexes, list) else True
        )
        if is_using_layer_indexes and target_module_found:
            layer_index = None
            if layers_pattern is None or len(layers_pattern) == 0:
                layer_index = re.match(r".*\.[^.]*\.(\d+)\.", name)
            else:
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern
                for pattern in layers_pattern:
                    layer_index = re.match(rf".*\.{pattern}\.(\d+)\.", name)
                    if layer_index is not None:
                        break

            if layer_index is None:
                target_module_found = False
            else:
                layer_index = int(layer_index.group(1))
                if isinstance(layer_indexes, int):
                    target_module_found = layer_index == layer_indexes
                else:
                    target_module_found = layer_index in layer_indexes

        return target_module_found

    def _find_and_replace(self):
        modules_to_replace = []
        for name, module in self.model.named_modules():
            if not self._is_target_module(name):
                continue

            parent, target_name = self._get_parent_module(name)
            lora_rank, lora_alpha = self._resolve_rank_alpha(module)

            if self.peft_config.dynamic_rank:
                if isinstance(module, nn.Linear):
                    out_channels = module.out_features
                elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
                    out_channels = module.out_channels
                else:
                    continue
                lora_rank = self.get_dynamic_rank(out_channels)
                lora_alpha = float(lora_rank)

            new_module = None

            if isinstance(module, nn.Linear):
                new_module = LoRALinear(
                    module,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    merge_weights=self.peft_config.merge_weights,
                    init_lora_weights=self.peft_config.init_lora_weights,
                )
            elif isinstance(module, nn.Conv1d):
                new_module = LoRAConv1d(
                    module,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    merge_weights=self.peft_config.merge_weights,
                    init_lora_weights=self.peft_config.init_lora_weights,
                )
            elif isinstance(module, nn.Conv2d):
                new_module = LoRAConv2d(
                    module,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    merge_weights=self.peft_config.merge_weights,
                    init_lora_weights=self.peft_config.init_lora_weights,
                )
            elif isinstance(module, nn.Conv3d):
                new_module = LoRAConv3d(
                    module,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    merge_weights=self.peft_config.merge_weights,
                    init_lora_weights=self.peft_config.init_lora_weights,
                )
            elif isinstance(module, nn.ConvTranspose2d):
                new_module = LoRAConvTranspose2d(
                    module,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    merge_weights=self.peft_config.merge_weights,
                    init_lora_weights=self.peft_config.init_lora_weights,
                )

            if new_module is not None:
                modules_to_replace.append((name, parent, target_name, new_module, module))

        for _, parent, target_name, new_module, old_module in modules_to_replace:
            self._replace_module(parent, target_name, new_module, old_module)
            with gather_params_ctx(new_module.weight):
                new_module.init()

    def _resolve_rank_alpha(self, module: nn.Module) -> Tuple[int, float]:
        if isinstance(module, nn.Linear):
            rank = self.peft_config.linear_lora_rank
            alpha = self.peft_config.linear_lora_alpha
        else:
            rank = self.peft_config.conv_lora_rank
            alpha = self.peft_config.conv_lora_alpha

        if rank is None:
            rank = self.peft_config.lora_rank
        if alpha is None:
            alpha = self.peft_config.lora_alpha
        return int(rank), float(alpha)

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
            elif old_w.dim() > 2 and tgt.dim() == 2:
                tgt.data.copy_(old_w.reshape(old_w.shape[0], -1).to(device=tgt.device, dtype=tgt.dtype))
            elif old_w.dim() == 2 and tgt.dim() > 2:
                tgt.data.copy_(old_w.reshape(tgt.shape).to(device=tgt.device, dtype=tgt.dtype))
            else:
                raise RuntimeError(f"weight shape mismatch: old={tuple(old_w.shape)} new={tuple(tgt.shape)}")

        if hasattr(old_module, "bias") and old_module.bias is not None:
            if hasattr(new_module, "bias") and new_module.bias is not None:
                new_module.bias.data.copy_(old_module.bias.detach().to(device=new_module.bias.device, dtype=new_module.bias.dtype))

    def mark_only_lora_as_trainable(self):
        trainable_keywords = ["lora_", "s_basis", "u_dw", "head"]
        for n, p in self.model.named_parameters():
            p.requires_grad = any(k in n for k in trainable_keywords)

        bias_mode = self.peft_config.bias
        if bias_mode in (None, "none"):
            return
        if bias_mode == "all":
            for n, p in self.model.named_parameters():
                if "bias" in n:
                    p.requires_grad = True
            return
        if bias_mode == "lora_only":
            loca_layers = (LoRALinear, LoRAConv1d, LoRAConv2d, LoRAConv3d, LoRAConvTranspose2d)
            for module in self.model.modules():
                if isinstance(module, loca_layers) and getattr(module, "bias", None) is not None:
                    module.bias.requires_grad = True
            return
        raise NotImplementedError(f"Requested bias mode `{bias_mode}` is not supported.")

    def get_dynamic_rank(self, out_channels: int) -> int:
        r = int(max(1, out_channels) * 0.03)
        r = 2 ** (r - 1).bit_length()
        return max(4, min(r, 32))

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
        init_lora_weights: str = "zero",
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
        self.init_lora_weights = str(init_lora_weights).lower()

    @torch.no_grad()
    def init(self):
        if self.r <= 0 or self.init_lora_weights != "svd":
            return

        weight = self.weight
        dtype = weight.dtype
        device = weight.device
        w = weight.to(torch.float32)
        if torch.cuda.is_available():
            w = w.cuda()

        U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        Ur = U[:, -self.r :]
        Sr = S[-self.r :] / self.scaling
        Vhr = Vh[-self.r :, :]
        sqrtSr = torch.sqrt(Sr)

        lora_A = torch.diag(sqrtSr) @ Vhr
        lora_B = Ur @ torch.diag(sqrtSr)
        self.lora_A.weight.copy_(lora_A.to(dtype=dtype, device=device))
        self.lora_B.weight.copy_(lora_B.to(dtype=dtype, device=device))

        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        self.weight.sub_(delta)
        self.merged = False

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
    std = W_flat.std(dim=1, keepdim=True, unbiased=False) + 1e-6
    W_norm = (W_flat - mean) / std
    C_sp = W_norm.T @ W_norm
    U, _, _ = torch.linalg.svd(C_sp, full_matrices=False)
    return U.T.reshape(spatial_dim, kh, kw).contiguous().to(device=device, dtype=dtype)


def init_spatial_basis_from_svd_1d(W: torch.Tensor):
    out_c, in_c_g, k = W.shape
    spatial_dim = k
    device = W.device
    dtype = W.dtype

    W32 = W.detach().to(dtype=torch.float32)
    if torch.cuda.is_available():
        W32 = W32.to(device="cuda")

    W_flat = W32.reshape(-1, spatial_dim).contiguous()
    mean = W_flat.mean(dim=1, keepdim=True)
    std = W_flat.std(dim=1, keepdim=True, unbiased=False) + 1e-6
    W_norm = (W_flat - mean) / std
    C_sp = W_norm.T @ W_norm
    U, _, _ = torch.linalg.svd(C_sp, full_matrices=False)
    return U.T.reshape(spatial_dim, k).contiguous().to(device=device, dtype=dtype)


def init_spatial_basis_from_svd_3d(W: torch.Tensor):
    out_c, in_c_g, kd, kh, kw = W.shape
    spatial_dim = kd * kh * kw
    device = W.device
    dtype = W.dtype

    W32 = W.detach().to(dtype=torch.float32)
    if torch.cuda.is_available():
        W32 = W32.to(device="cuda")

    W_flat = W32.reshape(-1, spatial_dim).contiguous()
    mean = W_flat.mean(dim=1, keepdim=True)
    std = W_flat.std(dim=1, keepdim=True, unbiased=False) + 1e-6
    W_norm = (W_flat - mean) / std
    C_sp = W_norm.T @ W_norm
    U, _, _ = torch.linalg.svd(C_sp, full_matrices=False)
    return U.T.reshape(spatial_dim, kd, kh, kw).contiguous().to(device=device, dtype=dtype)


class LoRAConv2d(nn.Module):
    def __init__(
        self,
        conv: nn.Conv2d,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        merge_weights: bool = False,
        init_lora_weights: str = "zero",
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

        self.r = min(int(lora_rank), Wf.shape[0], Wf.shape[1])
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
        self.init_lora_weights = str(init_lora_weights).lower()
        self.use_weight_std = False
        self.spatial_rank = kh * kw
        self.s_basis = nn.Parameter(init_spatial_basis_from_svd(W).to(device=device, dtype=dtype))
        self.diag = min(out_c, in_c)
        self.u_dw = nn.Parameter(torch.zeros(self.diag, self.spatial_rank, device=device, dtype=dtype))

    @torch.no_grad()
    def init(self):
        if self.r == 0 or self.init_lora_weights != "svd":
            return
        W = self.weight
        W32 = W.to(dtype=torch.float32)
        U, S, Vt = torch.linalg.svd(W32, full_matrices=False)
        Ur = U[:, -self.r :]
        Sr = S[-self.r :] / self.scaling
        Vhr = Vt[-self.r :, :]
        sqrtSr = torch.sqrt(Sr)
        A = torch.diag(sqrtSr) @ Vhr
        B = Ur @ torch.diag(sqrtSr)
        self.lora_a.copy_(A.to(dtype=W.dtype).contiguous())
        self.lora_b.copy_(B.to(dtype=W.dtype).contiguous())
        self.weight.sub_(self.scaling * (self.lora_b @ self.lora_a).contiguous())
        self.merged = False

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


class LoRAConv1d(nn.Module):
    def __init__(
        self,
        conv: nn.Conv1d,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        merge_weights: bool = False,
        init_lora_weights: str = "zero",
    ):
        super().__init__()

        device = conv.weight.device
        dtype = conv.weight.dtype
        W = conv.weight.detach().clone().to(device=device, dtype=dtype)
        out_c, in_c, k = W.shape
        Wf = W.view(out_c, -1).contiguous()

        self.weight = nn.Parameter(Wf.clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(conv.bias.detach().clone().to(device=device, dtype=dtype), requires_grad=False)
            if conv.bias is not None
            else None
        )

        self.r = min(int(lora_rank), Wf.shape[0], Wf.shape[1])
        self.scaling = (lora_alpha / self.r) if self.r > 0 else 1.0
        self.lora_a = nn.Parameter(torch.zeros(self.r, Wf.shape[1], device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(Wf.shape[0], self.r, device=device, dtype=dtype))
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

        self.weight_shape = (out_c, in_c, k)
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.merge_weights = merge_weights
        self.merged = False
        self.init_lora_weights = str(init_lora_weights).lower()
        self.spatial_rank = k
        self.s_basis = nn.Parameter(init_spatial_basis_from_svd_1d(W).to(device=device, dtype=dtype))
        self.diag = min(out_c, in_c)
        self.u_dw = nn.Parameter(torch.zeros(self.diag, self.spatial_rank, device=device, dtype=dtype))

    @torch.no_grad()
    def init(self):
        if self.r == 0 or self.init_lora_weights != "svd":
            return
        W = self.weight
        W32 = W.to(dtype=torch.float32)
        U, S, Vt = torch.linalg.svd(W32, full_matrices=False)
        Ur = U[:, -self.r :]
        Sr = S[-self.r :] / self.scaling
        Vhr = Vt[-self.r :, :]
        sqrtSr = torch.sqrt(Sr)
        A = torch.diag(sqrtSr) @ Vhr
        B = Ur @ torch.diag(sqrtSr)
        self.lora_a.copy_(A.to(dtype=W.dtype).contiguous())
        self.lora_b.copy_(B.to(dtype=W.dtype).contiguous())
        self.weight.sub_(self.scaling * (self.lora_b @ self.lora_a).contiguous())
        self.merged = False

    def _get_delta_sp_diag(self):
        return (self.u_dw @ self.s_basis.view(self.spatial_rank, -1)).contiguous()

    def _delta_weight(self):
        dW = ((self.lora_b @ self.lora_a) * self.scaling).contiguous()
        out_c, in_c, k = self.weight_shape
        delta_sp_flat = self._get_delta_sp_diag()
        dW_view = dW.view(out_c, in_c, k)
        dW_view.diagonal(dim1=0, dim2=1).add_(delta_sp_flat.T)
        return dW.contiguous()

    def _weight_eff_flat(self):
        W = self.weight
        if not (self.merge_weights and self.merged):
            delta_ch = ((self.lora_b @ self.lora_a) * self.scaling)
            delta_sp = self._get_delta_sp_diag()
            W = W + delta_ch
            out_c, in_c, k = self.weight_shape
            W_view = W.view(out_c, in_c, k)
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
        return F.conv1d(
            x, W, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class LoRAConv3d(nn.Module):
    def __init__(
        self,
        conv: nn.Conv3d,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        merge_weights: bool = False,
        init_lora_weights: str = "zero",
    ):
        super().__init__()

        device = conv.weight.device
        dtype = conv.weight.dtype
        W = conv.weight.detach().clone().to(device=device, dtype=dtype)
        out_c, in_c, kd, kh, kw = W.shape
        Wf = W.view(out_c, -1).contiguous()

        self.weight = nn.Parameter(Wf.clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(conv.bias.detach().clone().to(device=device, dtype=dtype), requires_grad=False)
            if conv.bias is not None
            else None
        )

        self.r = min(int(lora_rank), Wf.shape[0], Wf.shape[1])
        self.scaling = (lora_alpha / self.r) if self.r > 0 else 1.0
        self.lora_a = nn.Parameter(torch.zeros(self.r, Wf.shape[1], device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(Wf.shape[0], self.r, device=device, dtype=dtype))
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

        self.weight_shape = (out_c, in_c, kd, kh, kw)
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.merge_weights = merge_weights
        self.merged = False
        self.init_lora_weights = str(init_lora_weights).lower()
        self.spatial_rank = kd * kh * kw
        self.s_basis = nn.Parameter(init_spatial_basis_from_svd_3d(W).to(device=device, dtype=dtype))
        self.diag = min(out_c, in_c)
        self.u_dw = nn.Parameter(torch.zeros(self.diag, self.spatial_rank, device=device, dtype=dtype))

    @torch.no_grad()
    def init(self):
        if self.r == 0 or self.init_lora_weights != "svd":
            return
        W = self.weight
        W32 = W.to(dtype=torch.float32)
        U, S, Vt = torch.linalg.svd(W32, full_matrices=False)
        Ur = U[:, -self.r :]
        Sr = S[-self.r :] / self.scaling
        Vhr = Vt[-self.r :, :]
        sqrtSr = torch.sqrt(Sr)
        A = torch.diag(sqrtSr) @ Vhr
        B = Ur @ torch.diag(sqrtSr)
        self.lora_a.copy_(A.to(dtype=W.dtype).contiguous())
        self.lora_b.copy_(B.to(dtype=W.dtype).contiguous())
        self.weight.sub_(self.scaling * (self.lora_b @ self.lora_a).contiguous())
        self.merged = False

    def _get_delta_sp_diag(self):
        return (self.u_dw @ self.s_basis.view(self.spatial_rank, -1)).contiguous()

    def _delta_weight(self):
        dW = ((self.lora_b @ self.lora_a) * self.scaling).contiguous()
        out_c, in_c, kd, kh, kw = self.weight_shape
        delta_sp_flat = self._get_delta_sp_diag()
        dW_view = dW.view(out_c, in_c, kd * kh * kw)
        dW_view.diagonal(dim1=0, dim2=1).add_(delta_sp_flat.T)
        return dW.contiguous()

    def _weight_eff_flat(self):
        W = self.weight
        if not (self.merge_weights and self.merged):
            delta_ch = ((self.lora_b @ self.lora_a) * self.scaling)
            delta_sp = self._get_delta_sp_diag()
            W = W + delta_ch
            out_c, in_c, kd, kh, kw = self.weight_shape
            W_view = W.view(out_c, in_c, kd * kh * kw)
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
        return F.conv3d(
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
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        merge_weights: bool = False,
        init_lora_weights: str = "zero",
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

        self.r = min(int(lora_rank), Wf.shape[0], Wf.shape[1])
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
        self.init_lora_weights = str(init_lora_weights).lower()
        self.spatial_rank = kh * kw
        self.s_basis = nn.Parameter(init_spatial_basis_from_svd(W).to(device=device, dtype=dtype))
        self.diag = min(cin, cout_g)
        self.u_dw = nn.Parameter(torch.zeros(self.diag, self.spatial_rank, device=device, dtype=dtype))

    @torch.no_grad()
    def init(self):
        if self.r == 0 or self.init_lora_weights != "svd":
            return
        W = self.weight
        W32 = W.to(dtype=torch.float32)
        U, S, Vt = torch.linalg.svd(W32, full_matrices=False)
        Ur = U[:, -self.r :]
        Sr = S[-self.r :] / self.scaling
        Vhr = Vt[-self.r :, :]
        sqrtSr = torch.sqrt(Sr)
        A = torch.diag(sqrtSr) @ Vhr
        B = Ur @ torch.diag(sqrtSr)
        self.lora_a.copy_(A.to(dtype=W.dtype).contiguous())
        self.lora_b.copy_(B.to(dtype=W.dtype).contiguous())
        self.weight.sub_(self.scaling * (self.lora_b @ self.lora_a).contiguous())
        self.merged = False

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

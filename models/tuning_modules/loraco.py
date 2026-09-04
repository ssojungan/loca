from typing import Optional, Tuple, Union, List
from dataclasses import asdict, dataclass, field
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.utils.integrations import gather_params_ctx


def make_contiguous(grad):
    if grad is None:
        return None

    if grad.is_contiguous():
        return grad.reshape(-1).view(grad.shape)
    return grad.contiguous()


@dataclass
class LoracoConfig:
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

    def __post_init__(self):
        self.peft_type = "LoRACo"
        self.lora_rank = self.r


class LoracoModel(nn.Module):
    def __init__(self, config: LoracoConfig, model: nn.Module):
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
        return any(name.endswith("." + t) or name.endswith(t) or (t in name) for t in self.peft_config.target_modules)

    def _extract_block_index(self, module_name: str) -> Optional[int]:
        match = re.search(r'blocks\.(\d+)\.', module_name)
        if match:
            return int(match.group(1))
        return None

    def _find_and_replace(self):
        modules_to_replace = []
        for name, module in self.model.named_modules():
            if "head" in name or "patch_embed" in name and not self._is_target_module(name):
                continue

            parent, target_name = self._get_parent_module(name)
            new_module = None

            if isinstance(module, nn.Linear):
                new_module = LoRALinear(
                    module,
                    lora_rank=self.peft_config.r,
                    lora_alpha=self.peft_config.lora_alpha,
                    merge_weights=self.peft_config.merge_weights,
                )
            elif isinstance(module, nn.Conv2d):
                new_module = LoRAConv2d(
                    module,
                    lora_rank=self.peft_config.r,
                    lora_alpha=self.peft_config.lora_alpha,
                    merge_weights=self.peft_config.merge_weights,
                )
            elif isinstance(module, nn.ConvTranspose2d):
                new_module = LoRAConvTranspose2d(
                    module,
                    lora_rank=self.peft_config.r,
                    lora_alpha=self.peft_config.lora_alpha,
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
        trainable_keywords = ['lora_', 'mask_token', 'head']

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
        use_r_dw: bool = True,
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
        self.u_dw = None

        if self.r > 0:
            self.lora_A = nn.Linear(in_f, self.r, bias=False, device=device, dtype=dtype)
            self.lora_B = nn.Linear(self.r, out_f, bias=False, device=device, dtype=dtype)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            
            # Apply stride fix hooks
            self.lora_A.weight.register_hook(make_contiguous)
            self.lora_B.weight.register_hook(make_contiguous)

        self.merge_weights = bool(merge_weights)
        self.merged = False

    @torch.no_grad()
    def init(self):
        if self.r <= 0:
            return

        weight = self.weight
        dtype = weight.dtype
        device = weight.device

        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError(
                "Initialization requires float32, float16, or bfloat16."
            )

        w = weight.to(torch.float32)
        if torch.cuda.is_available():
            w = w.cuda()

        U, S, Vh = torch.linalg.svd(w, full_matrices=False)

        Ur = U[:, -self.r :]
        Sr = S[-self.r :]
        Vhr = Vh[-self.r :, :]

        Sr = Sr / self.scaling
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
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling

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

        self.weight = nn.Parameter(Wf.contiguous().clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(conv.bias.detach().clone().to(device=device, dtype=dtype), requires_grad=False)
            if conv.bias is not None
            else None
        )

        self.r = min(lora_rank, Wf.shape[0])
        self.scaling = (lora_alpha / self.r) if self.r > 0 else 1.0

        self.lora_a = nn.Parameter(torch.zeros(self.r, Wf.shape[1], device=device, dtype=dtype).contiguous().clone())
        self.lora_b = nn.Parameter(torch.zeros(Wf.shape[0], self.r, device=device, dtype=dtype).contiguous().clone())

        self.weight_shape = (out_c, in_c, kh, kw)
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.merge_weights = merge_weights
        self.merged = False

        self.lora_a.register_hook(make_contiguous)
        self.lora_b.register_hook(make_contiguous)

        self.use_weight_std = conv.__class__.__name__ == "StdConv2d"

    @torch.no_grad()
    def init(self):
        if self.r == 0:
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

    def _delta_weight(self):
        dW = torch.zeros_like(self.weight)
        if self.r > 0:
            dW = dW + (self.lora_b @ self.lora_a) * self.scaling

        return dW

    def _weight_eff_flat(self):
        if self.merge_weights and self.merged:
            return self.weight
        return self.weight + self._delta_weight()

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

        self.weight = nn.Parameter(Wf.contiguous().clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(conv.bias.detach().clone().to(device=device, dtype=dtype), requires_grad=False)
            if conv.bias is not None
            else None
        )

        self.r = min(lora_rank, Wf.shape[0])
        self.scaling = (lora_alpha / self.r) if self.r > 0 else 1.0

        self.lora_a = nn.Parameter(torch.zeros(self.r, Wf.shape[1], device=device, dtype=dtype).contiguous().clone())
        self.lora_b = nn.Parameter(torch.zeros(Wf.shape[0], self.r, device=device, dtype=dtype).contiguous().clone())

        self.weight_shape = W.shape
        self.stride = conv.stride
        self.padding = conv.padding
        self.output_padding = conv.output_padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.merge_weights = merge_weights
        self.merged = False

        self.lora_a.register_hook(make_contiguous)
        self.lora_b.register_hook(make_contiguous)

    @torch.no_grad()
    def init(self):
        if self.r == 0:
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

    def _delta_weight(self):
        dW = torch.zeros_like(self.weight)
        if self.r > 0:
            dW = dW + (self.lora_b @ self.lora_a) * self.scaling

        return dW

    def _weight_eff_flat(self):
        if self.merge_weights and self.merged:
            return self.weight
        return self.weight + self._delta_weight()

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
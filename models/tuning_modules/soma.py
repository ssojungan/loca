from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, lora_rank=8, lora_alpha=8.0, merge_weights=False):
        super().__init__()
        with torch.no_grad():
            W = linear.weight.detach().clone()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(lora_rank, S.shape[0])
            self.r = r
            self.scaling = lora_alpha / r if r > 0 else 1.0
            Ur = U[:, -r:]
            Sr = S[-r:] / self.scaling
            Vhr = Vt[-r:, :]
            lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
            lora_B = Ur @ torch.diag(torch.sqrt(Sr))
            weight_frozen = W - self.scaling * (lora_B @ lora_A)
            bias = linear.bias.detach().clone() if linear.bias is not None else None

        self.lora_a = nn.Parameter(lora_A)
        self.lora_b = nn.Parameter(lora_B)
        self.weight_frozen = nn.Parameter(weight_frozen, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.merge_weights = merge_weights
        self.merged = False

    def _weight(self):
        if self.r > 0:
            return self.weight_frozen + (self.lora_b @ self.lora_a) * self.scaling
        return self.weight_frozen

    def forward(self, x):
        W = self.weight_frozen if (self.merge_weights and self.merged) else self._weight()
        return F.linear(x, W, self.bias)

    def merge(self):
        if not self.merged:
            with torch.no_grad():
                self.weight_frozen.copy_(self._weight())
            self.merged = True


def apply_lora_svd(model, tuning_config):
    lora_rank = tuning_config['lora_rank']
    lora_alpha = tuning_config['lora_alpha']
    target_list = tuning_config['lora_target'].split() if isinstance(tuning_config['lora_target'], str) else tuning_config['lora_target']
    
    use_mlp = 'mlp' in target_list
    use_conv = 'conv' in target_list
    use_downsample = 'downsample' in target_list
    use_upsample = 'upsample' in target_list
    
    if not (use_mlp or use_conv or use_downsample or use_upsample):
        return model
    
    spatial_rank_config = tuning_config['spatial_rank']
    #use_spatial = (use_conv or use_downsample or use_upsample) and spatial_rank_config is not None
    use_spatial = True
    
    mlp_targets = []
    conv_replacements = []
    
    for name, module in model.named_modules():
        if 'stem' in name or 'head' in name:
            continue
        
        if use_mlp and isinstance(module, nn.Linear):
            mlp_targets.append((name, module))
        
        if isinstance(module, nn.Conv2d):
            is_conv_dw = module.groups == module.in_channels == module.out_channels
            is_downsample_conv = module.stride[0] != 1 or module.stride[1] != 1
            
            #if (use_conv and (is_conv_dw or not is_downsample_conv)) or (use_downsample and is_downsample_conv):
            conv_replacements.append((name, module, is_conv_dw))
        
        if isinstance(module, nn.ConvTranspose2d):
            conv_replacements.append((name, module, False))
    
    for name, old in mlp_targets:
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = parent._modules[part]
        parent._modules[parts[-1]] = LoRALinear(old, lora_rank, lora_alpha)
    
    for name, old, is_conv_dw in conv_replacements:
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = parent._modules[part]
        attr_name = parts[-1]
        
        if spatial_rank_config is None:
            spatial_rank = lora_rank
        else:
            channel_count = old.in_channels if isinstance(old, nn.ConvTranspose2d) else max(old.in_channels, old.out_channels)
            spatial_rank = max(1, channel_count // spatial_rank_config)
        
        if isinstance(old, nn.ConvTranspose2d):
            parent._modules[attr_name] = LoRAConvTranspose2d(old, lora_rank, lora_alpha, spatial_rank, use_spatial)
        else:
            parent._modules[attr_name] = LoRAConv2d(old, lora_rank, lora_alpha, spatial_rank, use_spatial)
    
    trainable_keywords = ['lora_a', 'lora_b']
    if use_spatial:
        trainable_keywords.extend(['s_basis', 'u_dw'])
    
    for p in model.parameters():
        p.requires_grad = False
    
    for n, p in model.named_parameters():
        if 'head' in n or any(k in n for k in trainable_keywords):
            p.requires_grad = True

    return model


def init_spatial_basis_from_svd(W, spatial_rank):
    out_c, in_c, kh, kw = W.shape
    spatial_dim = kh * kw

    dtype = W.dtype
    W32 = W.float()

    W_sp = W32.reshape(-1, spatial_dim)
    C_sp = W_sp.T @ W_sp
    U, _, _ = torch.linalg.svd(C_sp, full_matrices=False)

    basis = U[:, -spatial_rank:]
    return basis.T.reshape(spatial_rank, kh, kw).to(dtype=dtype)


class LoRAConv2d(nn.Module):
    def __init__(
        self,
        conv: nn.Conv2d,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        spatial_rank: int = 4,
        use_spatial: bool = False,
        merge_weights: bool = False,
    ):
        super().__init__()

        with torch.no_grad():
            W = conv.weight.detach().clone()
            cout, cin, kh, kw = W.shape
            Wf = W.view(cout, -1)
            U, S, Vt = torch.linalg.svd(Wf, full_matrices=False)
            r = min(lora_rank, S.shape[0])
            self.r = r
            self.scaling = (lora_alpha / r) if r > 0 else 1.0

            Ur = U[:, -r:]
            Sr = S[-r:] / self.scaling
            Vhr = Vt[-r:, :]
            lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
            lora_B = Ur @ torch.diag(torch.sqrt(Sr))
            weight_frozen = Wf - self.scaling * (lora_B @ lora_A)
            bias = conv.bias.detach().clone() if conv.bias is not None else None

        self.lora_a = nn.Parameter(lora_A.contiguous())
        self.lora_b = nn.Parameter(lora_B.contiguous())
        self.weight_frozen = nn.Parameter(weight_frozen, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

        self.weight_shape = (cout, cin, kh, kw)
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.merge_weights = merge_weights
        self.merged = False
        self.use_spatial = use_spatial

        spatial_dim = kh * kw
        spatial_rank = min(spatial_rank, spatial_dim)
        self.spatial_rank = spatial_rank

        if use_spatial:
            with torch.no_grad():
                s_init = init_spatial_basis_from_svd(W, spatial_rank)
            self.s_basis = nn.Parameter(s_init.contiguous())
            if cin == cout:
                self.u_dw = nn.Parameter(torch.zeros(spatial_rank))
            else:
                self.diag = min(cout, cin)
                self.u_dw = nn.Parameter(torch.zeros(self.diag, spatial_rank))

        self.use_weight_std = conv.__class__.__name__ == 'StdConv2d'

    def _delta_ch(self):
        if self.r == 0:
            return torch.zeros_like(self.weight_frozen)
        return (self.lora_b @ self.lora_a) * self.scaling

    def _delta_sp(self):
        if not self.use_spatial:
            return 0

        cout, cin, kh, kw = self.weight_shape

        if cin == cout:
            dW = torch.einsum("r, rxy -> xy", self.u_dw, self.s_basis).contiguous()
            dW_sp = torch.zeros(cout, cin, kh, kw, device=dW.device, dtype=dW.dtype)
            idx = torch.arange(cout, device=dW.device)
            dW_sp[idx, idx, :, :] = dW
            return dW_sp
        else:
            diag = self.diag
            dW = torch.einsum("ir, rxy -> ixy", self.u_dw, self.s_basis).contiguous()
            dW_sp = torch.zeros(cout, cin, kh, kw, device=dW.device, dtype=dW.dtype)
            idx = torch.arange(diag, device=dW.device)
            dW_sp[idx, idx, :, :] = dW
            return dW_sp

    def _weight_eff_flat(self):
        W = self.weight_frozen
        if not (self.merge_weights and self.merged):
            W = W + self._delta_ch()
            if self.use_spatial:
                W = W + self._delta_sp().reshape(W.shape[0], -1).contiguous()
        return W

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

    def merge(self):
        if not self.merged:
            with torch.no_grad():
                self.weight_frozen.copy_(self._weight_eff_flat())
            self.merged = True


class LoRAConvTranspose2d(nn.Module):
    def __init__(
        self,
        conv: nn.ConvTranspose2d,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        spatial_rank: int = 4,
        use_spatial: bool = False,
        merge_weights: bool = False,
    ):
        super().__init__()

        with torch.no_grad():
            W = conv.weight.detach().clone()
            cin, cout, kh, kw = W.shape
            Wf = W.permute(1, 0, 2, 3).contiguous().view(cout, -1)
            U, S, Vt = torch.linalg.svd(Wf, full_matrices=False)
            r = min(lora_rank, S.shape[0])
            self.r = r
            self.scaling = (lora_alpha / r) if r > 0 else 1.0

            Ur = U[:, -r:]
            Sr = S[-r:] / self.scaling
            Vhr = Vt[-r:, :]
            lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
            lora_B = Ur @ torch.diag(torch.sqrt(Sr))
            weight_frozen = Wf - self.scaling * (lora_B @ lora_A)
            bias = conv.bias.detach().clone() if conv.bias is not None else None

        self.lora_a = nn.Parameter(lora_A)
        self.lora_b = nn.Parameter(lora_B)
        self.weight_frozen = nn.Parameter(weight_frozen, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

        self.weight_shape = W.shape
        self.stride = conv.stride
        self.padding = conv.padding
        self.output_padding = conv.output_padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.merge_weights = merge_weights
        self.merged = False
        self.use_spatial = use_spatial

        spatial_dim = kh * kw
        spatial_rank = min(spatial_rank, spatial_dim)
        self.spatial_rank = spatial_rank

        if use_spatial:
            with torch.no_grad():
                s_init = init_spatial_basis_from_svd(W, spatial_rank)
            self.s_basis = nn.Parameter(s_init.contiguous())
            if cin == cout:
                self.u_dw = nn.Parameter(torch.zeros(spatial_rank))
            else:
                self.diag = min(cin, cout)
                self.u_dw = nn.Parameter(torch.zeros(self.diag, spatial_rank))

    def _delta_ch(self):
        if self.r == 0:
            return torch.zeros_like(self.weight_frozen)
        return (self.lora_b @ self.lora_a) * self.scaling

    def _delta_sp(self):
        if not self.use_spatial:
            return 0

        cin, cout, kh, kw = self.weight_shape

        if cin == cout:
            dW = torch.einsum("r, rxy -> xy", self.u_dw, self.s_basis).contiguous()
            dW_sp = torch.zeros(cin, cout, kh, kw, device=dW.device, dtype=dW.dtype)
            idx = torch.arange(cin, device=dW.device)
            dW_sp[idx, idx, :, :] = dW
            return dW_sp
        else:
            diag = self.diag
            dW = torch.einsum("ir, rxy -> ixy", self.u_dw, self.s_basis).contiguous()
            dW_sp = torch.zeros(cin, cout, kh, kw, device=dW.device, dtype=dW.dtype)
            idx = torch.arange(diag, device=dW.device)
            dW_sp[idx, idx, :, :] = dW
            return dW_sp

    def _weight_eff_flat(self):
        W = self.weight_frozen
        if not (self.merge_weights and self.merged):
            W = W + self._delta_ch()
            if self.use_spatial:
                W = W + self._delta_sp().reshape(W.shape[0], -1).contiguous()
        return W

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

    def merge(self):
        if not self.merged:
            with torch.no_grad():
                self.weight_frozen.copy_(self._weight_eff_flat())
            self.merged = True
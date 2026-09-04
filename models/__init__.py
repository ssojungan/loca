import torch
import torch.nn as nn
import logging


from .backbones import *
from .heads import *
from .tuning_modules import set_tuning_config
from .layers.ws_conv import WSConv2d
from .tuning_modules.loca import LoCAModel

__all__ = ['build_model']


def replace_conv2d_with_my_conv2d(net, ws_eps=None):
	if ws_eps is None:
		return

	for m in net.modules():
		to_update_dict = {}
		for name, sub_module in m.named_children():
			if isinstance(sub_module, nn.Conv2d) and sub_module.bias is None:
				# only replace conv2d layers that are followed by normalization layers (i.e., no bias)
				to_update_dict[name] = sub_module
		for name, sub_module in to_update_dict.items():
			m._modules[name] = WSConv2d(
				sub_module.in_channels, sub_module.out_channels, sub_module.kernel_size, sub_module.stride,
				sub_module.padding, sub_module.dilation, sub_module.groups, sub_module.bias is not None,
			)
			# load weight
			m._modules[name].load_state_dict(sub_module.state_dict())
			# load requires_grad
			m._modules[name].weight.requires_grad = sub_module.weight.requires_grad
			if sub_module.bias is not None:
				m._modules[name].bias.requires_grad = sub_module.bias.requires_grad
	# set ws_eps
	for m in net.modules():
		if isinstance(m, WSConv2d):
			m.ws_eps = ws_eps


def build_model(model_name, pretrained=True, num_classes=1000, input_size=224, tuning_method='full',  args=None, **kwargs):
    tuning_config = set_tuning_config(tuning_method, args)

    if model_name.startswith('mambavision'):
        from .backbones import mambavision as mambavision_module
        globals()[model_name] = getattr(mambavision_module, model_name)

    model = eval(model_name)(
        pretrained=pretrained,
        input_resolution=input_size,
        tuning_config=tuning_config,
        **kwargs
    )
    if model_name == 'dinov3_vit_base' or model_name == 'dinov3_convnext_base':
        embed_dim = model.num_features
        model.head = LinearHead(embed_dim, num_classes, 0.2)
    else:
        model.head = LinearHead(model.num_features, num_classes, 0.2)

    if tuning_method == 'loca':
        loca_config = tuning_config['loca_config']
        model = LoCAModel(loca_config, model)

    elif tuning_method == 'conv_adapt':
        for name, param in model.named_parameters():
            if name.startswith('head'):
                continue
            
            if 'tuning_module' in name:
                continue
            
            # add a norm layer before average pooling
            if 'norm' in name:
                continue

            param.requires_grad = False
    elif tuning_method == 'conv_adapt_norm':
        for name, param in model.named_parameters():
            if name.startswith('head'):
                continue
            
            if 'tuning_module' in name:
                continue

            if 'bn' in name:
                continue

            if 'gn' in name:
                continue

            if 'norm' in name:
                continue
            
            # adjust last group norm
            if 'before_head' in name:
                continue

            param.requires_grad = False    
    elif tuning_method == 'conv_adapt_bias':
        for name, param in model.named_parameters():
            if name.startswith('head'):
                continue
            
            if 'tuning_module' in name:
                continue
                
            if 'bias' in name:
                continue
            
            # add a norm layer before average pooling
            if name.startswith('norm'):
                continue

            param.requires_grad = False

    elif tuning_method == 'full':
        for p in model.parameters():
            p.requires_grad = True

    elif tuning_method == 'linear':
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith('head')

    elif tuning_method == 'norm':
        for n, p in model.named_parameters():
            p.requires_grad = ('norm' in n) or n.startswith('head')

    elif tuning_method == 'bias':
        # LayerNorm: train both weight and bias to avoid backward gradient shape error
        # (NativeLayerNormBackward0 index 2 got [0] when only bias was trainable)
        for n, p in model.named_parameters():
            p.requires_grad = ('bias' in n) or n.startswith('head') or ('norm' in n)

    elif tuning_method == 'prompt':
        for n, p in model.named_parameters():
            if n.startswith('head') or 'tuning_module' in n or n.startswith('norm'):
                p.requires_grad = True
            else:
                p.requires_grad = False

    else:
        raise ValueError(f'Unknown tuning_method: {tuning_method}')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.getLogger(__name__).info('[build_model] %s: %d / %d trainable', tuning_method, trainable, total)

    return model

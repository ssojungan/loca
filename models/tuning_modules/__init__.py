from .prompter import PadPrompter
from .conv_adapter import ConvAdapter, LinearAdapter
#from .soma import LoRALinear, LoRAConv2d, LoRAConvTranspose2d
#from .soma_v1 import LoracoConfig
#from .soma_v4 import LoracoConfig
from .loraco import LoracoConfig
from .loraconv import LoraconvConfig
from .loca import LoCAConfig
from .lora import Linear as LoraLinear, Conv2d as LoraConv2d, LoRALayer
from .program_module import ProgramModule


def set_tuning_config(tuning_method, args):

    # conv adapter
    if tuning_method in ['conv_adapt', 'conv_adapt_norm', 'conv_adapt_bias']:
        return {
            'method': tuning_method,
            'kernel_size': args.kernel_size,
            'adapt_size': args.adapt_size,
            'adapt_scale': args.adapt_scale,
        }

    # prompt
    if tuning_method == 'prompt':
        return {
            'method': tuning_method,
            'prompt_size': args.prompt_size,
        }


    if tuning_method == 'loraco':
        import json
        
        # lora_target 파싱
        if isinstance(args.lora_target, list):
            target_modules = args.lora_target
        elif isinstance(args.lora_target, str):
            try:
                parsed = json.loads(args.lora_target)
                target_modules = parsed if isinstance(parsed, list) else args.lora_target.split()
            except (json.JSONDecodeError, ValueError):
                target_modules = args.lora_target.split()
        else:
            target_modules = None
        
        loraco_config = LoracoConfig(
            type=getattr(args, 'type', 'loraco'),
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            merge_weights=False,
            start_lora_idx=0,
        )
        
        return {
            'method': tuning_method,
            'lora_rank': args.lora_rank,
            'lora_alpha': args.lora_alpha,
            'lora_target': args.lora_target,
            'loraco_config': loraco_config,
        }
    elif tuning_method == 'loca':
        import json

        if isinstance(args.lora_target, list):
            target_modules = args.lora_target
        elif isinstance(args.lora_target, str):
            try:
                parsed = json.loads(args.lora_target)
                target_modules = parsed if isinstance(parsed, list) else args.lora_target.split()
            except (json.JSONDecodeError, ValueError):
                target_modules = args.lora_target.split()
        else:
            target_modules = None

        loca_config = LoCAConfig(
            type=getattr(args, 'type', 'loca'),
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            merge_weights=False,
            start_lora_idx=0,
        )

        return {
            'method': tuning_method,
            'lora_rank': args.lora_rank,
            'lora_alpha': args.lora_alpha,
            'lora_target': args.lora_target,
            'loca_config': loca_config,
        }
    elif tuning_method == 'loraconv':
        import json
        # lora_target 파싱
        if isinstance(args.lora_target, list):
            target_modules = args.lora_target
        elif isinstance(args.lora_target, str):
            try:
                parsed = json.loads(args.lora_target)
                target_modules = parsed if isinstance(parsed, list) else args.lora_target.split()
            except (json.JSONDecodeError, ValueError):
                target_modules = args.lora_target.split()
        else:
            target_modules = None
            
        loraconv_config = LoraconvConfig(
            type=getattr(args, 'type', 'loraconv'),
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            merge_weights=False,
            start_lora_idx=0,
        )
        return {
            'method': tuning_method,
            'lora_rank': args.lora_rank,
            'lora_alpha': args.lora_alpha,
            'lora_target': args.lora_target,
            'loraconv_config': loraconv_config,
        }
    elif tuning_method == 'lora':
        import json
        if isinstance(args.lora_target, list):
            target_modules = args.lora_target
        elif isinstance(args.lora_target, str):
            try:
                parsed = json.loads(args.lora_target)
                target_modules = parsed if isinstance(parsed, list) else args.lora_target.split()
            except (json.JSONDecodeError, ValueError):
                target_modules = args.lora_target.split()
        else:
            target_modules = None
        loraconv_config = LoraconvConfig(
            type=getattr(args, 'type', 'lora'),
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            merge_weights=False,
            start_lora_idx=0,
        )
        return {
            'method': tuning_method,
            'lora_rank': args.lora_rank,
            'lora_alpha': args.lora_alpha,
            'lora_target': args.lora_target,
            'loraconv_config': loraconv_config,
        }

    if tuning_method in ['full', 'linear', 'norm', 'repnet', 'repnet_bias', 'bias', 'colora']:
        return { 'method': tuning_method }

    raise NotImplementedError(f"Tuning method '{tuning_method}' not supported.")

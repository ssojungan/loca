import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os
import logging
from tqdm import tqdm
from dataclasses import dataclass

from pathlib import Path

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from timm.optim import create_optimizer_v2, optimizer_kwargs

from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils


def _get_weight_tensor_for_svd(module):
    """Extract weight or LoRA delta for SVD. Returns 2D/4D tensor or (None, False)."""
    cand = None

    # 1) _delta_weight() (loraconv 등)
    if hasattr(module, "_delta_weight"):
        try:
            dW = module._delta_weight().detach().float()
            if dW.dim() in (2, 4):
                cand = dW
            elif dW.dim() == 2 and hasattr(module, "weight_shape"):
                shape = getattr(module, "weight_shape")
                if isinstance(shape, tuple) and len(shape) == 4:
                    try:
                        cand = dW.reshape(shape)
                    except Exception:
                        cand = dW
        except Exception:
            pass

    # 2) lora_A/lora_B (lora.py) 또는 lora_a/lora_b (loraconv): B@A delta 사용
    if cand is None:
        A, B = (module.lora_A, module.lora_B) if hasattr(module, "lora_A") and hasattr(module, "lora_B") else (
            (module.lora_a, module.lora_b) if hasattr(module, "lora_a") and hasattr(module, "lora_b") else (None, None)
        )
        if A is not None and B is not None:
            try:
                delta = (B @ A) * float(getattr(module, "scaling", 1.0))
                conv = getattr(module, "conv", None)
                if conv is not None and getattr(conv, "weight", None) is not None:
                    target_shape = conv.weight.shape
                    try:
                        delta = delta.view(target_shape)
                    except Exception:
                        # lora.py depthwise: B@A shape (out_c//g*kh, in_c*kh) 등 → numel = out_c*(in_c//g)*kh*kw
                        out_c, in_c = conv.out_channels, conv.in_channels
                        g, kh, kw = conv.groups, target_shape[2], target_shape[3]
                        need = out_c * (in_c // max(g, 1)) * kh * kw
                        if delta.dim() == 2 and delta.numel() == need:
                            delta = delta.reshape(out_c, in_c // max(g, 1), kh, kw)
                elif hasattr(module, "weight_shape"):
                    try:
                        delta = delta.view(module.weight_shape)
                    except Exception:
                        pass
                cand = delta.detach().float()
            except Exception:
                pass

    # 3) 일반 weight / conv.weight (LoRA 아닐 때만)
    if cand is None and getattr(module, "weight", None) is not None:
        w = module.weight.detach().float()
        if w.dim() in (2, 4):
            cand = w
    if cand is None and hasattr(module, "conv") and getattr(module.conv, "weight", None) is not None:
        w = module.conv.weight.detach().float()
        if w.dim() in (2, 4):
            cand = w

    return (cand, True) if cand is not None else (None, False)


def get_channel_svd(weight):
    """Channel-wise singular values from 2D/4D weight."""
    if weight.dim() == 4:
        w_flat = weight.reshape(weight.shape[0], -1)
    elif weight.dim() == 2:
        w_flat = weight
    else:
        return None
    _, S, _ = torch.linalg.svd(w_flat, full_matrices=False)
    return S.cpu().numpy()


def get_spatial_svd(weight):
    """Spatial singular values averaged over (out_c, in_c) kernels."""
    if weight.dim() != 4:
        return None
    _, _, kh, kw = weight.shape
    kernels = weight.reshape(-1, kh, kw)
    S = torch.linalg.svdvals(kernels)
    return S.mean(dim=0).cpu().numpy()


def collect_singular_values(model, layer_name):
    """Return channel and spatial singular values for a given layer.
    - channel_sv: (rank,)
    - spatial_sv: (min(kh, kw),) or None for non-conv layers
    """
    m = None
    for name_try in (layer_name, "model." + layer_name):
        try:
            m = model.get_submodule(name_try)
            break
        except AttributeError:
            continue
    if m is None:
        return None
    weight, ok = _get_weight_tensor_for_svd(m)
    if not ok or weight is None:
        return None, None
    channel_sv = get_channel_svd(weight)
    spatial_sv = get_spatial_svd(weight)
    return channel_sv, spatial_sv


def parse_layer_list(layer_arg):
    """Parse either a JSON list string or single layer string to list[str]."""
    if not layer_arg:
        return []
    try:
        parsed = json.loads(layer_arg) if isinstance(layer_arg, str) else layer_arg
        return [parsed] if isinstance(parsed, str) else list(parsed)
    except (json.JSONDecodeError, TypeError):
        return [layer_arg]


@dataclass(frozen=True)
class SingularValueConfig:
    layers: list[str]
    interval: int = 1
    include_spatial: bool = True

    @classmethod
    def from_args(cls, args):
        layer_arg = args.sv_layers if args.sv_layers is not None else args.sv_layer
        return cls(
            layers=parse_layer_list(layer_arg),
            interval=max(1, int(args.sv_interval)),
            include_spatial=bool(args.sv_include_spatial),
        )

    @property
    def enabled(self):
        return bool(self.layers)


def write_singular_value_logs(model, config, output_dir, epoch):
    if not config.enabled or not output_dir or epoch % config.interval != 0:
        return

    model.eval()
    with torch.no_grad():
        for layer_name in config.layers:
            channel_sv, spatial_sv = collect_singular_values(model, layer_name)
            if channel_sv is None:
                raise RuntimeError(
                    f"sv_layers={layer_name!r}: layer not found or has no 2D/4D weight. "
                    "Check the module path, for example model.stages.3.2.dwconv."
                )

            safe_name = layer_name.replace(".", "_")
            mode = "w" if epoch == 0 else "a"
            sv_path = os.path.join(output_dir, f"sv_history_{safe_name}.txt")
            with open(sv_path, mode, encoding="utf-8") as f:
                f.write(f"epoch_{epoch}\t" + "\t".join(f"{x:.6e}" for x in channel_sv) + "\n")

            if config.include_spatial and spatial_sv is not None:
                spatial_path = os.path.join(output_dir, f"sv_history_spatial_{safe_name}.txt")
                with open(spatial_path, mode, encoding="utf-8") as f:
                    f.write(f"epoch_{epoch}\t" + "\t".join(f"{x:.6e}" for x in spatial_sv) + "\n")
    model.train()


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('Parameter Efficient Tuning', add_help=False)
    
    parser.add_argument('--tuning_method', type=str, default='prompt')

    parser.add_argument('--prompt_size', default=10, type=int, help='prompt size')
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--adapt_size', default=8, type=float)
    parser.add_argument('--adapt_scale', default=1.0, type=float)

    parser.add_argument('--lora_rank', default=16, type=int, help='LoRA rank')
    parser.add_argument('--lora_alpha', default=16.0, type=float, help='LoRA alpha')
    parser.add_argument('--lora_target', default='mlp', type=str, help='LoRA target layers')
    parser.add_argument('--type', default='loraco', type=str, help='LoRA type (loraco, sora, etc.)')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    parser.add_argument('--fs_shot', default=16, type=int)

    # Model parameters
    parser.add_argument('--model', default='resnet50_clip', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--crop_ratio', default=0.875, type=float,
                        help='image input size')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--sv_layers', default=None, type=str,
                        help='JSON list or single module path for singular-value logging.')
    parser.add_argument('--sv_layer', default=None, type=str,
                        help='Deprecated alias for --sv_layers.')
    parser.add_argument('--sv_interval', default=1, type=int,
                        help='Log singular values every N epochs.')
    parser.add_argument('--sv_include_spatial', default=True, type=str2bool,
                        help='Also log spatial singular values for convolutional kernels.')

    # Dataset parameters
    parser.add_argument('--is_tuning', default=False, type=str2bool, help='Hyperparameter Tuning Mode')
    parser.add_argument('--data', default='vtab-1k', type=str, help='dataset name')
    parser.add_argument('--dataset', default='stanford_cars', type=str, help='dataset name')
    parser.add_argument('--data_path', default='/media/Auriga/Parameter_Efficient/data/few-shot/stanford_cars', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--output_dir', default='./experiments/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=False)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=2, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', '--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=False, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--wandb', type=str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='parameter_efficient_tuning_cv', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_name', default='convnext_adapter', type=str,
                        help="The name of the W&B run.")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")
    parser.add_argument('--print_params_only', type=str2bool, default=False,
                        help="Only print model param counts and exit (no training).")
    parser.add_argument('--profile_memory', type=str2bool, default=False,
                        help="Run a dummy forward pass to estimate model memory.")

    return parser

def main(args):
    utils.init_distributed_mode(args)
    
    handlers = []
    if args.output_dir and utils.is_main_process():
        handlers.append(logging.FileHandler(os.path.join(args.output_dir, 'training_log.txt')))
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
    logger = logging.getLogger(__name__)
    
    if utils.is_main_process():
        logger.info(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(args=args, is_train=True)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(args=args, is_train=False)
    if utils.is_main_process() and dataset_val is not None:
        logger.info("Validation examples: %d", len(dataset_val))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    if utils.is_main_process():
        logger.info("Train sampler: %s", sampler_train.__class__.__name__)
    sampler_val = None
    if dataset_val is not None and args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            logger.warning(
                "Distributed evaluation dataset is not divisible by the process count; "
                "extra duplicate samples may slightly alter validation results."
            )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    elif dataset_val is not None:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        logger.info("Mixup is activated.")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = build_model(
        args.model, 
        # set to True for fine-tuning
        pretrained=True, 
        num_classes=args.nb_classes, 
        tuning_method=args.tuning_method,
        args=args,
        )
    # Calculate model memory usage
    def calculate_model_memory(model, input_shape, device='cuda'):
        """Calculate model memory usage including parameters and activations."""
        model.eval()
        
        # Calculate parameter size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # Parameter memory (assuming 32-bit for trainable, 8-bit for frozen)
        param_size_trainable = trainable_params * 4  # 32 bits = 4 bytes
        param_size_frozen = frozen_params * 1  # 8 bits = 1 byte
        param_size = param_size_trainable + param_size_frozen
        
        # Calculate activation memory by forward pass
        model = model.to(device)
        dummy_input = torch.randn(input_shape).to(device)
        
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            mem_after = torch.cuda.memory_allocated()
            # Activation memory = total memory after forward - param memory
            # Note: This includes gradients buffer, so it's an approximation
            act_size = max(0, mem_after - mem_before - param_size)
            total_memory = mem_after
        else:
            # CPU fallback: estimate activation size
            # Rough estimation based on input size and model depth
            with torch.no_grad():
                _ = model(dummy_input)
            # Estimate: input_size * 4 bytes * approximate depth factor
            input_size_bytes = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * 4
            act_size = input_size_bytes * 5  # rough estimate for intermediate activations
            total_memory = param_size + act_size
        
        return {
            'memory_cost': total_memory,
            'param_size': param_size,
            'act_size': act_size,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
        }
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    memory_info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }
    if args.profile_memory:
        memory_info.update(
            calculate_model_memory(
                model,
                (8, 3, args.input_size, args.input_size),
                device=args.device,
            )
        )

    if utils.is_main_process():
        logger.info(
            "Parameters: total=%d trainable=%d frozen=%d",
            memory_info["total_params"],
            memory_info["trainable_params"],
            memory_info["frozen_params"],
        )
        if args.profile_memory:
            logger.info(
                "Model memory: total=%.2fMB params=%.2fMB activations=%.2fMB",
                memory_info["memory_cost"] / 1e6,
                memory_info["param_size"] / 1e6,
                memory_info["act_size"] / 1e6,
            )

    if args.print_params_only:
        if args.output_dir and utils.is_main_process():
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(args.output_dir, "log.txt"), "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "total_params": memory_info["total_params"],
                    "n_parameters": memory_info["trainable_params"],
                    "frozen_params": memory_info["frozen_params"],
                }) + "\n")
        if args.distributed:
            torch.distributed.destroy_process_group()
        return

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        logger.info("Load ckpt from %s", args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                logger.info("Load state_dict by model_key=%s", model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.info("Removing key %s from pretrained checkpoint", k)
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        logger.info("Using EMA with decay %.8f", args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Number of trainable params: %d", n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    logger.info("LR: %.8f", args.lr)
    logger.info("Total batch size: %d", total_batch_size)
    logger.info("Update frequency: %d", args.update_freq)
    logger.info("Training examples: %d", len(dataset_train))
    logger.info("Training steps per epoch: %d", num_training_steps_per_epoch)

    if args.distributed:
        # Set find_unused_parameters=True to handle parameters that may not be used in forward pass
        # This is needed when using LoRA or gradient checkpointing where some parameters may be conditionally used
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer_v2(model_without_ddp, **optimizer_kwargs(cfg=args))

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

    logger.info("Use cosine LR scheduler.")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    logger.info("Weight decay schedule: max=%.7f min=%.7f", max(wd_schedule_values), min(wd_schedule_values))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    logger.info("Criterion: %s", criterion)

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, data=args.data, use_amp=args.use_amp, args=args)
        if args.distributed:
            torch.distributed.destroy_process_group()
        return

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    sv_config = SingularValueConfig.from_args(args)
    if sv_config.enabled and utils.is_main_process():
        logger.info("SV logging: layers=%s interval=%d include_spatial=%s", sv_config.layers, sv_config.interval, sv_config.include_spatial)

    logger.info("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Training", disable=not utils.is_main_process()):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        log_writer and log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        wandb_logger and wandb_logger.set_steps()
        
        test_stats = None
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, data=args.data, use_amp=args.use_amp)
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                args.output_dir and args.save_ckpt and utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            log_writer and log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
            log_writer and log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
            log_writer and log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)
            wandb_logger and wandb_logger.log_epoch_metrics({'epoch': epoch, **{f'test_{k}': v for k, v in test_stats.items()}}, commit=True)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp
        )
        
        if args.output_dir and args.save_ckpt and ((epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs):
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)

        if utils.is_main_process():
            write_singular_value_logs(model_without_ddp, sv_config, args.output_dir, epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **({f'test_{k}': v for k, v in test_stats.items()} if test_stats else {}),
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'total_params': memory_info['total_params']}
        
        if data_loader_val is not None and args.model_ema and args.model_ema_eval:
            test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, data=args.data, use_amp=args.use_amp)
            if max_accuracy_ema < test_stats_ema["acc1"]:
                max_accuracy_ema = test_stats_ema["acc1"]
                args.output_dir and args.save_ckpt and utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
            log_writer and log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
            log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})

        if args.output_dir and utils.is_main_process():
            log_writer and log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        wandb_logger and wandb_logger.log_epoch_metrics(log_stats)

    wandb_logger and args.wandb_ckpt and args.save_ckpt and args.output_dir and wandb_logger.log_checkpoints()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    
    if args.output_dir and utils.is_main_process():
        final_acc = max_accuracy
        if args.model_ema and args.model_ema_eval:
            final_acc = max(max_accuracy, max_accuracy_ema)
        # acc1 may be ratio (0-1) or percentage (0-100); normalize for display
        to_pct = lambda x: x * 100 if x <= 1.0 else x
        result_file = os.path.join(args.output_dir, "final_result.txt")
        with open(result_file, "w") as f:
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Final Accuracy: {to_pct(final_acc):.2f}%\n")
            f.write(f"Max Accuracy: {to_pct(max_accuracy):.2f}%\n")
            if args.model_ema and args.model_ema_eval:
                f.write(f"Max Accuracy (EMA): {to_pct(max_accuracy_ema):.2f}%\n")
            f.write(f"Training Time: {total_time_str}\n")
    
    if args.distributed:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if not args.is_tuning:
        args = utils.auto_load_optim_param(args, args.model, args.tuning_method, args.dataset)
    else:
        args.save_ckpt = False
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    #args.data = args.data_path.split('/')[-2]
    main(args)

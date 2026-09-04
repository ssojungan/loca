import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.utils import get_state_dict

from .convnext import ConvNeXt

__all__ = [
    'convnext_tiny',
    'convnext_tiny_22k',
    'convnext_small',
    'convnext_small_22k',
    'convnext_base',
    'convnext_base_22k',
    'convnext_large',
    'convnext_large_22k',
    'convnext_xlarge_22k'
]

def convnext_tiny(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        timm_model = timm.create_model('convnext_tiny', pretrained=True)
        state_dict = get_state_dict(timm_model)
        state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model

def convnext_tiny_22k(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        timm_model = timm.create_model('convnext_tiny.fb_in22k', pretrained=True)
        state_dict = get_state_dict(timm_model)
        state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model

def convnext_small(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        timm_model = timm.create_model('convnext_small', pretrained=True)
        state_dict = get_state_dict(timm_model)
        state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model

def convnext_small_22k(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        timm_model = timm.create_model('convnext_small.fb_in22k', pretrained=True)
        state_dict = get_state_dict(timm_model)
        state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model


def convnext_base(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        timm_model = timm.create_model('convnext_base', pretrained=True)
        state_dict = get_state_dict(timm_model)
        state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model

def convnext_base_22k(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        timm_model = timm.create_model('convnext_base.fb_in22k', pretrained=True)
        state_dict = get_state_dict(timm_model)
        state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model


def convnext_large(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        timm_model = timm.create_model('convnext_large', pretrained=True)
        state_dict = get_state_dict(timm_model)
        state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model


def convnext_large_22k(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        timm_model = timm.create_model('convnext_large.fb_in22k', pretrained=True)
        state_dict = get_state_dict(timm_model)
        state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model


def convnext_xlarge_22k(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        timm_model = timm.create_model('convnext_xlarge.fb_in22k', pretrained=True)
        state_dict = get_state_dict(timm_model)
        state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model




import torch.nn as nn
import timm

__all__ = [
    'MambaOut',
    'mambaout_base',
    'mambaout_small',
    'mambaout_tiny',
]

class MambaOut(nn.Module):
    def __init__(
        self,
        model_name='mambaout_base.in1k',
        pretrained=True,
        **kwargs,
    ):
        super().__init__()
 
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.num_features = self.model.num_features

    def forward(self, x):
        if hasattr(self.model, "forward_features"):
            x = self.model.forward_features(x)
        else:
            x = self.model(x)
        if x.ndim == 4:
            x = x.mean(dim=(-2, -1))
        elif x.ndim == 3:
            x = x.mean(dim=1)
        return self.head(x) if hasattr(self, "head") else x


def mambaout_tiny(pretrained=False, **kwargs):
    return MambaOut(model_name='mambaout_tiny.in1k', pretrained=pretrained, **kwargs)


def mambaout_small(pretrained=False, **kwargs):
    return MambaOut(model_name='mambaout_small.in1k', pretrained=pretrained, **kwargs)


def mambaout_base(pretrained=False, **kwargs):
    return MambaOut(model_name='mambaout_base.in1k', pretrained=pretrained, **kwargs)

import torch.nn as nn
import timm


class DINOv3ConvNeXtLarge(nn.Module):
    def __init__(
        self,
        model_name="convnext_large.dinov3_lvd1689m", # convnext_large.fb_in1k
        out_indices=(0, 1, 2, 3),
        in_chans=3,
        pretrained=True,
        lora_cfg=None,
        **kwargs,
    ):
        super().__init__()
        kwargs.pop('weights', None)
        kwargs.pop('arch', None)
        
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=tuple(out_indices),
            in_chans=in_chans,
            **kwargs,
        )

        self.lora_cfg = lora_cfg if lora_cfg is not None else {}
        self.out_indices = tuple(out_indices)

        finfo = getattr(self.model, "feature_info", None)
        if finfo is not None and len(finfo) > 0:
            self.embed_dim = finfo.channels()[-1]
            self.strides = list(finfo.reduction())
        else:
            self.embed_dim = 1536
            self.strides = [4, 8, 16, 32]

        self.patch_size = 4

    def forward(self, x, masks=None):
        if isinstance(x, list):
            return [self.forward(x_i, None) for x_i in x]
        return self.model(x)


class DINOv3ViTL16Large(nn.Module):
    def __init__(
        self,
        model_name="vit_large_patch16_dinov3.lvd1689m",
        out_indices=(7, 11, 15, 23),
        pretrained=True,
        lora_cfg=None,
        **kwargs,
    ):
        super().__init__()
        kwargs.pop('weights', None)
        kwargs.pop('arch', None)

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,       
            out_indices=out_indices,
            num_classes=0,
            global_pool="",
            dynamic_img_size=True,    
            **kwargs
        )

        if hasattr(self.model, 'feature_info'):
            self.embed_dim = self.model.feature_info.channels()[-1]
        else:
            self.embed_dim = 1024

        if hasattr(self.model, 'model'):
            raw_model = self.model.model 
        else:
            raw_model = self.model
        self.num_prefix_tokens = getattr(raw_model, 'num_prefix_tokens', 1)
        
        patch_embed = getattr(raw_model, 'patch_embed', None)
        if patch_embed:
            ps = patch_embed.patch_size
            self.patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
        else:
            self.patch_size = 16

        self.lora_cfg = lora_cfg if lora_cfg is not None else {}

    def forward(self, x, masks=None):
        features = self.model(x)
        processed_feats = []

        for feat in features:
            if feat.shape[-1] == self.embed_dim:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            processed_feats.append(feat)

        return processed_feats
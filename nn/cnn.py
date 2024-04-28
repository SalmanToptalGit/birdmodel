
import torch.nn as nn
import timm
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
import torch
import numpy as np
from utils.torch_augs import *
import warnings
warnings.filterwarnings("ignore")




class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_min, norm_max = \
            Xstd.min(-1)[0].min(-1)[0], Xstd.max(-1)[0].max(-1)[0]
        fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
            (norm_max - norm_min)
        )
        V = torch.zeros_like(Xstd)
        if fix_ind.sum():
            V_fix = Xstd[fix_ind]
            norm_max_fix = norm_max[fix_ind, None, None]
            norm_min_fix = norm_min[fix_ind, None, None]
            V_fix = torch.max(
                torch.min(V_fix, norm_max_fix),
                norm_min_fix,
            )
            V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
            V[fix_ind] = V_fix
        return V


class PoolingLayer(nn.Module):
    def __init__(self, pool_type: str, p=3, eps=1e-6):
        super().__init__()

        self.pool_type = pool_type

        if self.pool_type == "AdaptiveAvgPool2d":
            self.pool_layer = nn.AdaptiveAvgPool2d((1, 1))
        elif self.pool_type == "GeM":
            self.pool_layer = nn.AdaptiveAvgPool2d((1, 1))
            self.p = torch.nn.Parameter(torch.ones(1) * p)
            self.eps = eps
        else:
            raise RuntimeError(f"{self.pool_type} is invalid pool_type")

    def forward(self, x):
        bs, ch, h, w = x.shape
        if self.pool_type == "AdaptiveAvgPool2d":
            x = self.pool_layer(x)
            x = x.view(bs, ch)
        elif self.pool_type == "GeM":
            x = self.pool_layer(x.clamp(min=self.eps).pow(self.p)).pow(
                1.0 / self.p
            )
            x = x.view(bs, ch)
        return x


class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        mel_spec_params = config["mel_spec_params"]
        self.logmelspec_extractor = nn.Sequential(
            MelSpectrogram(
                sample_rate=mel_spec_params["sample_rate"],
                n_mels=mel_spec_params["n_mels"],
                f_min=mel_spec_params["f_min"],
                f_max=mel_spec_params["f_max"],
                n_fft=mel_spec_params["n_fft"],
                hop_length=mel_spec_params["hop_length"],
                normalized=True,
            ),
            AmplitudeToDB(top_db=80.0),
            NormalizeMelSpec(),
        )

        out_indices = (3, 4)
        self.backbone = timm.create_model(
            config["backbone"],
            features_only=True,
            pretrained=config["pretrained"],
            in_chans=config["in_chans"],
            num_classes=0,
            out_indices=out_indices,
        )
        feature_dims = self.backbone.feature_info.channels()
        print(f"feature dims: {feature_dims}")
        self.global_pools = torch.nn.ModuleList([PoolingLayer("GeM") for _ in out_indices])
        self.mid_features = np.sum(feature_dims)
        self.neck = torch.nn.BatchNorm1d(self.mid_features)
        self.head = nn.Linear(self.mid_features, len(config["target_columns"]))

        self.mixup = Mixup(mix_beta=1.0)
        self.cutmix = Cutmix(mix_beta=1.0)

    def apply_mixup_cutmix(self, x, y, weight):

        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, t, c, f)
        if np.random.random() <= 1.0:
            x, y, weight = self.mixup(x, y, weight)
        if np.random.random() <= 0.5:
            x, y, weight = self.cutmix(x, y, weight)
        x = x.reshape(b, t, c, f)
        x = x.permute(0, 2, 3, 1)
        return x, y, weight

    def backbone_pass(self, x):
        spec = self.logmelspec_extractor(x["wave"]).unsqueeze(1)

        if self.training:
            spec, x["smooth_targets"], x["rating"] = self.apply_mixup_cutmix(x=spec, y=x["smooth_targets"], weight=x["rating"])

        ms = self.backbone(spec)
        h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
        features = self.neck(h)
        features = self.head(features)
        x["logit"] = features
        return x

    def forward(self, x):
        return self.backbone_pass(x)

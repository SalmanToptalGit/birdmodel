
import torch.nn as nn
import timm
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from utils.spec_utils import SpecAugment
import torch
import numpy as np
from utils.torch_augs import Mixup
import warnings
import torch.nn.functional as F
from utils.spec_utils import TraceableMelspec, NormalizeMelSpec
from torchvision.transforms import Normalize
warnings.filterwarnings("ignore")


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

def gem(x, p=3, eps=1e-6):
    return


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        bs, ch, h, w = x.shape
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)
        x = x.view(bs, ch)
        return x



class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        mel_spec_params = config["mel_spec_params"]
        top_db = mel_spec_params.pop("top_db")
        self.logmelspec_extractor = nn.Sequential(
            TraceableMelspec(**mel_spec_params),
            AmplitudeToDB(top_db=top_db),
            NormalizeMelSpec(exportable=True),
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
        self.global_pools = torch.nn.ModuleList([PoolingLayer('GeM') for _ in out_indices])
        self.mid_features = np.sum(feature_dims)
        self.neck = torch.nn.BatchNorm1d(self.mid_features)
        self.head = nn.Linear(self.mid_features, len(config["target_columns"]))

        self.spec_aug = SpecAugment(freq_mask_config=config["spec_augment_config"]["freq_mask"],
                                    time_mask_config=config["spec_augment_config"]["time_mask"])
        self.mixup = Mixup(mix_beta=1.0)
        self.KD = config["KD"]
        self.in_chans = config["in_chans"]
        self.device = config["device"]

    def imagenet_norm(self, img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0):
        mean = torch.tensor(mean).unsqueeze(0).unsqueeze(1).unsqueeze(2).to(self.device)
        std = torch.tensor(std).unsqueeze(0).unsqueeze(1).unsqueeze(2).to(self.device)


        mean *= max_pixel_value
        std *= max_pixel_value

        denominator = 1.0 / std

        img -= mean
        img *= denominator
        return img

    def normalize_images(self, images):
        # Assuming images is a PyTorch tensor of size [BS x C x W x H]
        bs, c, w, h = images.size()

        # Reshape images to [BS x (C*W*H)] to calculate min and max across all channels and pixels
        reshaped_images = images.contiguous().view(bs, c, -1)

        # Calculate min and max values across all images, channels, and pixels
        min_vals, _ = reshaped_images.min(dim=2, keepdim=True)
        max_vals, _ = reshaped_images.max(dim=2, keepdim=True)

        # Normalize images to the range [0, 1]
        normalized_images = (images - min_vals.unsqueeze(3)) / (max_vals.unsqueeze(3) - min_vals.unsqueeze(3) + 1e-6)

        # Scale the normalized images to the range [0, 255]
        normalized_images *= 255.0

        # # Convert the tensor to uint8 data type
        # normalized_images = normalized_images.to(torch.uint8)

        return normalized_images

    def norm_to_rgb(self, A):
        A = self.normalize_images(A)
        A = self.imagenet_norm(A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return A


    def apply_mixup(self, x, y, weight, teacher_preds=None):

        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, t, c, f)
        if np.random.random() <= 1.0:
            if teacher_preds is not None:
                x, y, weight, teacher_preds = self.mixup(x, y, weight, teacher_preds)
            else:
                x, y, weight = self.mixup(x, y, weight)
        x = x.reshape(b, t, c, f)
        x = x.permute(0, 2, 3, 1)
        return x, y, weight, teacher_preds

    def backbone_pass(self, x):
        spec = self.logmelspec_extractor(x["wave"]).unsqueeze(1)

        if self.training:
            spec = self.spec_aug(spec)
            if self.KD:
                teacher_preds = None
            else:
                teacher_preds = x["teacher_preds"]

            spec, x["smooth_targets"], x["rating"], x["teacher_preds"] = self.apply_mixup(x=spec, y=x["smooth_targets"], weight=x["rating"], teacher_preds=teacher_preds)

        spec = spec.expand(-1, 3, -1, -1)
        ms = self.backbone(spec)
        h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
        features = self.neck(h)
        features = self.head(features)
        x["logit"] = features
        return x

    def forward(self, x):
        return self.backbone_pass(x)

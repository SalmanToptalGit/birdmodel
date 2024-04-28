import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from utils.spec_utils import SpecAugment
from torch.distributions import Beta

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


class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None, teacher_preds=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            teacher_preds = coeffs.view(-1, 1) * teacher_preds + (1 - coeffs.view(-1, 1)) * teacher_preds[perm]
            return X, Y, weight, teacher_preds


class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        self.head = nn.Linear(self.mid_features, config["data_config"]["num_classes"])
        self.mixup_p = config["mixup_p"]
        self.mixup = Mixup(mix_beta=1)
        self.spec_augment = SpecAugment(config["freq_mask"], config["time_mask"])


    def get_mixup_spec(self, input):
        x = input['spec']
        y = input["loss_target"]
        weight = input["rating"]
        teacher_preds = input["teacher_preds"]
        if self.training:
            x = self.spec_augment(x)
            if np.random.random() <= 0.5:
                y2 = torch.repeat_interleave(y, 1, dim=0)
                weight2 = torch.repeat_interleave(weight, 1, dim=0)
                teacher_preds2 = torch.repeat_interleave(
                    teacher_preds, 1, dim=0
                )

                for i in range(0, x.shape[0], 1):
                    x[i: i + 1], _, _, _ = self.mixup(
                        x[i: i + 1],
                        y2[i: i + 1],
                        weight2[i: i + 1],
                        teacher_preds2[i: i + 1],
                    )

            b, c, f, t = x.shape
            x = x.permute(0, 3, 1, 2)
            x = x.reshape(b, t, c, f)

            if np.random.random() <= self.mixup_p:
                x, y, weight, teacher_preds = self.mixup(x, y, weight, teacher_preds)

            x = x.reshape(b, t, c, f)
            x = x.permute(0, 2, 3, 1)
        return {
            "spec": x,
            "loss_target" : y,
            "rating" : weight,
            "teacher_preds" : teacher_preds,
        }



    def get_features(self, input):
        if self.training:
            input = self.get_mixup_spec(input)
            x = input['spec']
            y = input["loss_target"]
            weight = input["rating"]
            teacher_preds = input["teacher_preds"]

            ms = self.backbone(x)
            h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
            x = self.neck(h)
            return {
                "spec": x,
                "loss_target": y,
                "rating": weight,
                "teacher_preds": teacher_preds,
            }
        else:
            ms = self.backbone(input)
            h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
            input = self.neck(h)
            return input



    def forward(self, x):

        input = self.get_features(x)
        if self.training:
            return {
                "logit": self.head(input['spec']),
                "loss_target": input["loss_target"],
                "rating": input["rating"],
                "teacher_preds": input["teacher_preds"],
            }
        else:
            return self.head(input)



class BCEKDLoss(nn.Module):
    def __init__(self, weights=[0.1, 0.9], class_weights=None, num_classes=182):
        super().__init__()

        self.weights = weights
        self.num_classes = num_classes
        self.T = 20

    def forward(self, output):
        input_ = output["logit"]
        target = output["loss_target"].float()
        rating = output["rating"]
        teacher_preds = output["teacher_preds"]

        rating = rating.unsqueeze(1).repeat(1, self.num_classes)
        loss = nn.BCEWithLogitsLoss(
            weight=rating,
            reduction='mean',
        )(input_, target)

        KD_loss = nn.KLDivLoss()(
            F.log_softmax(input_ / self.T, dim=1),
            F.softmax(teacher_preds / self.T, dim=1)
            ) * (self.weights[1] * self.T * self.T)

        return self.weights[0] * loss + KD_loss
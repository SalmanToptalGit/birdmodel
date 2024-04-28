import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F



class BCEKDLoss(torch.nn.Module):
    def __init__(self, weights=[0.1, 0.9]):
        super().__init__()

        self.weights = weights
        self.T = 20

    def forward(self, x, num_classes=182):
        input_ = x["logit"]
        target = x["smooth_targets"]
        rating = x["rating"]
        teacher_preds = x["teacher_preds"]

        target = target.float()

        rating = rating.unsqueeze(1).repeat(1, num_classes)
        loss = nn.BCEWithLogitsLoss(
            weight=rating,
            reduction='mean',
        )(input_, target)

        KD_loss = nn.KLDivLoss()(
            F.log_softmax(input_ / self.T, dim=1),
            F.softmax(teacher_preds / self.T, dim=1)
            ) * (self.weights[1] * self.T * self.T)

        return self.weights[0] * loss + KD_loss

class BCELoss(torch.nn.Module):
    def __init__(self, scale_down = 0.3):
        super().__init__()
        self.scale_down = scale_down

    def forward(self, x, num_classes=182):
        inputs = x["logit"]
        targets = x["smooth_targets"]
        rating = x["rating"]

        rating = rating.unsqueeze(1).repeat(1, num_classes)
        loss = nn.BCEWithLogitsLoss(
            weight=rating,
            reduction='mean',
        )(inputs, targets)
        return loss * self.scale_down

class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, x):
        inputs = x["logit"]
        targets = x["smooth_targets"]
        return torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
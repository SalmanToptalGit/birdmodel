import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
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
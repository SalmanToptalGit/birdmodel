import torch
from torch import dtype
import numpy as np
class Mixup(torch.nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = torch.distributions.Beta(mix_beta, mix_beta)

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
            if teacher_preds is None:
                return X, Y, weight
            teacher_preds = coeffs.view(-1, 1) * teacher_preds + (1 - coeffs.view(-1, 1)) * teacher_preds[perm]
            return X, Y, weight, teacher_preds

class ConcatMix(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    @torch.no_grad()
    def forward(self, audios, labels, weights=None):
        '''
            input:
                :param: audios, - torch.Tensor of shape (batch_size, length)
                :param: labels - torch.Tensor of shape (batch_size, num_classes)
                :param: weights - torch.Tensor of shape (batch_size, )
            output:
                :param: cat_audios - torch.Tensor of shape (batch_size, 2 * length)
                :param: new_labels - torch.Tensor of shape (batch_size, num_classes)
                :param: new_weights - torch.Tensor of shape (batch_size, )

            This method performs concatenation of random pairs of audios from batch
            and also changes labels of given audio to the union of labels of pair.

            NB: works for n >= 2
        '''
        bs = audios.shape[0]

        if bs >= 3:
            shifts = torch.randint(1, bs - 1, size=(1,)).item()
        else:
            shifts = 1
        perm = torch.roll(torch.arange(0, bs), shifts=shifts).long()
        # print(f"permutation : {perm}")
        shuffled_audios = audios[perm].to(self.device)
        shuffled_labels = labels[perm].to(self.device)

        if weights is not None:
            shuffled_weights = weights[perm].to(self.device)

        cat_audios = torch.cat([audios, shuffled_audios], dim=1)
        new_labels = torch.clip(shuffled_labels + labels, min=0, max=1)

        if weights is not None:
            new_weights = (weights + shuffled_weights) / 2
            return cat_audios, new_labels, new_weights
        else:
            return cat_audios, new_labels


import torch
from torch.distributions import Beta
import numpy as np

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class Cutmix(torch.nn.Module):
    def __init__(self, mix_beta):

        super(Cutmix, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample((1, )).to(X.device)

        assert n_dims == 4

        bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), coeffs[0].item())

        X[:, :, bbx1:bbx2, bby1:bby2] = X[perm, :, bbx1:bbx2, bby1:bby2]

        coeffs = torch.Tensor([1 - ((bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2]))]).to(X.device)

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight





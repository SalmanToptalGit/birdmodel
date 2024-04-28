import torch


def sumup(waves: torch.Tensor, labels: torch.Tensor):
    batch_size = len(labels)
    perm = torch.randperm(batch_size)

    waves = waves + waves[perm]

    return {
        "waves": waves,
        "labels": torch.clip(labels + labels[perm], min=0, max=1)
    }

def sumix(waves: torch.Tensor, labels: torch.Tensor, max_percent: float = 1.0, min_percent: float = 0.3):
    batch_size = len(labels)
    perm = torch.randperm(batch_size)
    coeffs_1 = torch.rand(batch_size, device=waves.device).view(-1, 1) * (
        max_percent  - min_percent
    ) + min_percent
    coeffs_2 = torch.rand(batch_size, device=waves.device).view(-1, 1) * (
        max_percent  - min_percent
    ) + min_percent
    label_coeffs_1 = torch.where(coeffs_1 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_1))
    label_coeffs_2 = torch.where(coeffs_2 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_2))
    labels = label_coeffs_1 * labels + label_coeffs_2 * labels[perm]

    waves = coeffs_1 * waves + coeffs_2 * waves[perm]
    return {
        "waves": waves,
        "labels": torch.clip(labels, 0, 1)
    }

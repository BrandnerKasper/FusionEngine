import torch
import torch.nn as nn


class WeightedL1Loss(nn.Module):
    def __init__(self, class_weights: list[float], normalize:bool =True):
        super().__init__()
        self.register_buffer("wcls", torch.tensor(class_weights, dtype=torch.float32))
        self.normalize = normalize

    def forward(self, pred: torch.Tensor, target: torch.Tensor, one_hot: torch.Tensor):
        w = (one_hot * self.wcls.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)

        if self.normalize:
            w = w / (w.mean() + 1e-8)

        return (w* (pred - target).abs()).mean()
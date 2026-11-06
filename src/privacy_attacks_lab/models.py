"""Small target and shadow models.

LiRA needs many shadow models, so every architecture here is sized to train in
seconds to about a minute on an Apple M1 (MPS). All of them avoid BatchNorm: batch
statistics mix examples, which breaks per-sample gradients in DP-SGD, so the CNN
uses GroupNorm and the MLPs use none. The same architecture serves as target,
shadow and reference model, as the attacks assume.
"""

from __future__ import annotations

import itertools
from typing import Literal

import torch
from torch import nn

Arch = Literal["cnn", "cnn-small", "mlp", "mlp-small"]


class SmallCNN(nn.Module):
    """Three conv blocks (conv, GroupNorm, ReLU, max-pool) and a linear head.

    ``width`` scales the channel counts (``width``, ``2 width``, ``4 width``); width 32
    gives 0.12M parameters.
    """

    def __init__(
        self, num_classes: int = 10, width: int = 32, in_channels: int = 3, image_size: int = 32
    ) -> None:
        super().__init__()
        chans = [in_channels, width, 2 * width, 4 * width]
        layers: list[nn.Module] = []
        for c_in, c_out in itertools.pairwise(chans):
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, c_out), c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*layers)
        side = image_size // 8
        if side < 1:
            raise ValueError("images must be at least 8 pixels wide")
        self.head = nn.Linear(chans[-1] * side * side, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.features(x), 1))


class MLP(nn.Module):
    """Fully connected network with Tanh activations (as in Nasr et al. 2018 for Purchase100)."""

    def __init__(self, in_features: int, num_classes: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        dims = [in_features, *hidden]
        layers: list[nn.Module] = []
        for d_in, d_out in itertools.pairwise(dims):
            layers += [nn.Linear(d_in, d_out), nn.Tanh()]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


def build(arch: Arch, in_shape: tuple[int, ...], num_classes: int) -> nn.Module:
    """Construct a model for inputs of shape ``in_shape`` (without the batch axis)."""
    if arch == "cnn":
        return SmallCNN(num_classes, 32, in_shape[0], in_shape[-1])
    if arch == "cnn-small":
        return SmallCNN(num_classes, 16, in_shape[0], in_shape[-1])
    features = 1
    for d in in_shape:
        features *= d
    if arch == "mlp":
        return MLP(features, num_classes, (512, 256, 128))
    if arch == "mlp-small":
        return MLP(features, num_classes, (128,))
    raise ValueError(f"unknown architecture {arch!r}")


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

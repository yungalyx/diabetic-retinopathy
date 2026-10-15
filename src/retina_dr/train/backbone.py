"""Adapter for the timm model zoo. All timm calls go through this module."""

from typing import cast

import timm
from torch import nn


def create_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return a feature extractor (pooled features, no classifier) and its feature width."""
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    return model, int(cast(int, model.num_features))

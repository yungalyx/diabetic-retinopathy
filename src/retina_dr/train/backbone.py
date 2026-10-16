"""Adapter for the timm model zoo. All timm calls go through this module."""

from typing import cast

import timm
from torch import nn

Triple = tuple[float, float, float]
Normalization = tuple[Triple, Triple]
"""Per-channel mean and std that the pretrained weights expect."""

_IMAGENET: Normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def create_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return a feature extractor (pooled features, no classifier) and its feature width."""
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    return model, int(cast(int, model.num_features))


def _triple(value: object, default: Triple) -> Triple:
    if isinstance(value, tuple | list) and len(value) == 3:
        a, b, c = (float(cast(float, v)) for v in value)
        return (a, b, c)
    return default


def normalization_of(backbone: nn.Module) -> Normalization:
    """Mean and std from the backbone's pretrained config. ImageNet values when absent (D-023)."""
    config = getattr(backbone, "pretrained_cfg", None)
    if not isinstance(config, dict):
        return _IMAGENET
    return (_triple(config.get("mean"), _IMAGENET[0]), _triple(config.get("std"), _IMAGENET[1]))


def licence_of(backbone: nn.Module) -> str:
    """Licence of the pretrained weights from the timm config, or "not recorded"."""
    config = getattr(backbone, "pretrained_cfg", None)
    if isinstance(config, dict) and isinstance(config.get("license"), str):
        return str(config["license"])
    return "not recorded"

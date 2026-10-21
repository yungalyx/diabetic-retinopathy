"""Adapter for the timm model zoo. All timm calls go through this module."""

import os
from collections.abc import Mapping
from typing import Final, cast

import timm
from torch import Tensor, nn

Triple = tuple[float, float, float]
Normalization = tuple[Triple, Triple]
"""Per-channel mean and std that the pretrained weights expect."""

_IMAGENET: Normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


RETFOUND: Final = "retfound_mae"
"""Model name for RETFound MAE on colour fundus photos (D-038)."""
RETFOUND_REPO: Final = "YukunZhou/RETFound_mae_natureCFP"
RETFOUND_ARCH: Final = "vit_large_patch16_224"
RETFOUND_LICENCE: Final = "cc-by-nc-4.0"
RETFOUND_DROP_PATH: Final = 0.2
_MAE_ONLY: Final = ("mask_token", "decoder", "head.")


def create_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return a feature extractor (pooled features, no classifier) and its feature width."""
    if name == RETFOUND:
        model = retfound_vit(RETFOUND_ARCH, RETFOUND_DROP_PATH)
        if pretrained:
            load_retfound_state(model, _download_retfound())
    else:
        model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    return model, int(cast(int, model.num_features))


def retfound_vit(arch: str, drop_path: float) -> nn.Module:
    """ViT with global average pooling (RETFound fine-tuning) and ImageNet normalization."""
    model = timm.create_model(
        arch, pretrained=False, num_classes=0, global_pool="avg", drop_path_rate=drop_path
    )
    config = dict(getattr(model, "pretrained_cfg", {}) or {})
    config.update(mean=_IMAGENET[0], std=_IMAGENET[1], license=RETFOUND_LICENCE)
    setattr(model, "pretrained_cfg", config)  # noqa: B010 (nn.Module typing hides the attribute)
    return model


def load_retfound_state(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    """Load MAE encoder weights. Drop decoder keys; the encoder norm initializes fc_norm."""
    encoder = {k: v for k, v in state.items() if not k.startswith(_MAE_ONLY)}
    for key in ("weight", "bias"):
        if f"norm.{key}" in encoder:
            encoder[f"fc_norm.{key}"] = encoder.pop(f"norm.{key}")
    result = model.load_state_dict(encoder, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"RETFound weights do not fit: missing {result.missing_keys[:5]}, "
            f"unexpected {result.unexpected_keys[:5]}"
        )


def _download_retfound() -> Mapping[str, Tensor]:
    import torch
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    files = [f.path for f in HfApi(token=token).list_repo_tree(RETFOUND_REPO)]
    weights = next(p for p in files if p.endswith(".pth"))
    path = hf_hub_download(RETFOUND_REPO, weights, token=token)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    return cast(Mapping[str, Tensor], state)


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

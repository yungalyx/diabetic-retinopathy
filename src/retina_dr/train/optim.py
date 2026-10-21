"""Optimizer parameter groups, with optional layer-wise learning-rate decay for ViTs (D-038)."""

import re
from typing import TypedDict

from torch import nn

_BLOCK = re.compile(r"^backbone\.blocks\.(\d+)\.")
_EMBED = ("backbone.cls_token", "backbone.pos_embed", "backbone.patch_embed", "backbone.reg_token")
_NO_DECAY_NAMES = ("cls_token", "pos_embed", "reg_token")


class ParamGroup(TypedDict):
    params: list[nn.Parameter]
    names: list[str]
    lr: float
    weight_decay: float


def _layer_id(name: str, depth: int) -> int:
    """0 for the embeddings, i + 1 for block i, depth + 1 for the rest (final norm and heads)."""
    if name.startswith(_EMBED):
        return 0
    match = _BLOCK.match(name)
    if match:
        return int(match.group(1)) + 1
    return depth + 1


def parameter_groups(
    model: nn.Module, lr: float, weight_decay: float, layer_decay: float | None
) -> list[ParamGroup]:
    """One group without layer decay. With layer decay: lr * decay ** (depth + 1 - layer id),
    and no weight decay on biases, norms, and position tokens."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if layer_decay is None:
        return [
            {
                "params": [p for _, p in named],
                "names": [n for n, _ in named],
                "lr": lr,
                "weight_decay": weight_decay,
            }
        ]
    blocks = getattr(getattr(model, "backbone", None), "blocks", None)
    depth = len(blocks) if isinstance(blocks, nn.Sequential | nn.ModuleList) else 0
    groups: dict[tuple[int, bool], ParamGroup] = {}
    for name, param in named:
        layer = _layer_id(name, depth)
        no_decay = param.ndim == 1 or name.split(".")[-1] in _NO_DECAY_NAMES
        key = (layer, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "names": [],
                "lr": lr * layer_decay ** (depth + 1 - layer),
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        groups[key]["params"].append(param)
        groups[key]["names"].append(name)
    return list(groups.values())

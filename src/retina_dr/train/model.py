"""One backbone with three heads: referable DR (D-002), severity (D-009), image quality (D-025)."""

from typing import Final

import torch
from torch import Tensor, nn

from retina_dr.train.backbone import create_backbone

QUALITY_PRIOR_LOGIT: Final = -1.5


class DrNet(nn.Module):
    """Binary head: logit. Severity head: continuous grade. Quality head: logit of "reject"."""

    def __init__(self, backbone: nn.Module, features: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(features, 1)
        self.severity_head = nn.Linear(features, 1)
        self.quality_head = nn.Linear(features, 1)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features: Tensor = self.dropout(self.backbone(images))
        binary_logit: Tensor = self.binary_head(features).squeeze(1)
        severity: Tensor = self.severity_head(features).squeeze(1)
        quality_logit: Tensor = self.quality_head(features).squeeze(1)
        return binary_logit, severity, quality_logit


def build_model(name: str, pretrained: bool, dropout: float = 0.3) -> DrNet:
    backbone, features = create_backbone(name, pretrained)
    model = DrNet(backbone, features, dropout)
    with torch.no_grad():
        # Start the severity output near the mean grade so early Huber gradients are sane.
        model.severity_head.bias.fill_(1.0)
        # About 18% of EyeQ images are "reject": start the quality logit near that rate.
        model.quality_head.bias.fill_(QUALITY_PRIOR_LOGIT)
    return model

"""Two-head model (D-002): one backbone, a binary referable-DR head and a severity head."""

import torch
from torch import Tensor, nn

from retina_dr.train.backbone import create_backbone


class TwoHeadNet(nn.Module):
    """Binary head returns a logit. Severity head returns a continuous grade (regression, D-009)."""

    def __init__(self, backbone: nn.Module, features: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(features, 1)
        self.severity_head = nn.Linear(features, 1)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features: Tensor = self.dropout(self.backbone(images))
        binary_logit: Tensor = self.binary_head(features).squeeze(1)
        severity: Tensor = self.severity_head(features).squeeze(1)
        return binary_logit, severity


def build_model(name: str, pretrained: bool, dropout: float = 0.3) -> TwoHeadNet:
    backbone, features = create_backbone(name, pretrained)
    model = TwoHeadNet(backbone, features, dropout)
    with torch.no_grad():
        # Start the severity output near the mean grade so early Huber gradients are sane.
        model.severity_head.bias.fill_(1.0)
    return model

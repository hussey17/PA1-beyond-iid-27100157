"""Seven-class linear PACS classifier."""

from torch import nn


class ClassifierHead(nn.Linear):
    def __init__(self, feature_dim: int = 512, num_classes: int = 7):
        super().__init__(feature_dim, num_classes)

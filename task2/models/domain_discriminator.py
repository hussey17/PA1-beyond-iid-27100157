"""Domain discriminator shared by DANN and CDAN."""

from torch import nn


class DomainDiscriminator(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.5):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

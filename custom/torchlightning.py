from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, optim, Tensor, utils
import pytorch_lightning as pl


class Network(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 3),
            nn.Softmax(dim=1)
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        loss = nn.functional.cross_entropy(z, y)
        self.log("train loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

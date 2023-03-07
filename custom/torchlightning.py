from typing import Any

import torch
from torch import nn, optim, Tensor, utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Network(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.encoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.LeakyReLU(inplace=True),
            nn.Linear(8, 8),
            nn.LeakyReLU(inplace=True),
            nn.Linear(8, 3),
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        loss = nn.functional.cross_entropy(z, y)
        self.log("train loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        loss = nn.functional.cross_entropy(z, y)
        self.log("validation loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.encoder(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        pred_labels = torch.argmax(y_pred, dim=1)
        accuracy = len(pred_labels[pred_labels == y]) / len(pred_labels)
        self.log("test loss", loss)
        self.log("accuracy", accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class TrainEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        pass

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self._run_early_stopping_check(trainer)
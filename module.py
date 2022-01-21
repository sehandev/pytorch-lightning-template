# Standard

# PIP
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, StepLR
import pytorch_lightning as pl

# Custom
from custom.model import CustomModel
import helper.loss as c_loss


class CustomModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = CustomModel(cfg.model)

        self.criterion = self.get_loss_function()
        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_lr_scheduler()

    def get_loss_function(self):
        name = self.cfg.module.criterion.lower()

        if name == 'RMSE'.lower():
            return c_loss.RMSELoss()
        elif name == 'MSE'.lower():
            return nn.MSELoss()
        elif name == 'MAE'.lower():
            return nn.L1Loss()
        elif name == 'CrossEntropy'.lower():
            return nn.CrossEntropyLoss()
        elif name == 'BCE'.lower():
            return nn.BCEWithLogitsLoss()

        raise ValueError(
            f'{name} is not on the custom criterion list!')

    def get_optimizer(self):
        name = self.cfg.optimizer.name.lower()

        if name == 'SGD'.lower():
            return torch.optim.SGD(self.parameters(), lr=self.cfg.optimizer.lr, momentum=self.cfg.optimizer.momentum)
        elif name == 'Adam'.lower():
            return torch.optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr)
        elif name == 'AdamW'.lower():
            return torch.optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)

        raise ValueError(
            f'{name} is not on the custom optimizer list!')

    def get_lr_scheduler(self):
        name = self.cfg.lr_scheduler.name.lower()

        if name == 'None'.lower():
            return None
        if name == 'OneCycleLR'.lower():
            return OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.cfg.optimizer.lr,
                total_steps=self.cfg.trainer.max_epochs,
                anneal_strategy=self.cfg.lr_scheduler.anneal_strategy,
            )
        elif name == 'CosineAnnealingWarmRestarts'.lower():
            return CosineAnnealingWarmRestarts(
                optimizer=self.optimizer,
                T_0=self.cfg.lr_scheduler.T_0,
                T_mult=self.cfg.lr_scheduler.T_mult,
                eta_min=self.cfg.lr_scheduler.eta_min,
            )
        elif name == 'StepLR'.lower():
            return StepLR(
                optimizer=self.optimizer,
                step_size=self.cfg.lr_scheduler.step_size,
                gamma=self.cfg.lr_scheduler.gamma,
            )

        raise ValueError(
            f'{name} is not on the custom scheduler list!')

    def forward(self, x):
        # x : (batch_size, ???)

        out = self.model(x)
        # out : (batch_size, ???)

        return out

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return self.optimizer

        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
        }

    def common_step(self, batch, state):
        x, y = batch
        # x: (batch_size, ???)
        # y: (batch_size, ???)

        x, x_hat = self(x)

        loss = self.criterion(x_hat, x)

        return loss

    def training_step(self, batch, batch_idx):

        loss = self.common_step(batch, state='train')

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):

        loss = self.common_step(batch, state='valid')

        self.log('val_loss', loss, prog_bar=True,
                 sync_dist=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):

        loss = self.common_step(batch, state='test')

        self.log('test_loss', loss, sync_dist=True)

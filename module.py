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
    def __init__(
        self,
        model_option,
        max_epoch,
        learning_rate=1e-2,
        criterion_name='RMSE',
        optimizer_name='Adam',
        lr_scheduler_name='StepLR',
        momentum=0.9,
    ):
        super().__init__()
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.model = CustomModel(model_option)

        self.criterion = self.get_loss_function(criterion_name)
        self.optimizer = self.get_optimizer(optimizer_name)
        self.lr_scheduler = self.get_lr_scheduler(lr_scheduler_name)

    @staticmethod
    def get_loss_function(loss_function_name):
        name = loss_function_name.lower()

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

        raise ValueError(f'{loss_function_name} is not on the custom criterion list!')

    def get_optimizer(self, optimizer_name):
        name = optimizer_name.lower()

        if name == 'SGD'.lower():
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif name == 'Adam'.lower():
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif name == 'AdamW'.lower():
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        raise ValueError(f'{optimizer_name} is not on the custom optimizer list!')

    def get_lr_scheduler(self, scheduler_name):
        name = scheduler_name.lower()

        if name == 'OneCycleLR'.lower():
            return OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.learning_rate,
                total_steps=self.max_epoch,
                anneal_strategy='cos',
            )
        elif name == 'CosineAnnealingWarmRestarts'.lower():
            return CosineAnnealingWarmRestarts(
                optimizer=self.optimizer,
                T_0=30,
                T_mult=1,
                eta_min=self.learning_rate/10000,
            )
        elif name == 'StepLR'.lower():
            return StepLR(
                optimizer=self.optimizer,
                step_size=10,
                gamma=0.1,
            )

        raise ValueError(f'{scheduler_name} is not on the custom scheduler list!')

    def forward(self, x):
        # x : (batch_size, ???)

        out = self.model(x)
        # out : (batch_size, ???)

        return out

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
        }

    def common_step(self, batch, state):
        x, y = batch
        # x: (batch_size, ???)
        # y: (batch_size, ???)

        y_hat = self(x)
        # y_hat: (batch_size, ???)

        loss = 0
        for batch_y_hat, batch_y in zip(y_hat, y):
            batch_loss = self.criterion(batch_y_hat, batch_y)
            loss += batch_loss

        loss /= len(y_hat)

        return loss

    def training_step(self, batch, batch_idx):

        loss = self.common_step(batch, state='train')

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):

        loss = self.common_step(batch, state='valid')

        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):

        loss = self.common_step(batch, state='test')

        self.log('test_loss', loss, sync_dist=True)

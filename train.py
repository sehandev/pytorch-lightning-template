# Standard

# PIP
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Custom
from config import Config
from data_module import CustomDataModule
from module import CustomModule


def train(cfg):
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        name=cfg.logger.name,
    )

    callback_list = []

    if cfg.callbacks.checkpoint:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='./checkout/',
            filename=cfg.logger.project + '-{epoch:04d}-{val_loss:.5f}',
            save_top_k=1,
            mode='min',
        )
        callback_list.append(checkpoint_callback)

    if cfg.callbacks.early_stop:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=cfg.callbacks.patience,
            verbose=False,
            mode='min'
        )
        callback_list.append(early_stop_callback)

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callback_list,
        gpus=cfg.trainer.gpus,
        enable_progress_bar=cfg.trainer.progress_bar,
        max_epochs=cfg.trainer.max_epochs,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        deterministic=cfg.trainer.deterministic,
    )

    data_module = CustomDataModule(cfg)
    module = CustomModule(cfg)

    print('Start model fitting')
    trainer.fit(module, datamodule=data_module)

    print('Start testing')
    trainer.test(datamodule=data_module, ckpt_path='best')

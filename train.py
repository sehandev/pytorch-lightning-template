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


cfg = Config()

wandb_logger = WandbLogger(
    project=cfg.PROJECT_TITLE,
    name=cfg.WANDB_NAME,
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=cfg.EARLYSTOP_PATIENCE,
    verbose=False,
    mode='min'
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./checkout/',
    filename=cfg.PROJECT_TITLE + '-{epoch:04d}-{val_loss:.5f}',
    save_top_k=1,
    mode='min',
)

trainer = Trainer(
    gpus=cfg.GPUS,
    max_epochs=cfg.MAX_EPOCHS,
    logger=wandb_logger,
    progress_bar_refresh_rate=1 if cfg.IS_PROGRESS_LOG_ON else 0,
    accelerator='dp',
    deterministic=True,
    precision=16,
    callbacks=[
        early_stop_callback,
        checkpoint_callback,
    ],
)

data_module = CustomDataModule(
    seq_len=cfg.SEQ_LEN,
    batch_size=cfg.BATCH_SIZE,
    num_workers=cfg.NUM_WORKERS,
)

module = CustomModule(
    model_option=cfg.model_option,
    max_epochs=cfg.MAX_EPOCHS,
    learning_rate=cfg.LEARNING_RATE,
    criterion_name=cfg.CRITERION,
    optimizer_name=cfg.OPTIMIZER,
    lr_scheduler_name=cfg.LR_SCHEDULER,
)

print('Start model fitting')
trainer.fit(module, datamodule=data_module)

print('Start testing')
trainer.test(datamodule=data_module)

print(f'Best model : {checkpoint_callback.best_model_path}')

module = CustomModule.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    model_option=cfg.model_option,
    max_epochs=cfg.MAX_EPOCHS,
    learning_rate=cfg.LEARNING_RATE,
    criterion_name=cfg.CRITERION,
    optimizer_name=cfg.OPTIMIZER,
    lr_scheduler_name=cfg.LR_SCHEDULER,
)

torch.save(module.model.state_dict(), f'./output/{cfg.PROJECT_TITLE}.pt')

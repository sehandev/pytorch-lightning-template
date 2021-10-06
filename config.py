# Standard
from pathlib import Path

# PIP
import pytorch_lightning as pl

# Custom


class Config:
    # User Setting
    SEED = 42

    # Path
    PROJECT_DIR = Path(__file__).parent.absolute()
    DATA_DIR = PROJECT_DIR / 'data'

    # Training
    GPUS = [0, 1, 2, 3]
    MAX_EPOCHS = 20
    EARLYSTOP_PATIENCE = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-2
    CRITERION = 'RMSE'
    OPTIMIZER = 'AdamW'
    LR_SCHEDULER = 'StepLR'

    # Model
    INPUT_SIZE = 2
    D_MODEL = 64
    NUM_LAYER = 3
    DROPOUT = 0.2

    # Dataset
    SEQ_LEN = 10

    # Log
    PROJECT_TITLE = 'Template'
    WANDB_NAME = 'test'
    IS_PROGRESS_LOG_ON = True

    def __init__(self, seed=None):
        self.set_random_seed(seed)
        self.set_model_option()

    def set_random_seed(self, seed):
        if seed:
            self.SEED = seed

        pl.seed_everything(self.SEED)

    def set_model_option(self):
        self.model_option = {
            'input_size': self.INPUT_SIZE,
            'd_model': self.D_MODEL,
            'num_layer': self.NUM_LAYER,
            'dropout': self.DROPOUT,
        }

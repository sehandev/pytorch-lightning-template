# Standard
from pathlib import Path

# PIP
import pytorch_lightning as pl

# Custom


class Config:
    # User Setting
    SEED = 42
    ROOT_DIR = Path(__file__).parent.absolute()
    DATA_DIR = ROOT_DIR/'data'

    # Training
    GPU_LIST = [0, 1]
    NUM_EPOCH = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-2
    CRITERION = 'RMSE'
    OPTIMIZER = 'AdamW'
    LR_SCHEDULER = 'StepLR'

    # Model & Dataset
    INPUT_SIZE = 2
    D_MODEL = 64
    NUM_LAYER = 3
    DROPOUT = 0.2
    SEQ_LEN = 10

    def __init__(self, seed=None):
        self.set_random_seed(seed)

    def set_random_seed(self, seed):
        if seed:
            self.SEED = seed

        pl.seed_everything(self.SEED)

# Standard
from pathlib import Path

# PIP
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

# Custom
from custom.dataset import CustomDataset


class CustomDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        work_dir = Path(cfg.common.work_dir).absolute()
        self.data_dir = work_dir / cfg.dataset.data_dir

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(
            data_dir=self.data_dir,
            seq_len=self.cfg.dataset.seq_len,
        )
        self.valid_dataset = CustomDataset(
            data_dir=self.data_dir,
            seq_len=self.cfg.dataset.seq_len,
        )
        self.test_dataset = CustomDataset(
            data_dir=self.data_dir,
            seq_len=self.cfg.dataset.seq_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data_module.batch_size,
            shuffle=True,
            num_workers=self.cfg.data_module.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.data_module.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_module.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.data_module.batch_size,
            shuffle=False,
            num_workers=self.cfg.data_module.num_workers,
        )

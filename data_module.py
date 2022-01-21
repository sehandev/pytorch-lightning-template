# Standard
from pathlib import Path

# PIP
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from pl_bolts.datasets import DummyDataset

# Custom
from custom.dataset import CustomDataset


class CustomDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        work_dir = Path(cfg.common.work_dir).absolute()
        self.data_dir = work_dir / cfg.data_module.data_dir

        self.set_datasets()

    def set_datasets(self):
        # self.train_dataset = CustomDataset(
        #     seq_len=self.seq_len,
        # )
        # self.valid_dataset = CustomDataset(
        #     seq_len=self.seq_len,
        # )
        # self.test_dataset = CustomDataset(
        #     seq_len=self.seq_len,
        # )
        self.train_dataset = DummyDataset((1, 28, 28), (1,))
        self.valid_dataset = DummyDataset((1, 28, 28), (1,))
        self.test_dataset = DummyDataset((1, 28, 28), (1,))

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

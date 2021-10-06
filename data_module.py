# Standard

# PIP
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

# Custom
from custom.dataset import CustomDataset


class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        seq_len,
        batch_size=1,
        num_workers=0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.set_datasets()

    def set_datasets(self):
        self.train_dataset = CustomDataset(
            seq_len=self.seq_len,
        )
        self.valid_dataset = CustomDataset(
            seq_len=self.seq_len,
        )
        self.test_dataset = CustomDataset(
            seq_len=self.seq_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

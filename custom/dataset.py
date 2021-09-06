# Standard

# PIP
import torch
from torch.utils.data import Dataset

# Custom


class CustomDataset(Dataset):
    def __init__(
        self,
        max_len=10,
    ):
        xs = self.create_sequence(max_len)

        self.xs = torch.tensor(xs, dtype=torch.float)

    def create_sequence(self, max_len):
        sequence = [[i + j for j in range(max_len)] for i in range(max_len)]
        return sequence

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx]


if __name__ == '__main__':
    test_dataset = CustomDataset(3)
    print(f'Length: {len(test_dataset)}')
    print(f'Index 0: {test_dataset[0]}')
    print(f'Index 1: {test_dataset[1]}')

import torch
from torch.utils.data import Dataset

class Pamap2Dataset(Dataset):
    """PyTorch Dataset for windowed PAMAP2 data."""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

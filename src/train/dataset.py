import numpy as np
import torch
from torch.utils.data import Dataset

class Pamap2Dataset(Dataset):
    def __init__(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        block_ids: np.ndarray,
        label_map: dict,
        window_size: int,
        step: int,
    ):
        self.X_data = X_data
        self.y_data = y_data
        self.label_map = label_map
        self.window_size = window_size
        
        # Pre-calculate the start indices of all possible PURE windows
        self.windows = []

        # Get the starting index of each unique, contiguous block
        _, block_start_indices = np.unique(block_ids, return_index=True)

        # The end of one block is the start of the next one.
        # Create a list of end indices by shifting the start indices and appending the total length.
        block_end_indices = np.append(block_start_indices[1:], len(block_ids))

        # Iterate over each block using its start and end index
        for start_idx, end_idx in zip(block_start_indices, block_end_indices):
            num_samples_in_block = end_idx - start_idx

            # Only create windows if the block is long enough
            if num_samples_in_block >= self.window_size:
                for i in range(0, num_samples_in_block - self.window_size + 1, step):
                    self.windows.append(start_idx + i)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start_idx = self.windows[idx]
        end_idx = start_idx + self.window_size

        features = self.X_data[start_idx:end_idx]

        raw_label = self.y_data[start_idx]
        label = self.label_map[raw_label]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

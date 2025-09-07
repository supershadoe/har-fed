import pandas as pd
import torch
from torch.utils.data import Dataset

class Pamap2Dataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_map: dict,
        window_size: int,
        step: int,
    ):
        self.feature_cols = feature_cols
        self.label_map = label_map
        self.window_size = window_size

        # Identify contiguous blocks of the same activity for each subject
        # A new block starts when the activity_id or subject_id changes
        df['block_id'] = (
            (df['subject_id'] != df['subject_id'].shift()) |
            (df['activity_id'] != df['activity_id'].shift())
        ).cumsum()

        # Pre-calculate the start indices of all possible PURE windows
        self.windows = []
        for _, block in df.groupby('block_id'):
            if len(block) >= self.window_size:
                # Get the raw numeric indices for the block
                start_index = block.index[0]
                num_samples_in_block = len(block)
                
                for i in range(0, num_samples_in_block - self.window_size + 1, step):
                    # Store the absolute start index of the window
                    self.windows.append(start_index + i)
        
        self.df = df
        # We can now be certain the label is the same across the whole window
        self.raw_labels = df['activity_id'].values

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start_idx = self.windows[idx]
        end_idx = start_idx + self.window_size

        features = self.df[self.feature_cols].iloc[start_idx:end_idx].values

        raw_label = self.raw_labels[start_idx]
        label = self.label_map[raw_label]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

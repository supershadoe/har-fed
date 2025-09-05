import torch.nn as nn

class CNNModel(nn.Module):
    """1D CNN for Human Activity Recognition."""
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(64 * 58, 128), # 64 filters, 58 sequence length
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        """
        Forward pass for the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, features)
        Returns:
            torch.Tensor: Output tensor of shape (batch, n_classes)
        """
        # PyTorch Conv1D expects (batch, features/channels, seq_len)
        x = x.permute(0, 2, 1)
        return self.layers(x)

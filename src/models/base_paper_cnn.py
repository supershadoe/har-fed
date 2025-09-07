import torch.nn as nn

class BasePaperCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(n_features, 196, kernel_size=16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(196 * 12, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.layers(x)

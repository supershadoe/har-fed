import torch.nn as nn

class SelectLastLSTMOutput(nn.Module):
    def forward(self, x):
        # The input 'x' from nn.LSTM is a tuple: (all_time_step_outputs, (last_hidden_state, last_cell_state))
        output, _ = x
        # We select the output from the very last time step
        return output[:, -1, :]

class HarLstmModel(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LSTM(input_size=n_features, hidden_size=16, batch_first=True),
            SelectLastLSTMOutput(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Forward pass for the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, features)
        Returns:
            torch.Tensor: Output tensor of shape (batch, n_classes)
        """
        return self.layers(x)

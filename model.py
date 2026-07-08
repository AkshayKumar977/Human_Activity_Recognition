import torch.nn as nn
class HAR_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = 9,
            hidden_size = 256,
            num_layers = 2,
            batch_first = True,
            dropout = 0.1
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(256,6)
    def forward(self,x):
        output,(hidden,cell) = self.lstm(x)
        hidden = self.dropout(hidden[-1])
        output = self.fc(hidden)
        return output
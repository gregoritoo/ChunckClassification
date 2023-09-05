import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, input_size, out_size, dropout_ratio, use_dropout=False):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.lin = nn.Linear(input_size, out_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_ratio)
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.bn(x)
        x = self.lin(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.drop(x)
        return x


class MLPModel(nn.Module):
    def __init__(self, input_sizes, dropout_ratio=0.3):
        super().__init__()
        use_dropout = [True] * (len(input_sizes) - 1)
        use_dropout[-1] = False
        self.layers = nn.ModuleList(
            [
                LinearLayer(
                    input_sizes[i],
                    input_sizes[i + 1],
                    dropout_ratio=use_dropout[i],
                    use_dropout=use_dropout,
                )
                for i in range(len(input_sizes) - 1)
            ]
        )
        self.out = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # x = self.out(x)
        return x

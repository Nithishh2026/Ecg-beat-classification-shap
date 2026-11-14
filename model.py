"""Small 1D CNN for single-beat classification.

Notes for SHAP: ReLU uses inplace=False (inplace ops break some explainers) and the
head keeps position information (flatten, not global pooling) so the model can use
*where* in the beat a pattern occurs (e.g. a P wave before the QRS).
"""
import torch
import torch.nn as nn


class BeatCNN(nn.Module):
    def __init__(self, n_classes: int = 3, in_len: int = 180,
                 channels=(16, 32, 64), kernel: int = 7, dropout: float = 0.3):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv1d(c_in, c, kernel, padding=kernel // 2),
                       nn.BatchNorm1d(c),
                       nn.ReLU(inplace=False),
                       nn.MaxPool1d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, in_len)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(flat, n_classes))

    def forward(self, x):  # x: (batch, 1, length)
        return self.head(self.features(x))

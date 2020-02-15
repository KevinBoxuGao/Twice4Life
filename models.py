import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class lstm_model(nn.Module):
    def __init__(self, n_hidden, n_lstm = 1):
        super().__init__()
        self.fc1 = nn.Linear(8192, n_hidden)
        self.lstm = nn.LSTM(n_hidden, n_hidden, n_lstm)
        self.fc2 = nn.Linear(n_hidden, 8192)
        self.hidden = torch.randn(n_lstm, 1, n_hidden).cuda()
        self.cell = torch.randn(n_lstm, 1, n_hidden).cuda()
        self.n_hidden = n_hidden
        self.n_lstm = n_lstm
    
    def forward(self, x):
        # x must have a shape (8192)
        out = self.fc1(x)
        out, (self.hidden, self.cell) = self.lstm(out.view(1, 1, -1), (self.hidden, self.cell))
        out = self.fc2(out)
        return out
    
    def reset(self):
        self.hidden = torch.randn(self.n_lstm, 1, self.n_hidden).cuda()
        self.cell = torch.randn(self.n_lstm, 1, self.n_hidden).cuda()

def generate_seed(data):
    idx = np.random.randint(len(data) - 1)


def generate_song(model, seed, length):
    return None

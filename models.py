import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf

class lstm_model(nn.Module):
    def __init__(self, n_hidden, n_lstm = 1):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(8194, n_hidden),
                                nn.Dropout(0.1))
        self.lstm = nn.LSTM(n_hidden, n_hidden, n_lstm, dropout=0.1)
        self.fc2 = nn.Sequential(nn.Dropout(0.1),
                                nn.Linear(n_hidden, 8194))
        self.hidden = None
        self.cell = None
        self.n_hidden = n_hidden
        self.n_lstm = n_lstm
    
    def forward(self, x):
        # x must have a shape (8192)
        out = self.fc1(x)
        if self.hidden != None:
          out, (self.hidden, self.cell) = self.lstm(out.view(1, 1, -1), (self.hidden, self.cell))
        else:
          out, (self.hidden, self.cell) = self.lstm(out.view(1, 1, -1), None)

        out = self.fc2(out)
        return out
    
    def reset(self):
        self.hidden = torch.randn(self.n_lstm, 1, self.n_hidden).cuda()
        self.cell = torch.randn(self.n_lstm, 1, self.n_hidden).cuda()


def fft(sample, sample_size):
    _fft = [np.fft.fft(x) for x in sample]
    _fft = [pack_complex_array(x[:len(x)//2+1]) for x in _fft]
    return np.array(_fft)
def unpack(_fft):
    ret = []
    [ret.append(_fft[i]+1j*_fft[i+1]) for i in range(0, len(_fft), 2)]
    ret = ret+[x.real-1j*x.imag for x in ret[-2:0:-1]]
    return np.array(ret)
def pack_complex_array(_fft):
    ret = []
    [(ret.append(x.real), ret.append(x.imag)) for x in _fft.tolist()]
    return np.array(ret)
def packed_fft_to_wav(_fft, sample_rate, path):
    sf.write(path, ifft(np.array([unpack(x) for x in _fft.tolist()])), sample_rate)
def ifft(_fft):
    return np.concatenate([[y.real for y in np.fft.ifft(x)] for x in _fft])


def generate_seed():
    song = np.random.randint(1, 4) + 1
    buckets = np.load("npydata/audio" + str(song) + ".npy")
    idx = np.random.randint(len(buckets) - 1)
    return buckets[idx]


def generate_song(model, save_path, n_buckets = 360):
    # 360 for ~3 minutes.
    output = generate_seed()
    song = []
    song.append(output)

    for i in range(n_buckets):
        output = model.forward(output).view(-1)
        song.append(output)
    song = np.array(song).reshape(-1)
    
    packed_fft_to_wav(song, 16000, save_path)

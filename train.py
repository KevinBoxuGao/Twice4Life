!pip install torchaudio
!nvidia-smi -L
!pip install soundfile

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from models import *
import soundfile as sf

# Constants
MODEL_ID    = "1"
DATA_PATH   = "npydata/"
START_EPOCH = 7
LEARN_RATE  = 0.04
N_HIDDEN    = 1024
N_EPOCHS    = 200
SAMPLE      = 20
ON_CUDA     = torch.cuda.is_available()

if ON_CUDA:
    print("GPU available. Training on GPU...")
    device = "cuda:0"
else:
    print("GPU not available. Training on CPU...")
    device = "cpu"

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
    sf.write(path, ifft(np.array([unpack(x) for x in _fft])), sample_rate)
def ifft(_fft):
    return np.concatenate([[y.real for y in np.fft.ifft(x)] for x in _fft])

# Load/Create models
model = lstm_model(N_HIDDEN, n_lstm = 6)
try:
    model.load_state_dict(torch.load("models/generator_" + MODEL_ID + ".pth"))
    print("model found")
except FileNotFoundError:
    torch.save(model.state_dict(), "models/generator_" + MODEL_ID + ".pth")

if ON_CUDA:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = LEARN_RATE)
criterion = nn.MSELoss()

min_loss = np.inf
for epoch in range(START_EPOCH, N_EPOCHS + 1):
    # Load data
    for i in range(1, 3):
        song = torch.Tensor(np.load("npydata/audio" + str(i) + ".npy")).to(device)
        outputs = []
        buckets = fft(song.cpu().numpy(), 8192)
        bucket = torch.Tensor(buckets[0]).to(device)
        print(bucket.shape)
        total_loss = 0
        for j in range(1, len(song) // 2):
            optimizer.zero_grad()
            output = model.forward(bucket).view(-1)
            outputs.append(output.detach().cpu().numpy())

            next_bucket = torch.Tensor(buckets[j]).to(device)
            loss = criterion(output, next_bucket)
            loss.backward(retain_graph = True)
            total_loss += loss.item()
            optimizer.step()
            bucket = next_bucket
        packed_fft_to_wav(outputs, 16000, "out/epoch" + str(epoch) + "_" + str(i) + ".wav")
        model.hidden = torch.randn(model.n_lstm, 1, model.n_hidden).cuda()
        model.cell = torch.randn(model.n_lstm, 1, model.n_hidden).cuda()
        avg_loss = total_loss / (len(song) - 1)
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), "models/generator_" + MODEL_ID + ".pth")
        print("Epoch:", epoch, i)
        print("Loss:", avg_loss) 

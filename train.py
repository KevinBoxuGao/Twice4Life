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
MODEL_ID    = "0"
DATA_PATH   = "npydata/"
START_EPOCH = 2
LEARN_RATE  = 0.01
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
    sf.write(path, ifft(np.array([unpack(x) for x in _fft.tolist()])), sample_rate)
def ifft(_fft):
    return np.concatenate([[y.real for y in np.fft.ifft(x)] for x in _fft])

# Load/Create models
model = lstm_model2(N_HIDDEN, n_lstm = 4)
try:
    model.load_state_dict(torch.load("models/generator_" + MODEL_ID + ".pth"))
except FileNotFoundError:
    torch.save(model.state_dict(), "models/generator_" + MODEL_ID + ".pth")

if ON_CUDA:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = LEARN_RATE)
criterion = nn.MSELoss()

min_loss = -np.inf
for epoch in range(START_EPOCH, N_EPOCHS + 1):
    # Load data
    for i in range(1, 68):
        song = torch.Tensor(np.load("npydata/audio" + str(i) + ".npy")).to(device)
        outputs = []

        bucket = song[0]
        bucket = np.fft.fft(bucket.cpu().numpy())
        bucket = bucket[:4096]
        bucket = torch.Tensor([x.real for x in bucket] + [x.imag for x in bucket]).to(device).view(-1)
        for j in range(1, len(song)):
            optimizer.zero_grad()

            output = model.forward(bucket).view(-1)
            outputs.append(output.detach().cpu().numpy())

            next_bucket = song[j]
            next_bucket = np.fft.fft(next_bucket.cpu().numpy())
            next_bucket = next_bucket[:4096]
            next_bucket = torch.Tensor([x.real for x in next_bucket] + [x.imag for x in next_bucket]).to(device).view(-1)

            loss = criterion(output.view(-1), next_bucket)
            loss.backward(retain_graph = True)
            optimizer.step()
            bucket = next_bucket
        f = fft(outputs, 8192)
        packed_fft_to_wav(f, 16000, "out/epoch" + str(epoch) + "_" + str(i) + ".wav")
        model.reset()
        if loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), "models/generator_" + MODEL_ID + ".pth")
        print("Epoch:", epoch, i)
        print("Loss:", loss.item())

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from models import *


# Constants
MODEL_ID    = "0"
DATA_PATH   = "npydata/"
START_EPOCH = 1
LEARN_RATE  = 0.1
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


# Load/Create models
model = lstm_model(N_HIDDEN)
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
    for i in range(1, 5):
        song = torch.Tensor(np.load("npydata/audio" + str(i) + ".npy"), device = device)
        outputs = []

        bucket = song[0]
        bucket = np.fft.fft(bucket)
        bucket = bucket[:4096]
        bucket = torch.Tensor([x.real for x in bucket] + [x.imag for x in bucket]).view(-1)
        for j in range(1, len(song)):
            optimizer.zero_grad()

            max_bucket = max(bucket).clone()
            bucket /= max_bucket

            output = model.forward(bucket) * max_bucket
            outputs.append(output)

            next_bucket = song[j]
            next_bucket = np.fft.fft(next_bucket)
            next_bucket = next_bucket[:4096]
            next_bucket = torch.Tensor([x.real for x in next_bucket] + [x.imag for x in next_bucket]).view(-1)

            loss = criterion(output.view(-1), next_bucket)
            loss.backward(retain_graph = True)
            optimizer.step()
            print("Step")

            bucket = next_bucket
        for j in range(len(outputs)):
            outputs[j] = torch.Tensor([complex(outputs[j][k], outputs[j][4096:][k]) for k in range(4096)])
            outputs[j] = outputs[j] + (-1*outputs[j])
            outputs[j] = torch.Tensor(np.fft.ifft(outputs[j].numpy()))
            for k in range(len(outputs[j])):
                outputs[j][k] = outputs[j][k].real
        outputs = outputs.view(-1, 1)
        torchaudio.save("out/epoch" + str(epoch) + ".wav", outputs)
        if loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), "models/generator_" + MODEL_ID + ".pth")
        print("Loss:", loss.item())

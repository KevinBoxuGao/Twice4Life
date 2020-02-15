import numpy as np
import os
import torchaudio
import torch

path = os.path.realpath("npydata/audio1.npy")

data = np.load(path)
song = []
for bucket in data:
    dft = np.fft.fft(bucket)
    song.append(np.fft.ifft(dft))
song = np.array(song).reshape(1, -1)
output = torch.Tensor(np.array([i.real for i in song]))
torchaudio.save("testfile1.wav", output, sample_rate=16000)
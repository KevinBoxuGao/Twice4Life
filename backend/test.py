import torch
import torchaudio
import numpy as np
import os
import matplotlib.pyplot as plt

def mix_stereo(waveform):
    return (waveform[:, [0]] + waveform[:, [1]]) / 2
path = os.path.realpath("data/")
for file in os.listdir(path):
    print(file)
    waveform, sample_rate = torchaudio.load(path + "/" + file)

    print("Shape of waveform:", waveform.size())
    print("Sample rate of waveform:", sample_rate)

    audio = mix_stereo(waveform.t().numpy())
    # SAMPLE RATE IS 48kHz, USE 100,000+ TO VIEW ACCURATELY
    dft = np.fft.fft(audio)
    real = [x.real for x in dft]
    imag = [x.imag for x in dft]

    X = 'npydata/' + file[:-4] + '_x.npy'
    np.save(X, real)

    Y = 'npydata/' + file[:-4] + '_y.npy'
    np.save(Y, imag)

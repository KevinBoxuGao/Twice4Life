import torch
import torchaudio
import numpy as np
import os
import matplotlib.pyplot as plt

def mix(audio):
    return (audio[:, [0]] + audio[:, [1]]) / 2

path = os.path.realpath("data/small_data")
files = [path + "/" + file for file in os.listdir(path)]

ctr = 1
for file in files:
    audio, sample_rate = torchaudio.load(file)
    audio = torchaudio.transforms.Resample(orig_freq = sample_rate, new_freq = 16000)(audio).t().numpy()
    audio = mix(audio)
    audio = audio[:len(audio) - len(audio)%8192]
    audio = audio.reshape(len(audio)//8192, 8192)
    np.save("npydata/audio" + str(ctr) + ".npy", audio)
    ctr += 1
    
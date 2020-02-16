import numpy as np
import os
import torchaudio
import torch
import matplotlib.pyplot as plt
import soundfile as sf

path = os.path.realpath("npydata/audio14.npy")
data = np.load(path)
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


f = fft(data, 8192)
print(f.shape)

# plt.figure()
# plt.plot(song)
# plt.show()
#packed_fft_to_wav(f, 16000, "testfile1.wav")
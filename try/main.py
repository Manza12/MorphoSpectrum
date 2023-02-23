from scipy.signal import stft
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt


sf, y = wav.read('nocturne2.wav')
duration = 10
offset = 2.068
N = int(duration * sf)
y = y[int(offset * sf): int(offset * sf) + N]

size = 0.1  # s
nperseg = int(size * sf)

overlap = 90
noverlap = int(nperseg * overlap / 100)

t, f, z = stft(y, fs=sf, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=4*nperseg)

plt.figure()
plt.pcolormesh(f, t, 20*np.log10(np.abs(z)), cmap='hot', vmin=-100, vmax=-20)
# plt.imshow(20*np.log10(np.abs(z)), origin='lower', aspect='auto', cmap='hot', vmin=-100, vmax=-20)
plt.colorbar()
plt.xlim([0, duration])
plt.ylim([0, 5000])

plt.show()

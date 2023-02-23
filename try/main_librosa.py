import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from tftb.processing import reassigned_spectrogram

sr, y = wav.read('nocturne2.wav')
duration = 10
offset = 2.068
N = int(duration * sr)
y = y[int(offset * sr): int(offset * sr) + N]

# Parameters
size = 0.1  # s
win_length = int(size * sr)
n_fft = max(4096, win_length)
time_resolution = 0.01
hop_length = int(time_resolution * sr)

# Show
y_axis = 'linear'

# Spectrogram
fig, ax = plt.subplots()

S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)) / (n_fft / 2)
img = librosa.display.specshow(librosa.amplitude_to_db(S), y_axis=y_axis, x_axis='time', ax=ax, cmap='hot')

ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
# plt.ylim([0, 2000])
plt.ylim([1.35, 2.25])
plt.ylim([700, 900])

# Reassigned

amin = 1e-10
freq, times, mags = reassigned_spectrogram(y, n_fbins=n_fft)

fig, ax = plt.subplots()

img = librosa.display.specshow(librosa.amplitude_to_db(mags / (n_fft / 2)), x_axis="s", y_axis="linear", cmap='hot')

ax.set(title="Reassigned Spectrogram", xlabel=None)
ax.label_outer()

fig.colorbar(img, ax=ax, format="%+2.f dB")
# plt.ylim([0, 2000])
plt.ylim([1.35, 2.25])
plt.ylim([700, 900])

plt.show()

import numpy as np
import librosa as rosa
import torch
from pathlib import Path

# Analysis parameters
FS = 44100  # in Hertz
TIME_RESOLUTION = 0.01  # in seconds
HOP_LENGTH = int(FS * TIME_RESOLUTION)  # in samples
F_MIN = 55.
F_MAX = 20000.
BINS_PER_OCTAVE = 48
N_BINS = int(np.floor(BINS_PER_OCTAVE * np.log2(F_MAX / F_MIN)))
NORM = 1  # Options: 1: L1 norm, 2: L2 norm
WINDOW = ('tukey', 0.5)  # Options: "hann": Hann window, ("tukey", alpha): Tukey window with taper parameter alpha
FREQUENCIES = rosa.core.cqt_frequencies(N_BINS, F_MIN, BINS_PER_OCTAVE)
EPS = np.finfo(np.float32).eps

# GPU parameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path parameters
CWD = Path(__file__).parent.absolute()
AUDIO_PATH = Path('audio')
MIDI_PATH = Path('midi')
SAMPLES_PATH = Path('samples')

# Plot parameters
PLOT_UNITS = False
DPI = 120
V_MIN = -60
V_MAX = 0
V_MIN_MOR = -30
V_MAX_MOR = 0
FULL_SCREEN = True

# MIDI parameters
TICKS_PER_BEAT = 960
BPM = 60

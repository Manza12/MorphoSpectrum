import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import time
from typing import Union

from logs import *

types = [Union]
time.time()
# plt.ion()

# Analysis parameters
FS = 44100  # in Hertz
TIME_RESOLUTION = 0.01  # in seconds
HOP_LENGTH = int(FS * TIME_RESOLUTION)  # in samples
F_MIN = 55. / 2
F_MAX = 20000.
BINS_PER_OCTAVE = 12 * 4
N_BINS = int(np.floor(BINS_PER_OCTAVE * np.log2(F_MAX / F_MIN)))
NORM = 1  # Options: 1: L1 norm, 2: L2 norm
WINDOW = "hann"  # Options:
# "hann": Hann window
# ('tukey', 0.5): Tukey window with taper parameter 0.5
# ("gaussian", 2048)
FREQUENCIES = F_MIN * 2**(np.arange(N_BINS) / BINS_PER_OCTAVE)
EPS = np.finfo(np.float32).eps
NOISE_THRESHOLD = -80  # in dB

# Path parameters
CWD = Path(__file__).parent.absolute()
AUDIO_PATH = Path('audio')
MIDI_PATH = Path('midi')
SAMPLES_INSTRUMENT = 'MyPiano'
SAMPLES_PATH = Path('samples') / Path(SAMPLES_INSTRUMENT)
SAMPLES_AUDIO_PATH = SAMPLES_PATH / Path('audio')
SAMPLES_ARRAYS_PATH = SAMPLES_PATH / Path('arrays')
SAMPLES_IMAGES_PATH = SAMPLES_PATH / Path('images')
SAMPLES_INFO_PATH = SAMPLES_PATH / Path('info')

# Plot parameters
BACKEND = 'TkAgg'  # 'WXAgg'
PLOT_UNITS = False
DPI = 120
C_MAP = 'hot'
V_MIN = NOISE_THRESHOLD
V_MAX = 0
V_MIN_MOR = -30
V_MAX_MOR = 0
FULL_SCREEN = True
TIME_FORMAT = 'milliseconds'  # '%M:%S'
TIME_LABEL = 'Time (mm:ss,ms)'  # '(mm:ss)'

plt.switch_backend(BACKEND)

# MIDI parameters
TICKS_PER_BEAT = 960
BPM = 60

# Samples
N_PARTIALS = 8
F_REF = 440
NUMBER_REF = 69
NUMBER_F_MIN = NUMBER_REF - 12 * np.log2(F_REF / F_MIN).astype(int)
PARTIALS_DISTRIBUTION_TYPE = "linear"
LOAD_ALL = True
USE_CQT = True
MASTER_VOLUME = 0.1

# GPU parameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if USE_CQT:
    log.info('Using device: ' + str(DEVICE))

# Logs
if __name__ == '__main__':
    configure_logs('parameters')

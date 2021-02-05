import scipy.io.wavfile as wav
import os.path as path
from parameters import *


def signal_from_file(file_name, audio_path=AUDIO_PATH):
    file_path = path.join(audio_path, file_name + ".wav")
    fs, signal = wav.read(file_path)

    if signal.dtype == np.int16:
        signal_float = signal / np.iinfo(signal.dtype).max
    else:
        Exception("Signal dtype not int16")

    if not len(signal_float.shape) == 1:
        raise Exception("Stereo not handled.")

    assert fs == FS

    return signal_float


def get_time_vector(signal):
    return np.arange(np.ceil(signal.size / HOP_LENGTH).astype(int)) / (FS / HOP_LENGTH)

import scipy.io.wavfile as wav
import warnings as warn
from parameters import *


warn.simplefilter("ignore", wav.WavFileWarning)


def signal_from_file(file_name, audio_path=AUDIO_PATH):
    file_path = Path(audio_path) / Path(file_name + ".wav")
    fs, signal = wav.read(file_path)

    if signal.dtype == np.int16:
        signal = signal / np.iinfo(signal.dtype).max
    else:
        Exception("Signal dtype not int16")

    if not len(signal.shape) == 1:
        raise Exception("Stereo not handled.")

    assert fs == FS

    return signal


def get_time_vector(signal):
    return np.arange(np.ceil(signal.size / HOP_LENGTH).astype(int)) / (FS / HOP_LENGTH)

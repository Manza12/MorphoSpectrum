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


def generate_dirac(duration: Union[int, float], onset: Union[int, float], height: Union[int, float],
                   numpy: bool = True):
    """ Generate a dirac of impulse

    Parameters
    ----------
    :param duration: float
        Duration of the signal.
    :param onset: float
        Onset of the dirac impulse respect to the start of the signal.
    :param height: float
         Value of the signal in the onset.
    :param numpy: bool
         If True, returns a numpy array. Else, returns a PyTorch tensor.
    Outputs
    -------
    :return signal: Numpy array or PyTorch tensor
        Signal of the dirac impulse.
    """

    assert type(duration) in [int, float], "Duration should be an integer or a float number."
    assert type(onset) in [int, float], "Onset should be an integer or a float number."
    assert type(height) in [int, float], "Height should be a float number."

    assert duration > 0
    assert 0 < onset < duration
    assert 0 < height

    signal = np.zeros(int(duration * FS))
    signal[int(onset * FS)] = height

    if numpy:
        return signal
    else:
        return torch.tensor(signal)

from parameters import *


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


def ticks2seconds(ticks):
    seconds = ((ticks / TICKS_PER_BEAT) / (BPM / 60))
    return seconds


def velocity2db(velocity):
    db = - NOISE_THRESHOLD * (velocity - 127) / 127
    return db


def velocity2amplitude(velocity, scale: str = 'lin'):
    if scale == 'lin':
        return velocity / 127
    elif scale == 'log':
        db = velocity2db(velocity)
        amplitude = 10**(db / 20)
        return amplitude
    else:
        raise ValueError('Invalid parameter scale.')

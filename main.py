from midi import midi2piece
from plots import plot_cqt
from samples import get_samples_set
from time_frequency import cqt
from signals import *
from parameters import *
import time
import sounddevice as sd


def example_1():
    _sta = time.time()
    _signal = signal_from_file('anastasia')
    _end = time.time()
    log.info("Time to recover signal: " + str(round(_end - _sta, 3)) + " seconds.")

    _sta = time.time()
    _spectrogram, _time_vector = cqt(_signal)
    _end = time.time()
    log.info("Time compute the CQT: " + str(round(_end - _sta, 3)) + " seconds.")

    _sta = time.time()
    plot_cqt(_spectrogram, _time_vector)
    _end = time.time()
    log.info("Time to plot: " + str(round(_end - _sta, 3)) + " seconds.")


if __name__ == '__main__':
    # Parameters
    play = True

    # Create the signal
    sta = time.time()
    samples_set = get_samples_set('basic')
    end = time.time()
    log.info("Time to create samples set: " + str(round(end - sta, 3)) + " seconds.")

    sta = time.time()
    piece = midi2piece('prelude_em')
    signal = samples_set.synthesize(piece)
    end = time.time()
    log.info("Time to synthesize the signal: " + str(round(end - sta, 3)) + " seconds.")

    # Time-frequency transform of the signal
    sta = time.time()
    spectrogram, time_vector = cqt(signal)
    end = time.time()
    log.info("Time to compute the CQT of the signal: " + str(round(end - sta, 3)) + " seconds.")

    if play:
        sd.play(signal, FS)
    plot_cqt(spectrogram, time_vector)

    # Morphological transform of the signal


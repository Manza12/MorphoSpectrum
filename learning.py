from midi import midi2piece
from plots import plot_cqt
from samples import get_samples_set
from time_frequency import cqt
from signals import *
import time
import sounddevice as sd

if __name__ == '__main__':
    # Parameters
    play = False

    # Create the signal
    sta = time.time()
    samples_set = get_samples_set('basic')
    end = time.time()
    log.info("Time to create samples set: " + str(round(end - sta, 3)) + " seconds.")

    sta = time.time()
    piece = midi2piece('tempest_3rd-start')
    signal = samples_set.synthetize(piece)
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

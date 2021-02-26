from plots import plot_cqt
from time_frequency import cqt
from utils import signal_from_file
from signals import signal_from_midi
from parameters import *
import time


signal_from = 'midi'  # 'file'

if __name__ == '__main__':
    sta = time.time()

    if signal_from == 'file':
        signal = signal_from_file('anastasia')
    elif signal_from == 'midi':
        signal = signal_from_midi('prelude_543')
    else:
        raise Exception('Invalid signal from.')

    end = time.time()
    log.info("Time to recover signal: " + str(round(end - sta, 3)) + " seconds.")

    sta = time.time()
    spectrogram, time_vector = cqt(signal)
    end = time.time()
    log.info("Time compute the CQT: " + str(round(end - sta, 3)) + " seconds.")

    sta = time.time()
    plot_cqt(spectrogram, time_vector)
    end = time.time()
    log.info("Time to plot: " + str(round(end - sta, 3)) + " seconds.")

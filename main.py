from plots import plot_cqt
from time_frequency import cqt
from signals import *
import time


if __name__ == '__main__':
    sta = time.time()
    signal = signal_from_file('anastasia')
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

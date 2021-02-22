from plots import plot_cqt
from time_frequency import cqt
from signals import *
import time


if __name__ == '__main__':
    sta = time.time()
    signal = signal_from_file('anastasia')
    time_vector = get_time_vector(signal)
    end = time.time()
    log.info("Time to recover signal: " + str(round(end - sta, 3)) + " seconds.")

    sta = time.time()
    spectrogram = cqt(signal)
    spectrogram_log = 20 * np.log10(spectrogram + EPS)
    end = time.time()
    log.info("Time compute the CQT: " + str(round(end - sta, 3)) + " seconds.")

    sta = time.time()
    plot_cqt(spectrogram_log, time_vector)
    end = time.time()
    log.info("Time to plot: " + str(round(end - sta, 3)) + " seconds.")

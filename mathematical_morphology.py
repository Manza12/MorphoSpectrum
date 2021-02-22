from parameters import *
from scipy.ndimage import grey_erosion

from plots import plot_cqt
from signals import get_time_vector, signal_from_file
from time_frequency import cqt


def erode(image, strel, origin):
    erosion = grey_erosion(image, origin=origin, structure=strel, mode='constant')
    return erosion


if __name__ == '__main__':
    _strel_1 = np.array([[-6], [0], [-6]])
    _origin_1 = [0, 0]
    _strel_2 = np.array([[-10], [0], [-10]])
    _origin_2 = [0, 0]

    sample_name = "A2_12.787_113"
    start = 0  # in seconds
    end = 25  # in seconds
    _signal = signal_from_file(sample_name, SAMPLES_AUDIO_PATH)
    _spectrogram, _time_vector = cqt(_signal)[:, np.floor(start / TIME_RESOLUTION).astype(int): np.ceil(end / TIME_RESOLUTION).astype(int)]

    plot_cqt(_spectrogram, _time_vector, fig_title=sample_name)

    a_erosion_1 = erode(_spectrogram, _strel_1, _origin_1)
    plot_cqt(a_erosion_1, _time_vector, fig_title="Erosion - 1", v_min=-40, c_map='Greys')

from parameters import *
import scipy.signal.windows as win


def gaussian_db(length, db_attenuation=60, sym=True):
    assert db_attenuation >= 0
    sigma = np.floor(- length / 2 / np.sqrt(- 2 * np.log(10 ** (- db_attenuation / 20))))
    return win.gaussian(length, sigma, sym=sym)


def to_db(power):
    return 20 * np.log10(power + EPS)


def db_to_amplitude(power_db):
    return 10**(power_db / 20)


def db_to_velocity(power_db):
    return int(db_to_amplitude(power_db) * 128)


def ticks2seconds(ticks):
    seconds = ((ticks / TICKS_PER_BEAT) / (BPM / 60))
    return seconds

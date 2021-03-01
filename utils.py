from parameters import *


def to_db(power):
    return 20 * np.log10(power + EPS)


def db_to_amplitude(power_db):
    return 10**(power_db / 20)


def db_to_velocity(power_db):
    return int(db_to_amplitude(power_db) * 128)


def ticks2seconds(ticks):
    seconds = ((ticks / TICKS_PER_BEAT) / (BPM / 60))
    return seconds

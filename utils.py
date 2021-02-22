from parameters import *


def ticks2seconds(ticks):
    seconds = ((ticks / TICKS_PER_BEAT) / (BPM / 60))
    return seconds

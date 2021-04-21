from parameters import *
import sounddevice as sd


def play_signal(signal, master_volume=MASTER_VOLUME):
    sd.play(signal * master_volume, FS)

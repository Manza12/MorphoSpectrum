from parameters import *
import music21 as m21


def ticks2seconds(ticks):
    seconds = ((ticks / TICKS_PER_BEAT) / (BPM / 60))
    return seconds


class Piece(list):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __str__(self):
        result = ""
        result += "Piece: " + self.name
        result += "Notes:\n"
        for i in range(len(self)):
            result += self[i].__str__() + "\n"

        return result

    def add_note(self, note):
        assert isinstance(note, Note)
        self.append(note)


class Note:
    def __init__(self, note_number, velocity, start_seconds, end_seconds):
        self.note_number = note_number
        self.velocity = velocity
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        self.duration = end_seconds - start_seconds

        self.pitch = m21.pitch.Pitch(midi=note_number)

    @classmethod
    def from_midi(cls, note_number, velocity, start_ticks, end_ticks):
        start_seconds = ticks2seconds(start_ticks)
        end_seconds = ticks2seconds(end_ticks)

        return cls(note_number, velocity, start_seconds, end_seconds)

    def __str__(self, name=True, time=True, velocity=False):
        result = ""

        if name:
            result += self.pitch.unicodeNameWithOctave
        if time:
            result += ", start: " + str(round(self.start_seconds, 3)) + ", duration: " + str(round(self.duration, 3))
        if velocity:
            result += ", velocity: " + str(self.velocity)
        return result

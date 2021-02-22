import music21 as m21
import collections.abc as abc
from utils import ticks2seconds
from parameters import log, configure_logs


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

    def __str__(self, name_str=True, time_str=True, velocity_str=False):
        result = ""

        if name_str:
            result += self.pitch.unicodeNameWithOctave
        if time_str:
            result += ", start: " + str(round(self.start_seconds, 3)) + ", duration: " + str(round(self.duration, 3))
        if velocity_str:
            result += ", velocity: " + str(self.velocity)
        return result


class Piece(abc.MutableSequence):
    def __init__(self, name: str = None):
        self.name = name
        self._notes_list = list()

    def insert(self, index: int, value: Note) -> None:
        if not type(value) is Note:
            raise TypeError("%r should be a Note" % value)
        self._notes_list.insert(index, value)

    def append(self, value: Note) -> None:
        if not type(value) is Note:
            raise TypeError("%r should be a Note" % value)
        self._notes_list.append(value)

    def __len__(self) -> int:
        return len(self._notes_list)

    def __delitem__(self, index: int) -> None:
        self._notes_list.__delitem__(index)

    def __setitem__(self, index: int, value: Note) -> None:
        if not type(value) is Note:
            raise TypeError("%r should be a Note" % value)
        self._notes_list.__setitem__(index, value)

    def __getitem__(self, index: int) -> Note:
        return self._notes_list.__getitem__(index)

    def __str__(self) -> str:
        result = ""
        result += "Piece: "

        if self.name:
            result += self.name + "\n"
        else:
            result += "[no name]\n"

        result += "Notes:\n"
        for i in range(len(self)):
            result += self[i].__str__() + "\n"

        return result


if __name__ == '__main__':
    configure_logs('music')

    _piece = Piece("Example")
    _note = Note(69, 100, 0., 1.)
    _piece.append(_note)

    log.info(_piece)
    log.info("Length of " + _piece.name + ": " + str(len(_piece)))

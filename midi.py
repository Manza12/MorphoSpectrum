import mido as mid
from parameters import *
from music import Note, Piece


def print_messages(midi):
    for i, track in enumerate(midi.tracks):
        print('Track {}: {}'.format(i, track.name))
        for msg in track:
            print(msg)


def check_pedal(midi):
    for msg in midi.tracks[0]:
        if msg.type == 'control_change':
            if msg.control == 64:
                return True
            else:
                Exception("Control change not handled")
    return False


def midi2piece(file_name):
    piece = Piece(file_name)
    file_path = MIDI_PATH / Path(file_name + '.mid')
    midi = mid.MidiFile(file_path)

    has_pedal = check_pedal(midi)

    if not has_pedal:
        time_ticks = 0
        for m, msg in enumerate(midi.tracks[0]):
            time_ticks += msg.time
            if msg.type == 'note_on':
                if msg.velocity != 0:
                    m_end = m + 1
                    delta_ticks = 0
                    while True:
                        delta_ticks += midi.tracks[0][m_end].time
                        if midi.tracks[0][m_end].note == msg.note and midi.tracks[0][m_end].velocity == 0:
                            note = Note.from_midi(msg.note, msg.velocity, time_ticks, time_ticks + delta_ticks)
                            piece.append(note)
                            break
                        m_end += 1
    else:
        raise Exception("Pedal not integrated yet")

    return piece


if __name__ == '__main__':
    _file_name = 'fugue_543'
    _piece = midi2piece(_file_name)

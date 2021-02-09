from midi import midi2piece
from parameters import *
from signals import signal_from_file, wav
from tqdm import tqdm


def parse_samples(samples_name, resonance_seconds=0.3):
    piece = midi2piece(samples_name)
    signal = signal_from_file(samples_name)

    for note in tqdm(piece.notes):
        start_samples = np.floor(note.start_seconds * FS).astype(int)
        end_samples = np.ceil((note.end_seconds + resonance_seconds) * FS).astype(int)
        note_signal = signal[start_samples:end_samples]
        output_name = note.pitch.nameWithOctave + "_" + str(round(note.duration, 3)) + "_" + str(note.velocity)
        wav.write(Path(SAMPLES_PATH) / Path(output_name + '.wav'), FS, note_signal)


if __name__ == '__main__':
    _samples_name = 'samples'
    parse_samples(_samples_name)

from midi import midi2piece
from parameters import *
from plots import plot_cqt
from signals import signal_from_file, wav, get_time_vector
from tqdm import tqdm
from time_frequency import cqt


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
    # _samples_name = 'samples'
    # parse_samples(_samples_name)

    sample_name = "A2_12.787_113"
    start = 0  # in seconds
    end = 25  # in seconds
    _signal = signal_from_file(sample_name, SAMPLES_PATH)
    _spectrogram = cqt(_signal, numpy=True)[:, np.floor(start / TIME_RESOLUTION).astype(int): np.ceil(end / TIME_RESOLUTION).astype(int)]
    _spectrogram_log = 20 * np.log10(_spectrogram + EPS)
    _time_vector = get_time_vector(_signal)
    plot_cqt(_spectrogram_log, _time_vector, fig_title=sample_name)

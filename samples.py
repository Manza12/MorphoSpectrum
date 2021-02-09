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

    # A0: ["A0_2.523_33", "A0_10.846_62", "A0_22.086_98"], [33, 62, 98]
    sample_names = ["A2_10.487_78", "A2_12.787_113"]
    velocities = [78, 113]

    spectrogram_list = list()
    for i, name in enumerate(sample_names):
        _signal = signal_from_file(name, SAMPLES_PATH)
        _time_vector = get_time_vector(_signal)
        _spectrogram = cqt(_signal, numpy=True)[:, 0:200]
        _spectrogram_log = 20 * np.log10(_spectrogram + EPS)
        spectrogram_list.append(_spectrogram_log)
        plot_cqt(_spectrogram_log, _time_vector)

    a = 0
    b = 1
    diff = spectrogram_list[a] - spectrogram_list[b]
    print(str(velocities[a]) + "," + str(velocities[b]) + "," + str(np.median(diff)))

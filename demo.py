from midi import midi2piece
from music import Piece, Note
from plots import plot_cqt
from samples import get_samples_set
from time_frequency import cqt
import sounddevice as sd
from mathematical_morphology import erode, dilate, closing, opening
from parameters import *
from utils import to_db, db_to_velocity
from librosa import hz_to_midi
from tqdm import tqdm

samples_set = get_samples_set('basic')

piece = midi2piece('prelude_em')

signal = samples_set.synthetize(piece)
spectrogram, time_vector = cqt(signal)

plot_cqt(spectrogram, time_vector)

# sd.play(signal, FS)

# Leakage
structural_element_leakage = np.array([[-5.8], [0], [-6]])
origin_leakage = [0, 0]

erosion_leakage = erode(spectrogram, structural_element_leakage, origin_leakage)
plot_cqt(erosion_leakage, time_vector, fig_title="Erosion of the leakage")

# Harmonic
partials_power_db = to_db(samples_set.partials_distribution.partial_power)
partials_height = np.round(np.log2(np.arange(N_PARTIALS) + 1) * BINS_PER_OCTAVE).astype(int)

strel_harmonic = np.zeros((partials_height.max() + 1, 1)) - 1000
origin_harmonic = [- strel_harmonic.shape[0] // 2 + 1, 0]
for i in range(partials_power_db.size):
    strel_harmonic[partials_height[i], 0] = partials_power_db[i]

# structural_element_harmonic = np.expand_dims(partials_power_db, 1)
# origin_harmonic = [- structural_element_harmonic.size // 2 + 1, 0]

erosion_harmonic = erode(erosion_leakage, strel_harmonic, origin_harmonic)
plot_cqt(erosion_harmonic, time_vector, fig_title="Erosion of the harmonics", c_map='Greys')

# # Threshold
# threshold = -100
# erosion_harmonic[erosion_harmonic < threshold] = -1000
# plot_cqt(erosion_harmonic, time_vector, fig_title="Thresholding to " + str(threshold) + " dB", c_map='Greys')

# Closing amplitude modulations
strel_closing = np.zeros((1, 7))

closing_modulation = closing(erosion_harmonic, strel_closing)
plot_cqt(closing_modulation, time_vector, fig_title="Closing amplitude modulations", c_map='Greys')

# Opening to detect of notes
strel_opening = np.zeros((1, 20))

opening_detection = opening(closing_modulation, strel_opening)
plot_cqt(opening_detection, time_vector, fig_title="Detection of onsets/offsets", c_map='Greys')

# Threshold detection
threshold_detection = -30
opening_threshold = opening_detection
opening_threshold[opening_threshold < threshold_detection] = -1000
plot_cqt(opening_threshold, time_vector, fig_title="Threshold detection", c_map='Greys')

# Dilation to piano roll
strel_piano_roll = np.zeros((3, 1))
origin_piano_roll = [0, 0]

piano_roll = dilate(opening_threshold, strel_piano_roll, origin=origin_piano_roll)
plot_cqt(piano_roll, time_vector, fig_title="Piano roll", c_map='Greys')

# Binary piano roll
plot_cqt(piano_roll, time_vector, fig_title="Binary piano roll", v_min=threshold_detection - EPS,
         v_max=threshold_detection, c_map='Greys')

# Note detection
# First, close the onsets to have the initial amplitude well set
strel_onsets = np.zeros((1, 3))
close_onsets = closing(opening_threshold, strel_onsets)

# TODO: this shall not be needed
# Erase frame due to not managing the origin well
close_onsets[close_onsets == 0] = -1000

# Plot
plot_cqt(close_onsets, time_vector, fig_title="Close onset amplitude", c_map='Greys')

# Initialisation
frequency = None
velocity = None
start_seconds = None

# THE loop
piece_reconstructed = Piece(piece.name + ' reconstructed')

for f in tqdm(range(close_onsets.shape[0])):
    for t in range(close_onsets.shape[1] - 1):
        if close_onsets[f, t] < threshold_detection <= close_onsets[f, t + 1]:
            # Note on
            frequency = int(round(hz_to_midi(FREQUENCIES[f])))
            velocity = db_to_velocity(close_onsets[f, t + 1])
            start_seconds = float(time_vector[t+1])
        elif close_onsets[f, t] >= threshold_detection > close_onsets[f, t + 1]:
            # Note off
            # TODO: decide between t and t+1 index to note off
            end_seconds = float(time_vector[t+1])
            note = Note(frequency, velocity=velocity, start_seconds=start_seconds, end_seconds=end_seconds)
            piece_reconstructed.append(note)

# Re-synthesize piece
signal_reconstructed = samples_set.synthetize(piece_reconstructed)
sd.play(signal_reconstructed, FS)

if __name__ == '__main__':
    pass

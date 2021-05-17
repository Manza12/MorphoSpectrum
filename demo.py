from midi import midi2piece
from music import Piece, Note
from plots import plot_cqt
from samples import get_samples_set
from time_frequency import cqt
import sounddevice as sd
from parameters import *
from utils import to_db, db_to_velocity
from librosa import hz_to_midi
from tqdm import tqdm
from nnMorpho.operations import erosion, dilation, opening, closing

samples_set = get_samples_set('basic')

piece = midi2piece('prelude_em')

signal = samples_set.synthesize(piece)
spectrogram, time_vector = cqt(signal, numpy=False)

plot_cqt(spectrogram, time_vector, numpy=False)

# sd.play(signal, FS)

# Leakage
structural_element_leakage = torch.tensor([[-5.8], [0], [-6]], device=DEVICE)
origin_leakage = (1, 0)

erosion_leakage = erosion(spectrogram, structural_element_leakage, origin_leakage, border_value='euclidean')
plot_cqt(erosion_leakage, time_vector, fig_title="Erosion of the leakage", numpy=False)

# Harmonic
partials_power_db = to_db(samples_set.partials_distribution.partial_power)
partials_height = np.round(np.log2(np.arange(N_PARTIALS) + 1) * BINS_PER_OCTAVE).astype(int)

strel_harmonic = torch.zeros((partials_height.max() + 1, 1), device=DEVICE) - 1000
origin_harmonic = (0, 0)
for i in range(partials_power_db.size):
    strel_harmonic[partials_height[i], 0] = partials_power_db[i]

# structural_element_harmonic = np.expand_dims(partials_power_db, 1)
# origin_harmonic = [- structural_element_harmonic.size // 2 + 1, 0]

erosion_harmonic = erosion(erosion_leakage, strel_harmonic, origin=origin_harmonic, border_value='euclidean')
plot_cqt(erosion_harmonic, time_vector, fig_title="Erosion of the harmonics", c_map='Greys', numpy=False)

# # Threshold
# threshold = -100
# erosion_harmonic[erosion_harmonic < threshold] = -1000
# plot_cqt(erosion_harmonic, time_vector, fig_title="Thresholding to " + str(threshold) + " dB", c_map='Greys')

# Closing amplitude modulations
strel_closing = torch.zeros((1, 7), device=DEVICE)

closing_modulation = closing(erosion_harmonic, strel_closing, origin=(0, 3), border_value='euclidean')
plot_cqt(closing_modulation, time_vector, fig_title="Closing amplitude modulations", c_map='Greys', numpy=False)

# Opening to detect of notes
strel_opening = torch.zeros((1, 21), device=DEVICE)

opening_detection = opening(closing_modulation, strel_opening, origin=(0, 10), border_value='euclidean')
plot_cqt(opening_detection, time_vector, fig_title="Detection of onsets/offsets", c_map='Greys', numpy=False)

# Threshold detection
threshold_detection = -30
opening_threshold = opening_detection
opening_threshold[opening_threshold < threshold_detection] = -1000
plot_cqt(opening_threshold, time_vector, fig_title="Threshold detection", c_map='Greys', numpy=False)

# Dilation to piano roll
strel_piano_roll = torch.zeros((3, 1), device=DEVICE)
origin_piano_roll = (1, 0)

piano_roll = dilation(opening_threshold, strel_piano_roll, origin=origin_piano_roll, border_value='euclidean')
plot_cqt(piano_roll, time_vector, fig_title="Piano roll", c_map='Greys', numpy=False, v_min=threshold_detection * 1.01, v_max=threshold_detection)

# Binary piano roll
plot_cqt(piano_roll, time_vector, fig_title="Binary piano roll", c_map='Greys', numpy=False)

# Note detection
# First, close the onsets to have the initial amplitude well set
strel_onsets = torch.zeros((1, 3), device=DEVICE)
close_onsets = closing(opening_threshold, strel_onsets)

# TODO: this shall not be needed
# Erase frame due to not managing the origin well
close_onsets[close_onsets == 0] = -1000

# Plot
plot_cqt(close_onsets, time_vector, fig_title="Close onset amplitude", c_map='Greys', numpy=False)

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
signal_reconstructed = samples_set.synthesize(piece_reconstructed)
sd.play(signal_reconstructed, FS)

if __name__ == '__main__':
    pass

from mathematical_morphology import erode
from parameters import *
from plots import plot_cqt
from signals import signal_from_file
from time_frequency import cqt

# Structural element for frequency leakage erosion
_strel_1 = np.array([[-6], [0], [-6]])
_origin_1 = [0, 0]

# Samples to use
sample_name = "A2_12.787_113"

# Partials
fundamental_bin = 2 * BINS_PER_OCTAVE

partials_pos = fundamental_bin + np.round(np.log2(np.arange(N_PARTIALS) + 1) * BINS_PER_OCTAVE).astype(int)

start = 2  # in seconds
end = 6  # in seconds
_signal = signal_from_file(sample_name, SAMPLES_AUDIO_PATH)
_spectrogram, _time_vector = cqt(_signal, numpy=True)[:, np.floor(start / TIME_RESOLUTION).astype(int): np.ceil(end / TIME_RESOLUTION).astype(int)]
plot_cqt(_spectrogram, _time_vector, fig_title=sample_name)

a_erosion_1 = erode(_spectrogram, _strel_1, _origin_1)
plot_cqt(a_erosion_1, _time_vector, fig_title="Erosion - 1", v_min=-40, c_map='Greys')

# Spectrogram
partials_amplitudes = _spectrogram[partials_pos, :]
partials_distribution = partials_amplitudes[:, 0] - np.max(partials_amplitudes[:, 0])

plt.figure()
for i in range(len(partials_pos)):
    plt.plot(_time_vector, partials_amplitudes[i, :])

strel_harmonic = np.zeros(partials_pos.max() - partials_pos.min() + 1) - 100
origin_harmonic = [- strel_harmonic.size // 2 + 1, 0]

for i in range(partials_distribution.size):
    strel_harmonic[partials_pos[i] - partials_pos.min()] = partials_distribution[i]

plt.figure()
plt.scatter(np.arange(partials_distribution.size), partials_distribution)

strel_harmonic = np.expand_dims(strel_harmonic, axis=1)

a_erosion_harmonic = erode(a_erosion_1, strel_harmonic, origin_harmonic)
plot_cqt(a_erosion_harmonic, _time_vector, fig_title="Erosion - harmonic", v_min=-60, c_map='Greys')

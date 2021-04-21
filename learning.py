from midi import midi2piece
from plots import plot_cqt
from samples import get_samples_set
from time_frequency import cqt
from signals import *
from scipy.signal.windows import get_window
from nnMorpho.operations import partial_erosion
import time
import sounddevice as sd

# Parameters
play = True


# Function for partial erosion
def generate_strel():
    from time_frequency import cqt_layer
    lengths = cqt_layer.lenghts.cpu().numpy()
    kernel_width = int(np.max(lengths))
    tempKernel = np.zeros((int(N_BINS), kernel_width), dtype=np.float32)

    for k in range(0, int(N_BINS)):
        l = lengths[k]

        # Centering the kernels
        if l % 2 == 1:  # pad more zeros on RHS
            start = int(np.ceil(kernel_width / 2.0 - l / 2.0)) - 1
        else:
            start = int(np.ceil(kernel_width / 2.0 - l / 2.0))

        sig = get_window_dispatch(WINDOW, int(l), fftbins=True)

        tempKernel[start:start + int(l), :] = sig / np.max(sig)

    return tempKernel


def get_window_dispatch(window, N, fftbins=True):
    if isinstance(window, str):
        return get_window(window, N, fftbins=fftbins)
    elif isinstance(window, tuple):
        if window[0] == 'gaussian':
            assert window[1] >= 0
            sigma = np.floor(- N / 2 / np.sqrt(- 2 * np.log(10 ** (- window[1] / 20))))
            return get_window(('gaussian', sigma), N, fftbins=fftbins)
        else:
            Warning("Tuple windows may have undesired behaviour regarding Q factor")
    elif isinstance(window, float):
        Warning("You are using Kaiser window with beta factor " + str(window) + ". Correct behaviour not checked.")
    else:
        raise Exception("The function get_window from scipy only supports strings, tuples and floats.")


if __name__ == '__main__':
    configure_logs('learning')

    # Create the signal
    sta = time.time()
    samples_set = get_samples_set('basic', decay=0.9)
    end = time.time()
    log.info("Time to create samples set: " + str(round(end - sta, 3)) + " seconds.")

    sta = time.time()
    piece = midi2piece('tempest_3rd-start')
    signal = signal_from_file('tempest_3rd-start')  # samples_set.synthesize(piece)
    end = time.time()
    log.info("Time to synthesize the signal: " + str(round(end - sta, 3)) + " seconds.")

    # Time-frequency transform of the signal
    sta = time.time()
    spectrogram, time_vector = cqt(signal, numpy=False)
    end = time.time()
    log.info("Time to compute the CQT of the signal: " + str(round(end - sta, 3)) + " seconds.")

    if play:
        sd.play(signal * MASTER_VOLUME, FS)
    plot_cqt(spectrogram, time_vector, show=False, numpy=False)

    # Morphological transform of the signal
    dirac_duration = 0.5
    dirac = generate_dirac(dirac_duration, dirac_duration / 2, 10)
    strel_dirac, time_dirac = cqt(dirac, numpy=False)

    strel_dirac_norm = strel_dirac
    for i in range(strel_dirac.shape[0]):
        strel_dirac_norm[i, :] = strel_dirac_norm[i, :] - torch.max(strel_dirac_norm[i, :])

    plot_cqt(strel_dirac_norm, time_dirac, show=False, numpy=False)

    width_erosion = partial_erosion(spectrogram, strel_dirac_norm, tuple([strel_dirac_norm.shape[1] // 2]))

    plot_cqt(width_erosion, time_vector, numpy=False)

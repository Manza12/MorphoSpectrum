from datetime import time

from midi import midi2piece
from play import play_signal
from plots import plot_cqt
from samples import get_samples_set
from time_frequency import cqt
from signals import *
from scipy.signal.windows import get_window
from nnMorpho.operations import partial_erosion
import time

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


def loss_exp(predicted, target, sigma=0.2):
    return torch.mean(torch.exp(sigma * torch.absolute(predicted - target)))


if __name__ == '__main__':
    configure_logs('learning')

    plot = False

    # Create the signal
    sta = time.time()
    samples_set = get_samples_set('basic', decay=2)
    end = time.time()
    log.info("Time to create samples set: " + str(round(end - sta, 3)) + " seconds.")

    sta = time.time()
    piece = midi2piece('tempest_3rd-start')
    signal = samples_set.synthesize(piece)  # signal_from_file('tempest_3rd-start')
    end = time.time()
    log.info("Time to synthesize the signal: " + str(round(end - sta, 3)) + " seconds.")

    # Time-frequency transform of the signal
    sta = time.time()
    spectrogram, time_vector = cqt(signal, numpy=False)
    end = time.time()
    log.info("Time to compute the CQT of the signal: " + str(round(end - sta, 3)) + " seconds.")

    # Create MIDI tensor
    midi_tensor = torch.zeros_like(spectrogram, device=DEVICE)

    for note in piece:
        onset = int(note.start_seconds / TIME_RESOLUTION)
        velocity = note.velocity
        velocity_decibel = (127 - velocity) * NOISE_THRESHOLD / 127
        frequency_bin = int(np.log2(note.pitch.frequency / F_MIN) * BINS_PER_OCTAVE)

        midi_tensor[frequency_bin, onset] = velocity_decibel

    if plot:
        plot_cqt(midi_tensor, time_vector, show=False, numpy=False, v_min=NOISE_THRESHOLD, v_max=0)

    if play:
        play_signal(signal)

    if plot:
        plot_cqt(spectrogram, time_vector, show=False, numpy=False)

    # Morphological transform of the signal
    dirac_duration = 0.5
    dirac = generate_dirac(dirac_duration, dirac_duration / 2, 10)
    strel_dirac, time_dirac = cqt(dirac, numpy=False)

    strel_dirac_norm = strel_dirac
    for i in range(strel_dirac.shape[0]):
        strel_dirac_norm[i, :] = strel_dirac_norm[i, :] - torch.max(strel_dirac_norm[i, :])

    if plot:
        plot_cqt(strel_dirac_norm, time_dirac, show=False, numpy=False)

    width_erosion = partial_erosion(spectrogram, strel_dirac_norm, tuple([strel_dirac_norm.shape[1] // 2]))

    if plot:
        plot_cqt(width_erosion, time_vector, numpy=False)

    from nnMorpho.modules import Erosion
    model = Erosion((200, 50), (20, 10), 1e20)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # Learning loop
    iterations = 30000
    sta = time.time()
    for t in range(iterations):
        # Forward pass: Compute predicted y by passing x to the model
        erosion_predicted = model(width_erosion)

        # Loss
        loss = loss_exp(erosion_predicted, midi_tensor)
        print("Step", t, "Loss:", loss.item())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end = time.time()
    print("Time to compute", iterations, "iterations:", round(end-sta), "seconds.")

    learned_structural_element = model.structural_element
    plt.figure()
    plt.imshow(learned_structural_element.detach().cpu().numpy(), cmap='hot')
    plt.show()

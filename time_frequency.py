import nnAudio.Spectrogram as Spec
from parameters import *
from plots import plot_cqt
from signals import signal_from_file, get_time_vector

cqt_layer = Spec.CQT(sr=FS, hop_length=HOP_LENGTH, fmin=F_MIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
                     norm=NORM, pad_mode='constant', window=WINDOW)
cqt_layer.to(DEVICE)


def cqt(signal, numpy=True):
    signal_tensor = torch.tensor(signal, device=DEVICE, dtype=torch.float)
    cqt_tensor = cqt_layer(signal_tensor)

    if numpy:
        cqt_array = cqt_tensor.cpu().numpy()[0, :, :]
        torch.cuda.empty_cache()
        return cqt_array
    else:
        return cqt_tensor


if __name__ == '__main__':
    _signal = signal_from_file('A0_22.086_98', SAMPLES_PATH)
    _time_vector = get_time_vector(_signal)
    _spectrogram = cqt(_signal, numpy=True)
    _spectrogram_log = 20 * np.log10(_spectrogram + EPS)
    plot_cqt(_spectrogram_log, _time_vector)

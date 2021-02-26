# import nnAudio.Spectrogram as Spec
import Gon.Spectrogram as Spec
from plots import plot_cqt
from parameters import *

if USE_CQT:
    cqt_layer = Spec.CQT(sr=FS, hop_length=HOP_LENGTH, fmin=F_MIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
                         norm=NORM, pad_mode='constant', window=WINDOW)
    cqt_layer.to(DEVICE)


def cqt(signal, numpy=True, db=True):
    time_array = np.arange(np.ceil(signal.size / HOP_LENGTH).astype(int)) / (FS / HOP_LENGTH)

    signal_tensor = torch.tensor(signal, device=DEVICE, dtype=torch.float)
    cqt_tensor = cqt_layer(signal_tensor)

    if db:
        cqt_tensor = 20 * torch.log10(cqt_tensor + EPS)

    if numpy:
        cqt_array = cqt_tensor.cpu().numpy()[0, :, :]
        torch.cuda.empty_cache()
        return cqt_array, time_array
    else:
        time_tensor = torch.tensor(time_array)
        return cqt_tensor, time_tensor


if __name__ == '__main__':
    _signal = np.zeros(FS*2)
    dirac_width = 1

    for i in range(dirac_width):
        _signal[i::int(FS*0.033)] = 1

    _signal += 0.1 * np.sin(2*np.pi*440*np.arange(FS*2) / FS)

    _spectrogram, _time_array = cqt(_signal)
    plot_cqt(_spectrogram, _time_array, v_min=-100)

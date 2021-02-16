import nnAudio.Spectrogram as Spec
from parameters import *

if not LOAD_ALL:
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
    pass

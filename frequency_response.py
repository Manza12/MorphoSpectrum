from parameters import *
import scipy.signal.windows as win
import matplotlib.pyplot as plt
from utils import to_db, gaussian_db
import scipy.signal as sig


def frequency_response(omega, window, normalize=True, db=True):
    if type(omega) == float:
        omega = np.array([omega])
    n = np.arange(len(window))
    a = np.sum(window * np.exp(-2 * np.pi * 1j * np.expand_dims(omega, 1) * np.expand_dims(n, 0)), 1)
    if normalize:
        a = a / np.sum(window)
    if db:
        a = to_db(np.abs(a))

    return a


if __name__ == '__main__':
    # Windows
    N = 32 * 2
    db_attenuation = 60
    h = win.hann(N, sym=False)
    g = gaussian_db(N, sym=False, db_attenuation=db_attenuation)

    # Frequency responses
    F = N * N + 1
    f = np.linspace(-0.5, 0.5, F)

    H = frequency_response(f, h)
    G = frequency_response(f, g)

    print(1 / N, frequency_response(1 / N, h))
    print(1 / N, frequency_response(1 / N, g))

    # Lobe info 1
    pk_idx_h, _ = sig.find_peaks(H)
    pk_idx_g, _ = sig.find_peaks(G)

    idx_sort_h = sorted(range(len(H[pk_idx_h])), key=lambda k: H[pk_idx_h][k])
    f_h_1 = f[pk_idx_h[idx_sort_h[-2]]]
    f_h_2 = f[pk_idx_h[idx_sort_h[-3]]]
    print('Lobe width of H:', np.abs(f_h_2-f_h_1))

    idx_sort_g = sorted(range(len(G[pk_idx_g])), key=lambda k: G[pk_idx_g][k])
    f_g_1 = f[pk_idx_g[idx_sort_g[-2]]]
    f_g_2 = f[pk_idx_g[idx_sort_g[-3]]]
    print('Lobe width of G:', np.abs(f_g_2 - f_g_1))

    lobe_height_h = H[pk_idx_h[idx_sort_h[-2]]]
    lobe_height_g = G[pk_idx_g[idx_sort_g[-2]]]

    print('Lobe attenuation of H:', lobe_height_h)
    print('Lobe attenuation of G:', lobe_height_g)

    # Lobe info 2
    db_att = -6
    lobe_width_h = np.sum(H >= db_att) / F
    lobe_width_g = np.sum(G >= db_att) / F

    print('Lobe width -6 dB of H:', lobe_width_h)
    print('Lobe width -6 dB of G:', lobe_width_g)

    # Plots
    plt.figure()
    plt.plot(f, H)
    plt.scatter(f[pk_idx_h], H[pk_idx_h], marker='x', c='r')
    plt.hlines(lobe_height_h, xmin=-0.5, xmax=0.5, colors='k')
    plt.xlim((-0.5, 0.5))
    plt.ylim((-100, 5))
    plt.title('Frequency response of a Hann window of length %r' % N)

    plt.figure()
    plt.plot(f, G)
    plt.scatter(f[pk_idx_g], G[pk_idx_g], marker='x', c='r')
    plt.hlines(lobe_height_g, xmin=-0.5, xmax=0.5, colors='k')
    plt.xlim((-0.5, 0.5))
    plt.ylim((-100, 5))
    plt.title('Frequency response of a Gaussian window of length %r \n and attenuation %r dB' % (N, db_attenuation))

    plt.show()

from datetime import time as tm
import math
import matplotlib.ticker as tick
from parameters import *


def format_freq(x, pos, f):
    if pos:
        pass
    n = int(round(x))
    if 0 <= n < f.size:
        if PLOT_UNITS:
            return str(f[n].astype(int)) + " Hz"
        else:
            return str(f[n].astype(int))
    else:
        return ""


def format_time(x, pos, t):
    if pos:
        pass
    n = int(round(x))
    if 0 <= n < t.size:
        if PLOT_UNITS:
            return str(round(t[n], 3)) + " s"
        else:
            decomposition = math.modf(round(t[n], 6))
            s = round(decomposition[1])
            hours = s // (60 * 60)
            minutes = (s - hours * 60 * 60) // 60
            seconds = s - (hours*60*60) - (minutes*60)
            return tm(second=seconds, minute=minutes, hour=hours, microsecond=round(decomposition[0] * 1e6)).isoformat(timespec='milliseconds')[3:]
    else:
        return ""


def plot_cqt(a, t, f=FREQUENCIES, fig_title=None, v_min=V_MIN, v_max=V_MAX, c_map=C_MAP, show=True):
    fig = plt.figure(figsize=(2*320/DPI, 2*240/DPI), dpi=DPI)

    if fig_title:
        fig.suptitle(fig_title)

    ax = fig.add_subplot(111)

    ax.imshow(a, cmap=c_map, aspect='auto', vmin=v_min, vmax=v_max, origin='lower')

    # Freq axis
    ax.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: format_freq(x, pos, f)))

    # Time axis
    ax.xaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: format_time(x, pos, t)))

    # Labels
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel('Frequency (Hz)')

    if FULL_SCREEN:
        manager = plt.get_current_fig_manager()
        if BACKEND == 'WXAgg':
            manager.frame.Maximize(True)
        elif BACKEND == 'TkAgg':
            manager.resize(*manager.window.maxsize())
        else:
            raise Exception("Backend not supported.")

    if show:
        plt.show()

    return fig


def plot_morphology(a_dilation, t, f=FREQUENCIES):
    # a_dilation_log = 20 * np.log10(a_dilation + EPS)
    fig = plt.figure(figsize=(2*320/DPI, 2*240/DPI), dpi=DPI)
    ax = fig.add_subplot(111)

    plt.imshow(a_dilation, cmap='Greys', aspect='auto', origin='lower', vmin=V_MIN_MOR, vmax=V_MAX_MOR)

    # Freq axis
    ax.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: format_freq(x, pos, f)))

    # Time axis
    ax.xaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: format_time(x, pos, t)))

    if FULL_SCREEN:
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())

    plt.xlabel('Time (mm:ss)')
    plt.ylabel('Frequency (Hz)')


def plot_both(a, a_dilation, t, f=FREQUENCIES):
    # a_log = 20 * np.log10(a + EPS)
    # a_dilation_log = 20 * np.log10(a_dilation + EPS)

    fig = plt.figure(figsize=(2*320/DPI, 2*240/DPI), dpi=DPI)
    ax_1 = fig.add_subplot(211)
    ax_2 = fig.add_subplot(212, sharex=ax_1, sharey=ax_1)

    ax_1.imshow(a, cmap='hot', aspect='auto', vmin=V_MIN, vmax=V_MAX, origin='lower')
    ax_2.imshow(a_dilation, cmap='Greys', aspect='auto', origin='lower', vmin=V_MIN_MOR, vmax=V_MAX_MOR)

    # Freq axis
    ax_1.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: format_freq(x, pos, f)))

    # Time axis
    ax_1.xaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: format_time(x, pos, t)))

    if FULL_SCREEN:
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())

    # Labels
    ax_1.set_xlabel('Time (mm:ss)')
    ax_1.set_ylabel('Frequency (Hz)')
    ax_2.set_xlabel('Time (mm:ss)')
    ax_2.set_ylabel('Frequency (Hz)')

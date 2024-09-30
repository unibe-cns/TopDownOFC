import pathlib
import matplotlib
import numpy as np
from matplotlib.figure import Figure
from ofcsst.utils import keys, paths, sql

# Color definitions
COLOR_REWARD = (18. / 255., 157. / 255., 73. / 255.)
COLOR_PUNISHMENT = (211. / 255., 31. / 255., 63. / 255.)
COLOR_CR = (0.6, 0.6, 0.6)
COLOR_NO_GAIN = (0.5, 0.5, 0.5)
COLOR_GAIN = (237. / 255., 196. / 255., 0.)
COLOR_DEFAULT = (160. / 255., 160. / 255., 160. / 255.)
COLOR_SST = (113. / 255., 179. / 255., 121. / 255.)
COLOR_FAKE_OFC = (195. / 255., 70. / 255., 108. / 255.)
COLOR_CONTEXT = (200. / 255., 164. / 255., 140. / 255.)
COLOR_VIP = (178. / 255., 86. / 255., 144. / 255.)
COLOR_LN = (166. / 255., 174 / 255., 195. / 255.)
COLOR_LE = (33. / 255., 35 / 255., 75. / 255.)
COLOR_RN = (213. / 255., 184 / 255., 188. / 255.)
COLOR_RE = (198. / 255., 38 / 255., 50. / 255.)
COLOR_BASELINE = (220. / 255., 220. / 255., 220. / 255.)
COLOR_STIMULUS = (173. / 255., 196. / 255., 217. / 255.)
COLOR_OUTCOME = (217. / 255., 173. / 255., 199. / 255.)
COLOR_CNO = (0.8, 0.8, 0.8)

# Figure and panel sizes
INCH = 2.54
DPI = 600
FIG_WPAD = 0.1  # inches
FIG_HPAD = 0.06  # inches
PANEL_LABEL_HEIGHT = 0.11110236
AXIS_WIDTH = 0.5
LINE_WIDTH = 0.7
MARKER_SIZE = 0.7
FIGURE_WIDTH = 17.8 / INCH
PANEL_WIDTH = FIGURE_WIDTH / 3.
PANEL_HEIGHT = 4.5 / INCH

# Annotations
FONT_SIZE = 6
FONT_SIGNIFICANT = {'fontsize': 7, 'weight': 'bold', 'ha': 'center', 'va': 'center'}
FONT_NON_SIGNIFICANT = {'s': "n.s.", 'fontsize': 5, 'ha': 'center'}
FONT_RULE_SWITCH = {'s': 'Rule-switch', 'color': (188. / 255., 34. / 255., 47. / 255.), 'style': 'italic',
                    'fontsize': 5., 'ha': 'center', 'va': 'center'}
STYLE_SIGNIFICANT = {'color': 'k', 'linewidth': AXIS_WIDTH}
STYLE_RULE_SWITCH = {'color': (0, 0, 0.5), 'linestyle': (0, (4, 4)), 'linewidth': AXIS_WIDTH}
STYLE_EXPERT_PERF = {'linestyle': (0, (3, 1, 1, 1)), 'color': (1, 0, 0), 'alpha': 0.3, 'linewidth': AXIS_WIDTH}


def set_style():
    """Default figure style"""
    matplotlib.rc('font', family='sans-serif')
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['text.usetex'] = 'false'
    matplotlib.rcParams['svg.fonttype'] = 'path'
    matplotlib.rcParams['font.size'] = FONT_SIZE
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['xtick.major.size'] = 2.
    matplotlib.rcParams['xtick.major.width'] = 0.5
    matplotlib.rcParams['xtick.minor.size'] = 1.2
    matplotlib.rcParams['xtick.minor.width'] = 0.35
    matplotlib.rcParams['ytick.major.size'] = matplotlib.rcParams['xtick.major.size']
    matplotlib.rcParams['ytick.major.width'] = matplotlib.rcParams['xtick.major.width']
    matplotlib.rcParams['ytick.minor.size'] = matplotlib.rcParams['xtick.minor.size']
    matplotlib.rcParams['ytick.minor.width'] = matplotlib.rcParams['xtick.minor.width']
    matplotlib.rcParams['axes.linewidth'] = AXIS_WIDTH
    matplotlib.rcParams['lines.linewidth'] = LINE_WIDTH
    matplotlib.rcParams['patch.linewidth'] = LINE_WIDTH
    matplotlib.rcParams['hatch.linewidth'] = 0.3


def adjust_figure(fig: Figure, wspace: float = None, hspace: float = None) -> None:
    """Adjust figure size and padding"""
    fig_width, fig_height = fig.get_size_inches()
    bottom = FIG_HPAD / fig_height
    left = FIG_WPAD / fig_width
    rect = (left, bottom, 1. - left, 1. - bottom - PANEL_LABEL_HEIGHT / fig_height)
    matplotlib.pyplot.tight_layout(pad=0., h_pad=0, w_pad=0, rect=rect)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)


def set_panel_label(label: str, fig: Figure):
    """Annotate panel label"""
    _, fig_height = fig.get_size_inches()
    fig.gca().text(FIG_WPAD, fig_height - FIG_HPAD - PANEL_LABEL_HEIGHT, label, transform=fig.dpi_scale_trans,
                   fontsize=8, fontweight='bold')


def plot_frame(fig: Figure):
    """Display ideal padding frame"""
    fig_width, fig_height = fig.get_size_inches()
    rect = matplotlib.pyplot.Rectangle((FIG_WPAD, FIG_HPAD), fig_width - 2 * FIG_WPAD, fig_height - 2 * FIG_HPAD,
                                       transform=fig.dpi_scale_trans, alpha=1, facecolor='none',
                                       clip_on=False, edgecolor="k", linewidth=0.2, linestyle="-.")
    fig.gca().add_patch(rect)


def save_or_show(save: bool, fig: Figure, plot_dir: pathlib.Path, plot_name: str, plot_dpi=750, png: bool = False,
                 transparent_background: bool = True):
    """Display figure or save to svg and pdf"""
    if save:
        (plot_dir / paths.SVG).mkdir(parents=True, exist_ok=True)
        matplotlib.pyplot.savefig(plot_dir / paths.SVG / f"{plot_name}.svg")
        matplotlib.pyplot.savefig(plot_dir / f"{plot_name}.pdf", dpi=DPI, transparent=transparent_background)
        if png:
            matplotlib.pyplot.savefig(plot_dir / f"{plot_name}.pdf", dpi=DPI)
        matplotlib.pyplot.close()
    else:
        plot_frame(fig=fig)
        matplotlib.pyplot.gcf().set_dpi(plot_dpi)
        matplotlib.pyplot.show()


def key_name(key: keys.Key) -> str:
    """Return the full name from the key of a variable."""
    if key == keys.DISTRACTOR_GAIN:
        return "Gain of distractor neurons"
    elif key == keys.SIGNAL_GAIN:
        return "Gain of go/no-go stimulus neurons"
    elif key in [keys.NR_DISTRACTOR]:
        return r"Number of distractors $\mathregular{N_d}$"
    elif key in [keys.NR_SIGNAL]:
        return "Number of signals"
    elif key in [keys.SNR]:
        return "Signal-to-noise ratio"
    elif key in [keys.LEARNING_RATE_PG]:
        return r"Learning Rate $\mathregular{\pi_{PG}}$"
    elif key in [keys.LEARNING_RATE_V]:
        return "TD Learning Rate"
    else:
        if keys.LOSS in key:
            return "Loss"
        elif keys.PERFORMANCE in key:
            return "Performance"
        else:
            raise ValueError(key)


def get_data_by_keys(path: pathlib.Path, table: str, key_list: []):
    # Prepare SNR computation
    get_cols = []
    snr_idx = None
    for idx, k in enumerate(key_list):
        if k in [keys.SNR]:
            get_cols += [keys.DISTRACTOR_GAIN, keys.SIGNAL_GAIN, keys.NR_DISTRACTOR]
            snr_idx = idx
        else:
            get_cols += [k]

    # Get data
    n_axes = len(key_list)
    rows = sql.select(db_path=path, table=table, get_cols=get_cols)
    data_matrix = np.zeros((n_axes, len(rows)))
    idx = 0
    for di in range(n_axes):
        if idx != snr_idx:
            data_matrix[di, :] = [row[idx] for row in rows]
            idx += 1
        else:
            data_matrix[di, :] = [(row[idx + 1] / row[idx]) ** 2 / row[idx + 2] for row in rows]
            idx += 3

    return data_matrix


def get_performance_cmap(phase: str = 'both'):
    """Return the color map for the desired learning phases. Can be either 'learning', 'reversal' or 'both."""
    l_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", [(171. / 255., 195. / 255., 212. / 255.), (35. / 255., 31. / 255., 84. / 255.)])
    r_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", [(251. / 255., 216. / 255., 216. / 255.), (173. / 255., 31. / 255., 42. / 255.)])

    if phase == 'learning':
        return l_cmap
    if phase == 'reversal':
        return r_cmap
    if phase == 'both':
        both_cmaps = np.vstack((l_cmap(np.linspace(0, 1, 128)),
                                r_cmap(np.linspace(0, 1, 128))))
        return matplotlib.colors.ListedColormap(both_cmaps, name='')
    else:
        raise ValueError(f'phase "{phase}" is neither of: learning, reversal, both')


def get_significance(p_value: float) -> str:
    """
    Get significance stars for a p-value.
    :param p_value: P-value.
    """
    if p_value > 0.05 or np.isnan(p_value):
        s = 'n.s.'
    elif p_value > 0.01:
        s = '*'
    elif p_value > 0.001:
        s = '**'
    else:
        s = '***'
    return s

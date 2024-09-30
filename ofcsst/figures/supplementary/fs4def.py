from ofcsst.simulation.scan import find_best_params
from ofcsst.utils import ids, paths
from ofcsst.figures.main.f1e_explore_distractor import plot_performance_2d
from ofcsst.figures.main.f1f_explore_gain import plot_gain_dependence
from ofcsst.figures.main.f1h_gain_vs_no_gain import plot_gain_vs_no_gain

PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs4"


def panel_d_explore_distractor_convex(save: bool = False) -> None:
    plot_performance_2d(simulation_type=ids.SIM_CONVEX_NO_GAIN, panel='d', plot_dir=PLOT_DIR, save=save)


def panel_e_explore_snr_convex(save: bool = False) -> None:
    plot_gain_dependence(simulation_type=ids.SIM_CONVEX_NO_GAIN, plot_dir=PLOT_DIR, panel='e', save=save)


def panel_f_gain_vs_no_gain(save: bool = False) -> None:
    plot_gain_vs_no_gain(simulation_types=(ids.SIM_CONVEX_NO_GAIN, ids.SIM_CONVEX_GAIN), plot_dir=PLOT_DIR, panel='f',
                         save=save)


def scan():
    find_best_params(simulation_type=ids.SIM_CONVEX_NO_GAIN, task_id=ids.BINARY_2VN, switch_task=False,
                     scan_type=ids.ST_VARY_DISTRACTION_2D)
    find_best_params(simulation_type=ids.SIM_CONVEX_GAIN, task_id=ids.BINARY_2VN, switch_task=False,
                     scan_type=ids.ST_VARY_DISTRACTOR_1D)


def plot(save: bool = True):
    panel_d_explore_distractor_convex(save=save)
    panel_e_explore_snr_convex(save=save)
    panel_f_gain_vs_no_gain(save=save)

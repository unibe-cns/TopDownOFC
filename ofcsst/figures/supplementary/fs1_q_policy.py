from ofcsst.simulation.scan import find_best_params
from ofcsst.utils import constants, ids, paths
from ofcsst.figures import helper
from ofcsst.figures.main.f1e_explore_distractor import plot_performance_2d
from ofcsst.figures.main.f1f_explore_gain import plot_gain_dependence
from ofcsst.figures.main.f1h_gain_vs_no_gain import plot_gain_vs_no_gain

PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs1"
SIM_TYPE = ids.SIM_SQ_NO_GAIN
SIM_TYPE_GAIN = ids.SIM_SQ_GAIN


def scan() -> None:
    """
    Run simulations with pure policy gradient without gain modulation with a variety
    """
    find_best_params(
        task_id=constants.TASK_ID,
        scan_type=ids.ST_VARY_DISTRACTION_2D,
        simulation_type=SIM_TYPE,
        switch_task=False,
    )
    find_best_params(
        simulation_type=SIM_TYPE_GAIN,
        task_id=constants.TASK_ID,
        switch_task=False,
        scan_type=ids.ST_VARY_DISTRACTOR_1D,
    )
    find_best_params(
        simulation_type=ids.SIM_PURE_SQ_GAIN,
        task_id=constants.TASK_ID,
        switch_task=False,
        scan_type=ids.ST_VARY_DISTRACTOR_1D,
    )


def plot(save: bool = True) -> None:
    plot_performance_2d(
        simulation_type=SIM_TYPE, panel="a", plot_dir=PLOT_DIR, save=save
    )
    plot_gain_dependence(
        simulation_type=SIM_TYPE, plot_dir=PLOT_DIR, panel="b", save=save
    )
    sts = (ids.SIM_SQ_NO_GAIN, ids.SIM_PURE_SQ_GAIN, ids.SIM_SQ_GAIN)
    lbs = ["Fixed gains", "Backprop. gains (Q loss)", "Dynamic gains (ours)"]
    cls = (helper.COLOR_NO_GAIN, helper.COLOR_BACKPROP_GAIN, helper.COLOR_GAIN)
    plot_gain_vs_no_gain(
        simulation_types=sts,
        plot_dir=PLOT_DIR,
        panel="c",
        save=save,
        st_labels=lbs,
        st_colors=cls,
    )

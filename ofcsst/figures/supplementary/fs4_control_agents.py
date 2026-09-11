from ofcsst.simulation.scan import find_best_params
from ofcsst.figures import helper
from ofcsst.figures.main.f1h_gain_vs_no_gain import plot_gain_vs_no_gain
from ofcsst.utils import constants, ids, paths


PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs4"


def scan():
    # Bottom-up plasticity without gain
    find_best_params(
        simulation_type=ids.SIM_PG_NO_GAIN_DEEP,
        task_id=constants.TASK_ID,
        switch_task=False,
        scan_type=ids.ST_VARY_DISTRACTOR_1D,
    )

    # Rescorla-Wagner gain
    find_best_params(
        simulation_type=ids.SIM_PG_RW,
        task_id=constants.TASK_ID,
        switch_task=False,
        scan_type=ids.ST_VARY_DISTRACTOR_1D,
    )

    # Pure policy gradient with backpropagated gain
    find_best_params(
        simulation_type=ids.SIM_PURE_PG_GAIN,
        task_id=constants.TASK_ID,
        switch_task=False,
        scan_type=ids.ST_VARY_DISTRACTOR_1D,
    )


def plot(save: bool = True) -> None:
    sts = (
        ids.SIM_PG_NO_GAIN,
        ids.SIM_PG_RW,
        ids.SIM_PURE_PG_GAIN,
        ids.SIM_PG_NO_GAIN_DEEP,
        ids.SIM_PG_GAIN,
    )
    lbs = [
        "Fixed gains",
        "Bottom-up plasticity",
        "Policy gradient gains",
        "Rescorla-Wagner gains",
        "Dynamic gains (ours)",
    ]
    cls = (
        helper.COLOR_NO_GAIN,
        helper.COLOR_BU_PLASTICITY,
        helper.COLOR_BACKPROP_GAIN,
        helper.COLOR_RESCORLA_WAGNER,
        helper.COLOR_GAIN,
    )
    plot_gain_vs_no_gain(
        simulation_types=sts, plot_dir=PLOT_DIR, save=save, st_labels=lbs, st_colors=cls
    )

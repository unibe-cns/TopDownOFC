from ofcsst import simulation
from ofcsst.utils import ids, keys, paths, constants
from ofcsst.figures import helper
from ofcsst.figures.main.f3efg_ofc_sst import (
    OFC_SIM,
    NO_OFC_SIM,
    final_simulation,
    plot_expert_time,
    FigureConfig,
)
from ofcsst.figures.main.f3efg_ofc_sst import (
    plot_performance_bars,
    plot_cumulative_distribution,
)
from ofcsst.figures.main.f3efg_ofc_sst import COLOR_NO_OFC, COLOR_WT, SAVE_PATH

TASK_ID = ids.BINARY_2VN
PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs8"
DB_PATH = paths.SUPPLEMENTARY_SIMULATION_DIR / "fig_s8_optimization.db"
COLORS = [(114 / 255.0, 146 / 255.0, 179 / 255.0), helper.COLOR_GAIN]


def simulate_condition(simulation_type: ids.SimulationType):
    # Initialize variables
    if simulation_type == NO_OFC_SIM:
        scan_type = ids.ST_FINAL
        non_stat = False
        seeds = list(range(constants.NR_SIM_MICE, 2 * constants.NR_SIM_MICE))

    elif simulation_type == OFC_SIM:
        scan_type = ids.ST_ALL
        non_stat = True
        seeds = list(range(2 * constants.NR_SIM_MICE, 3 * constants.NR_SIM_MICE))

    else:
        raise NotImplementedError(simulation_type)
    scan_db_path = simulation.scan.get_db_path(
        simulation_type=simulation_type, scan_type=scan_type, non_stationary=non_stat
    )

    # Run simulations
    final_simulation(
        sim_type=simulation_type,
        scan_path=scan_db_path,
        seeds=seeds,
        save_db_path=DB_PATH,
    )


def panel_a_performance_bars(save: bool = True, verbose: bool = False) -> None:
    plot_performance_bars(
        save=save,
        db_path=DB_PATH,
        panel="a",
        plot_name="a_performance_bars",
        plot_dir=PLOT_DIR,
        colors=[COLORS[1], COLORS[0]],
        labels=["R-Opt\n+lOFC", "L-Opt\n-lOFC"],
        verbose=verbose,
    )


def panel_b_cumulative_distribution(save: bool = True) -> None:
    plot_cumulative_distribution(
        save=save,
        db_path=DB_PATH,
        panel="b",
        phase_cosmetics=False,
        plot_name="b_expertise_cumulative_distribution",
        plot_dir=PLOT_DIR,
        colors=COLORS,
        labels=["R-Opt (+lOFC)", "L-Opt (-lOFC)"],
    )


def panel_c_expert_time(save: bool = True) -> None:
    pw = 1.7 * helper.PANEL_WIDTH
    fig_config = FigureConfig(
        figure_key=keys.FIGURE_S8,
        panel="c",
        plot_name="c_expert_time",
        plot_dir=PLOT_DIR,
        panel_width=pw,
        height_ratios=(3.0, 1.0),
        width_ratios=(1.0, 2.0),
        height_scale=1.0,
        sim_types=(OFC_SIM, NO_OFC_SIM),
        db_paths=[SAVE_PATH, DB_PATH],
        colors=[COLOR_WT, COLOR_NO_OFC] + COLORS,
        labels=[
            "R-Opt mixed (intact lOFC)",
            "R-Opt mixed (silenced lOFC)",
            "R-Opt with lOFC",
            "L-Opt without lOFC",
        ],
        positions=[[1.0, 2.0], [3.0, 4.0]],
        trace_seed_loop=True,
        reversal_scatter_threshold=constants.NR_SIM_MICE / 2,
        reversal_scatter_sim_types=None,
        shade_rect_clip_on=False,
        add_fake_ofc_patch=False,
        fake_ofc_patch_target=None,
        fake_ofc_patch_color_index=None,
        rule_switch_text=(constants.NR_TRIALS - 30 * helper.PANEL_WIDTH / pw, 55),
        rule_switch_line=(0.0, -11.5, 0.0, 4.5),
        legend_axis=(0, 1),
        legend_kwargs=dict(
            loc="lower center",
            frameon=False,
            borderpad=0,
            handletextpad=0.5,
            handlelength=1,
            ncols=2,
            columnspacing=1.5,
        ),
        performance_ylabel=True,
        expertise_ylabel_kwargs=dict(rotation=0, y=0.25, labelpad=15),
        supxlabel_kwargs=dict(fontsize=helper.FONT_SIZE, x=0.55, y=0.03),
        adjust_figure_kwargs=dict(hspace=0, wspace=0),
        subplots_adjust_kwargs=dict(bottom=0.2, right=0.975),
        left_y_ticks=(
            [0.0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.0],
            ["", "10", "", "30", "", "50", "", "70", "", "90", ""],
        ),
        bottom_xticks={
            (1, 0): ([0, 100, 200, 300, 400, 500], ["", "100", "", "", "400", ""]),
            (1, 1): (
                [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
                ["", "100", "", "", "400", "", "", "700", "", "", "1000", "", ""],
            ),
        },
        top_right_hide_y_ticks=True,
        y_lims_fixed=(0.0, 100.0),
        box_y_limits=(0.5, 4.5),
    )
    plot_expert_time(save=save, fig_config=fig_config)


def scan():
    simulation.scan.find_best_params(
        simulation_type=NO_OFC_SIM,
        task_id=TASK_ID,
        switch_task=False,
        scan_type=ids.ST_FINAL,
    )
    simulation.scan.find_best_params(
        simulation_type=OFC_SIM, task_id=TASK_ID, switch_task=True, scan_type=ids.ST_ALL
    )


def run():
    print("Running simulations for Figure S8")
    simulate_condition(simulation_type=OFC_SIM)
    simulate_condition(simulation_type=NO_OFC_SIM)


def plot(save: bool = True):
    panel_a_performance_bars(save=save)
    panel_b_cumulative_distribution(save=save)
    panel_c_expert_time(save=save)

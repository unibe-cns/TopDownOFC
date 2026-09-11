import sqlite3
import numpy as np
from ofcsst import simulation
from ofcsst.utils import ids, keys, paths, constants, sql
from ofcsst.figures import helper
from ofcsst.figures.main.f3efg_ofc_sst import FigureConfig, plot_expert_time

TASK_ID = ids.BINARY_2VN
PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs7"
SAVE_PATH = paths.SUPPLEMENTARY_SIMULATION_DIR / "fig_s7_fake_ofc.db"
NO_OFC_SIM = ids.SIM_CONVEX_GAIN
REV_RESET_SIM = ids.SIM_CONVEX_GAIN_REV_RESET
FAKE_OFC_SIM = ids.SIM_CONVEX_FAKE_OFC
SIM_NAMES = {
    NO_OFC_SIM: "Default",
    REV_RESET_SIM: "Reset values",
    FAKE_OFC_SIM: "Fake SST",
}
TRIAL_NRS = (constants.NR_TRIALS, constants.NR_TRIALS)


def simulate_condition(sim_type: ids.SimulationType):
    # Initialize database variables
    if sim_type == NO_OFC_SIM:
        seeds = list(range(3 * constants.NR_SIM_MICE, 4 * constants.NR_SIM_MICE))
    elif sim_type == REV_RESET_SIM:
        seeds = list(range(4 * constants.NR_SIM_MICE, 5 * constants.NR_SIM_MICE))
    elif sim_type == FAKE_OFC_SIM:
        seeds = list(range(5 * constants.NR_SIM_MICE, 6 * constants.NR_SIM_MICE))
    else:
        raise NotImplementedError(sim_type)

    # Initializations to save final simulation outcomes
    save_keys = [str(keys.SEED)] + [
        f"{keys.PERFORMANCE}_{t}" for t in range(sum(TRIAL_NRS))
    ]
    conn = sql.connect(db_path=SAVE_PATH)
    sql.drop_table(conn=conn, table_name=sim_type, verbose=False)
    sql.make_table(conn=conn, table_name=sim_type, col_names=save_keys, verbose=False)
    conn.close()
    insert_cmd = sql.get_insert_cmd(table=sim_type, col_keys=save_keys)

    scan_db_path = simulation.scan.get_db_path(
        scan_type=ids.ST_FINAL,
        simulation_type=NO_OFC_SIM,
        task_id=TASK_ID,
        non_stationary=False,
    )

    select_cols = simulation.databases.get_unique_cols(
        simulation_type=sim_type, table=simulation.databases.SCAN_TABLE
    )
    params = sql.get_max(
        db_path=scan_db_path,
        table=simulation.databases.BEST_PARAM_TABLE,
        group_cols=[],
        select_cols=select_cols,
        max_col=keys.PERFORMANCE,
        maxmin=True,
    )[0]

    # Run final simulations and save performance for each task
    outcomes = np.zeros((constants.NR_SIM_MICE, sum(TRIAL_NRS)))
    for s in range(constants.NR_SIM_MICE):
        print(f"\rSimulating seed {s}/{constants.NR_SIM_MICE}...", end="")
        outcomes[s, :] = simulation.simulate.simulate_params(
            task_id=TASK_ID,
            simulation_type=sim_type,
            seed=seeds[s],
            params=list(params),
            nr_contexts=2,
            n_trials=TRIAL_NRS,
        )
    print(f"\rSimulation of {sim_type} completed.")

    conn = sqlite3.connect(SAVE_PATH)
    cur = conn.cursor()
    for s in range(constants.NR_SIM_MICE):
        values = tuple([seeds[s]] + outcomes[s, :].tolist())
        cur.execute(insert_cmd, values)
    conn.commit()
    cur.close()
    conn.close()


def plot(save: bool = True) -> None:
    fig_config = FigureConfig(
        figure_key=keys.FIGURE_S7,
        panel="b",
        plot_name="b_perf_traces",
        plot_dir=PLOT_DIR,
        trial_nrs=TRIAL_NRS,
        panel_width=2 * helper.PANEL_WIDTH,
        height_ratios=(4.0, 1.0),
        width_ratios=(1.0, 1.0),
        height_scale=1.0,
        sim_types=(NO_OFC_SIM, REV_RESET_SIM, FAKE_OFC_SIM),
        sim_names=SIM_NAMES,
        db_paths=[SAVE_PATH],
        colors=[
            helper.COLOR_GAIN,
            helper.COLOR_FAKE_OFC,
            (89.0 / 255.0, 41.0 / 255.0, 105.0 / 255.0),
        ],
        labels=None,
        positions=[[1.0, 2.0, 3.0]],
        trace_seed_loop=True,
        reversal_scatter_threshold=None,
        reversal_scatter_sim_types={NO_OFC_SIM},
        shade_rect_clip_on=True,
        add_fake_ofc_patch=True,
        fake_ofc_patch_target=(0, 0),
        fake_ofc_patch_color_index=2,
        rule_switch_text=(constants.NR_TRIALS - 12, 25),
        rule_switch_line=(0.0, -3.875 * 3.0, 0.0, 3.5),
        legend_axis=(0, 0),
        legend_kwargs=dict(bbox_to_anchor=(0.4, 0.68), frameon=False),
        performance_ylabel=True,
        expertise_ylabel_kwargs=dict(rotation=0, y=0.02, labelpad=15),
        supxlabel_kwargs=dict(fontsize=helper.FONT_SIZE, x=0.55, y=0.03),
        adjust_figure_kwargs=dict(wspace=0, hspace=0),
        subplots_adjust_kwargs=dict(bottom=0.2),
        left_y_ticks=([0.0, 100.0], ["0", "100"]),
        bottom_xticks={
            (1, 0): ([0, 100, 200, 300, 400, 500], ["", "100", "", "300", "", "500"]),
            (1, 1): ([0, 100, 200, 300, 400, 500], ["", "100", "", "300", "", "500"]),
        },
        top_right_hide_y_ticks=True,
        y_lims_fixed=None,
        box_y_limits=(0.5, 3.5),
    )
    plot_expert_time(save=save, fig_config=fig_config)


def run():
    print("Running simulations for Figure S7")
    simulate_condition(sim_type=NO_OFC_SIM)
    simulate_condition(sim_type=REV_RESET_SIM)
    simulate_condition(sim_type=FAKE_OFC_SIM)

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mp_colors
from ofcsst.simulation import simulate, databases
from ofcsst.simulation import scan as scan_def
from ofcsst.utils import ids, keys, paths, sql, constants
from ofcsst.figures import helper
from ofcsst.figures.main.f1e_explore_distractor import PLOT_DIR

NAME = "relative_gain"
TABLES = [keys.PERFORMANCE, f"{keys.AGENT_GAIN}_T1", f"{keys.AGENT_GAIN}_T2"]
PATH = paths.RESULT_DIR / paths.SIMULATION_SUBDIR / f'fig_1i_{NAME}.db'
SIM_TYPE = ids.SIM_PG_GAIN
NR_SEEDS = constants.NR_SEEDS


def run(verbose: bool = True):
    if verbose:
        print('Running simulations for Figure 1i')

    # Initialize database variables
    scan_db_path = scan_def.get_db_path(scan_type=ids.ST_VARY_DISTRACTOR_1D, simulation_type=SIM_TYPE,
                                        task_id=constants.TASK_ID, non_stationary=False)
    distractor_nrs = [n[0] for n in sql.select(db_path=scan_db_path, table=databases.BEST_PARAM_TABLE,
                                               get_cols=[keys.NR_DISTRACTOR])][6:]
    base_keys = [keys.NR_DISTRACTOR, keys.SEED]
    col_keys = {TABLES[0]: base_keys + [f"{keys.PERFORMANCE}_{t}" for t in range(constants.NR_TRIALS)],
                TABLES[1]: base_keys + [f"{keys.AGENT_GAIN}_{t}" for t in range(constants.NR_TRIALS)],
                TABLES[2]: base_keys + [f"{keys.AGENT_GAIN}_{t}" for t in range(constants.NR_TRIALS)]}

    # Initialize database
    conn = sql.connect(db_path=PATH)
    insert_cmd = {}
    for table in TABLES:
        sql.drop_table(conn=conn, table_name=table, verbose=False)
        sql.make_table(conn=conn, table_name=table, col_names=col_keys[table], verbose=False)
        insert_cmd[table] = sql.get_insert_cmd(table=table, col_keys=col_keys[table])
    cur = conn.cursor()

    # Run for each number of distractor
    get_cols = databases.get_unique_cols(simulation_type=SIM_TYPE, table=databases.SCAN_TABLE)
    for n in range(len(distractor_nrs)):

        # Get the best parameters for each type of simulation
        where_values = {keys.NR_DISTRACTOR: distractor_nrs[n]}
        params = sql.select_where(db_path=scan_db_path, table=databases.BEST_PARAM_TABLE, get_cols=get_cols,
                                  where_values=where_values)[0]
        seeds = list(range((n + 1) * NR_SEEDS, (n + 2) * NR_SEEDS))

        # Run simulations
        for s in range(NR_SEEDS):
            if verbose:
                print(f"\rSimulating with {distractor_nrs[n]} distractors: {100 * s / NR_SEEDS:0.1f}% completed", end="")

            seed = seeds[s]
            correct, logger = simulate.run_simulation(
                task_id=constants.TASK_ID, simulation_type=SIM_TYPE, seed=seed, params=list(params), number_contexts=1,
                log_type=ids.LT_RGAIN
            )

            # Append to database
            values = tuple([distractor_nrs[n], seed] + correct.tolist())
            cur.execute(insert_cmd[TABLES[0]], values)
            values = tuple([distractor_nrs[n], seed] + logger[0].tolist())
            cur.execute(insert_cmd[TABLES[1]], values)
            values = tuple([distractor_nrs[n], seed] + logger[1].tolist())
            cur.execute(insert_cmd[TABLES[2]], values)

        if verbose:
            print(f"\rSimulating with {distractor_nrs[n]} distractors is now complete!")

    conn.commit()
    cur.close()
    del cur
    conn.close()


def plot(save: bool = True):
    get_cols = [f"{keys.AGENT_GAIN}_{t}" for t in range(constants.NR_TRIALS)]

    # Initializations
    scan_db_path = scan_def.get_db_path(scan_type=ids.ST_VARY_DISTRACTOR_1D, simulation_type=ids.SIM_PG_GAIN,
                                        task_id=constants.TASK_ID, non_stationary=False)
    distractor_nrs = [n[0] for n in sql.select(db_path=scan_db_path, table=databases.BEST_PARAM_TABLE,
                                               get_cols=[keys.NR_DISTRACTOR])][6:]
    nr_distractor_nrs = len(distractor_nrs)
    plot_nr_trials = 100

    # Init some plotting cosmetics
    base_colors = [(194./255., 222./255., 228./255.), 'k']
    cmap = mp_colors.LinearSegmentedColormap.from_list(name='', colors=base_colors, N=nr_distractor_nrs)
    colors = cmap(np.linspace(start=0, stop=1, num=nr_distractor_nrs))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mp_colors.Normalize(vmin=0, vmax=1))
    helper.set_style()
    with matplotlib.rc_context({'mathtext.fontset': 'cm'}):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(helper.PANEL_WIDTH, helper.PANEL_HEIGHT))

    # Get data and plot it
    values = np.array(sql.select(db_path=PATH, table=TABLES[1], get_cols=get_cols))
    trials = np.linspace(1, plot_nr_trials, num=plot_nr_trials, endpoint=True)
    for n in range(nr_distractor_nrs):
        y = np.mean(values[n * NR_SEEDS:(n + 1) * NR_SEEDS, :plot_nr_trials], axis=0)
        y_err = np.std(values[n * NR_SEEDS:(n + 1) * NR_SEEDS, :plot_nr_trials], axis=0) / np.sqrt(NR_SEEDS)
        axs.fill_between(trials, y-y_err, y+y_err,
                         alpha=0.2, color=colors[n], linewidth=0)
    for n in range(nr_distractor_nrs):
        axs.plot(trials, np.mean(values[n * NR_SEEDS:(n + 1) * NR_SEEDS, :plot_nr_trials], axis=0), color=colors[n])

    axs.set_ylim([0., 10])
    axs.set_ylabel('Relative gain for $\mathregular{go}$ neuron $\mathcal{\Gamma}_\mathrm{s_1}^\mathrm{ap}$')
    axs.set_xlim([1, plot_nr_trials])
    axs.set_xlabel('Trial')
    axs.spines[['right', 'top']].set_visible(False)
    axs.tick_params(axis='both', which='major', pad=2)
    cbar = plt.colorbar(sm, ax=axs, pad=0.01)
    cbar.set_ticks([(0.5 + i) / nr_distractor_nrs for i in range(nr_distractor_nrs)])
    cbar.set_ticklabels(distractor_nrs)
    cbar.ax.yaxis.set_tick_params(pad=2)
    cbar.ax.minorticks_off()
    cbar.set_label(helper.key_name(key=keys.NR_DISTRACTOR), rotation=270, labelpad=8)

    # Finalize formatting
    helper.adjust_figure(fig=fig)
    plt.subplots_adjust(bottom=0.2, right=0.91)
    helper.set_panel_label(label='i', fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name=f'i_{NAME}')

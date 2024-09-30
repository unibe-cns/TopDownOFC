import numpy as np
from ofcsst import simulation
from ofcsst.utils import ids, paths, constants
from ofcsst.figures import helper
from ofcsst.figures.main.f3efg_ofc_sst import OFC_SIM, NO_OFC_SIM, final_simulation
from ofcsst.figures.main.f3efg_ofc_sst import plot_performance_bars, plot_cumulative_distribution, get_data
from ofcsst.figures.main.f3efg_ofc_sst import COLOR_NO_OFC, COLOR_WT, TRIAL_NRS, SAVE_PATH
import matplotlib.pyplot as plt
import matplotlib.patches as patches

NAME = 'fig_s6_optimization'
TASK_ID = ids.BINARY_2VN
PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs6"
DB_PATH = paths.SUPPLEMENTARY_SIMULATION_DIR / f"{NAME}.db"
COLORS = [(114 / 255., 146 / 255., 179 / 255.), helper.COLOR_GAIN]


def simulate_condition(simulation_type: ids.SimulationType):

    # Initialize variables
    if simulation_type == NO_OFC_SIM:
        scan_type = ids.ST_FINAL
        non_stat = False
        seeds = list(range(constants.NR_SEEDS, 2 * constants.NR_SEEDS))

    elif simulation_type == OFC_SIM:
        scan_type = ids.ST_ALL
        non_stat = True
        seeds = list(range(2 * constants.NR_SEEDS, 3 * constants.NR_SEEDS))

    else:
        raise NotImplementedError(simulation_type)
    scan_db_path = simulation.scan.get_db_path(simulation_type=simulation_type, scan_type=scan_type,
                                               non_stationary=non_stat)

    # Run simulations
    final_simulation(sim_type=simulation_type, scan_path=scan_db_path, seeds=seeds, save_db_path=DB_PATH)


def panel_a_performance_bars(save: bool = True) -> None:
    plot_performance_bars(save=save, db_path=DB_PATH, panel='a', plot_name="a_performance_bars", plot_dir=PLOT_DIR,
                          colors=[COLORS[1], COLORS[0]], labels=['R-Opt\n+lOFC', 'L-Opt\n-lOFC'])


def panel_b_cumulative_distribution(save: bool = True) -> None:
    plot_cumulative_distribution(save=save, db_path=DB_PATH, panel='b', phase_cosmetics=False,
                                 plot_name='b_expertise_cumulative_distribution', plot_dir=PLOT_DIR,
                                 colors=COLORS, labels=['R-Opt (+lOFC)', 'L-Opt (-lOFC)'])


def panel_c_expert_time(save: bool = True) -> None:

    # Basic initializations
    simulation_types = [OFC_SIM, NO_OFC_SIM]
    colors = [COLOR_WT, COLOR_NO_OFC] + COLORS
    db_paths = [SAVE_PATH, DB_PATH]
    bp_props = {'vert': False,
                'widths': 0.5,
                'boxprops': dict(linewidth=helper.AXIS_WIDTH, color='black'),
                'capprops': dict(linewidth=helper.AXIS_WIDTH),
                'whiskerprops': dict(linewidth=helper.AXIS_WIDTH),
                'flierprops': dict(marker='o', markersize=helper.MARKER_SIZE, linewidth=helper.AXIS_WIDTH),
                'medianprops': dict(color='k', solid_capstyle='butt', linewidth=helper.AXIS_WIDTH)}
    pos = [[[1], [2]], [[3], [4]]]
    labels = ['R-Opt mixed (intact lOFC)', 'R-Opt mixed (silenced lOFC)', 'R-Opt with lOFC', 'L-Opt without lOFC']

    # Init plot
    helper.set_style()
    panel_width = 1.7 * helper.PANEL_WIDTH
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(panel_width, helper.PANEL_HEIGHT),
                            height_ratios=[3, 1], width_ratios=[1, 2])

    # Plot data
    for d in range(2):

        # Get data
        t1_trials, t2_trials, t1_xp_t, t2_xp_t, t1_perf, t2_perf = get_data(db_path=db_paths[d], tables=simulation_types)
        t1_perf_mean = np.zeros((2, len(t1_trials)))
        t2_perf_mean = np.zeros((2, len(t2_trials)))
        for i in range(len(simulation_types)):
            t1_perf_mean[i, :] = np.mean(t1_perf[i], axis=0)
            t2_perf_mean[i, :] = np.mean(t2_perf[i], axis=0)

        # Plot all performance traces
        for seed in range(constants.NR_SEEDS):
            for i in range(len(simulation_types)):
                axs[0, 0].plot(t1_trials, t1_perf[i][seed, :], color=colors[d * 2 + i], alpha=0.05)
                axs[0, 1].plot(t2_trials, t2_perf[i][seed, :], color=colors[d * 2 + i], alpha=0.05)

        # Plot mean performance traces
        for i in range(2):
            axs[0, 0].plot(t1_trials, t1_perf_mean[i, :], color=colors[d * 2 + i], linewidth=1.5 * helper.LINE_WIDTH)
            axs[0, 1].plot(t2_trials, t2_perf_mean[i, :], color=colors[d * 2 + i], label=labels[2 * d + i],
                           linewidth=1.5 * helper.LINE_WIDTH)

        # Expertise threshold trial distributions
        for i in range(len(simulation_types)):
            axs[1, 0].boxplot(t1_xp_t[~np.isnan(t1_xp_t[:, i]), i], positions=pos[d][i], **bp_props)
            n_reversal_experts = len(t2_xp_t[~np.isnan(t2_xp_t[:, i]), i])
            if n_reversal_experts > constants.NR_SEEDS / 2:
                axs[1, 1].boxplot(t2_xp_t[~np.isnan(t2_xp_t[:, i]), i], positions=pos[d][i], **bp_props)
            else:
                axs[1, 1].scatter(t2_xp_t[~np.isnan(t2_xp_t[:, i]), i], pos[d][i] * n_reversal_experts,
                                  s=helper.MARKER_SIZE,
                                  ec='k', linewidth=helper.AXIS_WIDTH, marker='o')

        # Expert performance threshold
        axs[0, d].plot([0, TRIAL_NRS[d] - 1], [constants.EXPERT_PERFORMANCE, constants.EXPERT_PERFORMANCE],
                       zorder=0, **helper.STYLE_EXPERT_PERF)

    # Plot some cosmetics
    rect_style = dict(height=1., linewidth=0, alpha=0.4)
    for i in range(4):
        rect = patches.Rectangle(xy=(0, i + 0.5), width=TRIAL_NRS[0], facecolor=colors[i], **rect_style)
        axs[1, 0].add_patch(rect)
        rect = patches.Rectangle(xy=(0, i + 0.5), width=TRIAL_NRS[1], facecolor=colors[i], **rect_style)
        axs[1, 1].add_patch(rect)
    axs[1, 1].plot([0, 0], [-11.5, 4.5], clip_on=False, zorder=10,
                   **helper.STYLE_RULE_SWITCH)
    axs[0, 0].text(constants.NR_TRIALS - 30 * helper.PANEL_WIDTH / panel_width, 55, rotation=90,
                   **helper.FONT_RULE_SWITCH)

    # Legend and labels
    axs[0, 0].set_ylabel("Performance")
    axs[1, 0].set_ylabel('Trials to\nexpertise', rotation=0, y=0.25, labelpad=15)
    fig.supxlabel("Trials (after start/reversal)", fontsize=helper.FONT_SIZE, x=0.55, y=0.03)
    axs[0, 1].legend(loc='lower center', frameon=False, borderpad=0, handletextpad=0.5, handlelength=1, ncols=2,
                     columnspacing=1.5)

    # Set axes
    for i in range(2):
        axs[0, i].set_xlim([0, TRIAL_NRS[i]])
        axs[1, i].set_xlim([0, TRIAL_NRS[i]])
        axs[0, i].set_ylim([0, 100])
        axs[1, i].set_ylim([0.5, 4.5])
        axs[1, i].invert_yaxis()
        axs[0, i].set_xticks([])
        axs[1, i].set_yticks([])
    axs[0, 0].spines[['bottom', 'right', 'top']].set_visible(False)
    axs[0, 1].spines[['bottom', 'left', 'right', 'top']].set_visible(False)
    axs[1, 0].spines[['right', 'top']].set_visible(False)
    axs[1, 1].spines[['left', 'right', 'top']].set_visible(False)
    axs[0, 0].spines['left'].set_bounds(0, 100)
    axs[1, 0].set_xticks([0, 100, 200, 300, 400, 500], ['', '100', '', '', '400', ''])
    axs[1, 1].set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
                         ['', '100', '', '', '400', '', '', '700', '', '', '1000', '', ''])
    axs[0, 0].set_yticks([0., 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.],
                         ['', '10', '', '30', '', '50', '', '70', '', '90', ''])
    axs[0, 1].set_yticks([])

    # Finalize formatting
    helper.adjust_figure(fig=fig, hspace=0, wspace=0)
    fig.subplots_adjust(bottom=0.2, right=0.975)
    helper.set_panel_label(label='c', fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name='c_expert_time', plot_dpi=500)


def scan():
    simulation.scan.find_best_params(simulation_type=NO_OFC_SIM, task_id=TASK_ID, switch_task=False,
                                     scan_type=ids.ST_FINAL)
    simulation.scan.find_best_params(simulation_type=OFC_SIM, task_id=TASK_ID, switch_task=True, scan_type=ids.ST_ALL)


def run():
    print('Running simulations for Figure S6')
    simulate_condition(simulation_type=OFC_SIM)
    simulate_condition(simulation_type=NO_OFC_SIM)


def plot(save: bool = True):
    panel_a_performance_bars(save=save)
    panel_b_cumulative_distribution(save=save)
    panel_c_expert_time(save=save)

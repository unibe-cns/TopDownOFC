import random
import torch
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ofcsst.utils import constants, ids, keys, paths, sql
from ofcsst import simulation
from ofcsst.figures import helper
from ofcsst.figures.main.f3efg_ofc_sst import OFC_SCAN

PLOT_DIR = paths.MAIN_FIG_DIR / "fig3"
NAME = "selectivity"
TABLES = [keys.PERFORMANCE, f"{keys.AGENT_GAIN}_T1", f"{keys.AGENT_GAIN}_T2"]
SAVE_DIR = paths.RESULT_DIR / paths.SIMULATION_SUBDIR
WITH_OFC = ids.SIM_CONVEX_OFC
WITHOUT_OFC = ids.SIM_CONVEX_GAIN
PHASE_LENGTH = constants.PERF_MAV_N  # Number of trials in which to evaluate the naive and expert states
NR_SEEDS = 10
COLOR_WT = helper.COLOR_DEFAULT
COLOR_NO_OFC = helper.COLOR_SST
S1S2_HITCR = True


def get_path(sim_type: ids.SimulationType):
    return SAVE_DIR / f'fig_3c_selectivity_{sim_type}.npy'


def simulate(simulation_type: ids.SimulationType, params: [str], seed: int = 0,
             task_id: ids.TaskID = ids.BINARY_2VN, number_contexts: int = 2) -> np.ndarray:

    # Set pseudorandom seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    number_trials = [constants.NR_TRIALS] * number_contexts
    corrects = np.array([-666.666] * sum(number_trials))
    assert simulation_type in [WITH_OFC, WITHOUT_OFC]
    agent, task, _ = simulation.simulate.init_simulation(
        task_id=task_id, simulation_type=simulation_type, params=params, trial_nrs=tuple(number_trials)
    )
    n_s1 = 0
    s1_response = 0.
    s2_response = 0.
    z_response = np.zeros((4, 2))

    for reversal_idx in range(number_contexts):
        naive_phase = [0, PHASE_LENGTH]
        expert_phase = [400 - PHASE_LENGTH, 400]

        # Set current rule of the task
        task.set_task(reversal=reversal_idx % 2 != 0)

        for trial in range(number_trials[reversal_idx]):
            trial_id = sum(number_trials[:reversal_idx]) + trial
            stim_1 = bool(random.getrandbits(1))
            basal_inputs = task.init_stimuli(stimulus=stim_1)
            z = agent.get_representation(basal_input=basal_inputs)
            action = agent.get_action(basal_input=basal_inputs)
            reward, correct = task.get_outcome(acted_upon=action)
            agent.update(reward=reward)
            corrects[trial_id] = correct
            if naive_phase[0] <= trial < naive_phase[1]:
                if S1S2_HITCR or correct:
                    if stim_1:
                        n_s1 += 1
                        s1_response += torch.sum(z)
                    else:
                        s2_response += torch.sum(z)
                if trial == naive_phase[1] - 1:
                    z_response[2 * reversal_idx, 0] = s1_response / n_s1
                    z_response[2 * reversal_idx, 1] = s2_response / (PHASE_LENGTH - n_s1)
                    n_s1 = 0
                    s1_response = 0.
                    s2_response = 0.
            elif expert_phase[0] <= trial < expert_phase[1]:
                if S1S2_HITCR or correct:
                    if stim_1:
                        n_s1 += 1
                        s1_response += torch.sum(z)
                    else:
                        s2_response += torch.sum(z)
                if trial == expert_phase[1] - 1:
                    z_response[2 * reversal_idx + 1, 0] = s1_response / n_s1
                    z_response[2 * reversal_idx + 1, 1] = s2_response / (PHASE_LENGTH - n_s1)
                    n_s1 = 0
                    s1_response = 0.
                    s2_response = 0.

    return z_response


def run_simulation(sim_type: ids.SimulationType = WITH_OFC):

    # Init and get the best parameters
    z_responses = np.zeros((NR_SEEDS, 4, 2))
    if sim_type == WITH_OFC:
        scan_db_path = simulation.scan.get_db_path(scan_type=OFC_SCAN, simulation_type=sim_type)
        seeds = list(range(constants.NR_SEEDS, 2 * constants.NR_SEEDS))
    elif sim_type == WITHOUT_OFC:
        scan_db_path = simulation.scan.get_db_path(scan_type=ids.ST_FINAL, simulation_type=sim_type)
        seeds = list(range(2 * constants.NR_SEEDS, 3 * constants.NR_SEEDS))
    else:
        raise ValueError(sim_type)
    get_cols = simulation.databases.get_unique_cols(simulation_type=sim_type, table=simulation.databases.SCAN_TABLE)
    where_values = {keys.NR_DISTRACTOR: constants.NUMBER_DISTRACTOR}
    params = sql.select_where(db_path=scan_db_path, table=simulation.databases.BEST_PARAM_TABLE, get_cols=get_cols,
                              where_values=where_values)[0]

    # Run simulations
    for s in range(NR_SEEDS):
        print(f"\rSimulating {100 * s / NR_SEEDS:0.1f}% completed", end="")

        seed = seeds[s]
        z_responses[s, :, :] = simulate(simulation_type=sim_type, seed=seed, params=list(params))

    print(f"\rSimulating {sim_type} complete!")

    # Save results
    np.save(get_path(sim_type=sim_type), z_responses)


def run():
    print('Running simulations for Figure 3c')
    run_simulation(sim_type=WITH_OFC)
    run_simulation(sim_type=WITHOUT_OFC)


def plot_responses(save: bool = True, verbose: bool = False):

    phase_idx = [1, 3]

    # Get data and initializations dependent on the data
    data_with_ofc = np.load(get_path(sim_type=WITH_OFC))
    data_without_ofc = np.load(get_path(sim_type=WITHOUT_OFC))
    data = ((data_with_ofc[:, phase_idx, 0], data_with_ofc[:, phase_idx, 1]),
            (data_without_ofc[:, phase_idx, 0], data_without_ofc[:, phase_idx, 1]))
    y_max = [85, 85]
    y_min = [60, 60]
    y_min_s = [60, 60]
    y_max_s = [80, 80]
    ylabel = 'Sensory response      '
    textstr = ['Intact\nlOFC', 'Silenced\nlOFC']
    y_txt = 1.18
    trial_type = [r'$\mathregular{s_1}$', r'$\mathregular{s_2}$']
    panel = 'c'
    plot_name = 'c_z_response'
    y_ticks = [60, 70, 80]

    # Init plot
    helper.set_style()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(1.1 * helper.PANEL_WIDTH, 0.8 * helper.PANEL_HEIGHT))
    outcome_colors = [helper.COLOR_LN, helper.COLOR_LE, helper.COLOR_RN, helper.COLOR_RE]
    outcome_colors = [outcome_colors[p] for p in phase_idx]
    cond_colors = [COLOR_WT, COLOR_NO_OFC]
    cond_x = 0.7
    phases = ['LN', 'LE', 'RN', 'RE']
    phases = [phases[p] for p in phase_idx]
    n_phases = len(phases)
    ps = list(range(n_phases))
    p_width = 3
    x_ticks = [p_width * p + 0.5 for p in ps]
    x_rev_i = next(x for x, val in enumerate(phase_idx) if val > 1.5)
    x_rev_data = x_rev_i * p_width - 1
    x_rev_ax = x_rev_i / n_phases
    b_plots = [None, None]
    box_width = 0.85
    bp_props = {'vert': True,
                'widths': box_width,
                'patch_artist': True,
                'showfliers': False,
                'boxprops': dict(linewidth=helper.AXIS_WIDTH),
                'capprops': dict(linewidth=helper.AXIS_WIDTH),
                'whiskerprops': dict(linewidth=helper.AXIS_WIDTH, solid_capstyle='butt'),
                'flierprops': dict(marker='o', markersize=helper.MARKER_SIZE, linewidth=helper.AXIS_WIDTH),
                'medianprops': dict(color='k', solid_capstyle='butt', linewidth=helper.AXIS_WIDTH)}

    # Plot box-plots
    for condition in range(2):
        for outcome in range(2):
            pos = [p_width * p + outcome for p in ps]
            box_y = [data[condition][outcome][:, i][~np.isnan(data[condition][outcome][:, i])] for i in range(len(pos))]
            b_plots[outcome] = axs[condition].boxplot(x=box_y, positions=pos, **bp_props)
            for median, patch, color in zip(b_plots[outcome]['medians'], b_plots[outcome]['boxes'], outcome_colors):
                if outcome == 0:
                    patch.set(linewidth=0)
                    patch.set_facecolor(color)
                    if color == helper.COLOR_LE:
                        median.set_color('white')
                else:
                    patch.set(color=color)
                    patch.set_facecolor('white')

        # Axis and label cosmetics
        if y_min is None:
            ymin, _ = axs[condition].get_ylim()
        else:
            ymin = y_min[condition]
        axs[condition].set_ylim([ymin, y_max[condition]])
        axs[condition].set_xticks(x_ticks, phases)
        axs[condition].set_xlim([-1, n_phases * p_width - 1])
        axs[condition].spines[['right', 'top', 'bottom']].set_visible(False)
        axs[condition].spines.left.set_bounds([y_min_s[condition], y_max_s[condition]])
        circ = patches.Circle(xy=(cond_x, 1.2), radius=0.2, facecolor=cond_colors[condition],
                              transform=axs[condition].transAxes, clip_on=False)
        axs[condition].add_patch(circ)
        axs[condition].text(cond_x, y_txt, textstr[condition], transform=axs[condition].transAxes, ha='center',
                            va='center')
        rule_switch_y = [ymin - (y_max[condition] - ymin) * 0.1, y_max[condition] - (y_max[condition] - ymin) * 0.05]
        axs[condition].plot([x_rev_data, x_rev_data], rule_switch_y, clip_on=False,
                            **helper.STYLE_RULE_SWITCH)
        axs[condition].tick_params(axis='x', which='major', pad=0)
        axs[condition].tick_params(bottom=False)
        if y_ticks is not None:
            axs[condition].set_yticks(y_ticks)
        tls = axs[condition].get_xticklabels()
        for p in ps:
            tls[p].set_color(outcome_colors[p])
            tls[p].set_fontweight('bold')

    # Legend
    axs[0].set_ylabel(ylabel)
    hit_patch = patches.Patch(color='black', label=trial_type[0])
    cr_patch = patches.Patch(facecolor='white', ec="black", label=trial_type[1])
    axs[1].legend(loc="lower right", handles=[hit_patch, cr_patch], handlelength=box_width, frameon=False,
                  bbox_to_anchor=(0.4, 0.99), borderpad=0, ncol=2, columnspacing=1)

    # Significant difference statistics
    conditions = ['WT', 'OFC_KO']
    box_x0 = [p_width * p for p in ps]
    c_sig, c_sig_ns = 0.75, 0.77
    for c in range(len(conditions)):
        ymin, y_max = axs[c].get_ylim()
        y_sig = ymin + c_sig * (y_max - ymin)
        y_sig_ns = ymin + c_sig_ns * (y_max - ymin)
        for p in ps:
            d1 = data[c][0][:, p]
            d2 = data[c][1][:, p]
            pv = stats.ttest_ind(d1[~np.isnan(d1)], d2[~np.isnan(d2)])[1] * n_phases  # Bonferroni corrected p-value
            if verbose:
                print(f"{conditions[c]} {phases[p]}: {trial_type[0]} vs. {trial_type[1]}", pv)
            axs[c].plot([box_x0[p], box_x0[p]+1], [y_sig, y_sig], **helper.STYLE_SIGNIFICANT)
            if pv < constants.SIGNIFICANCE_THRESHOLD:
                axs[c].text(x=box_x0[p] + 0.5, y=y_sig, s=helper.get_significance(pv), **helper.FONT_SIGNIFICANT)
            else:
                axs[c].text(x=box_x0[p] + 0.5, y=y_sig_ns, **helper.FONT_NON_SIGNIFICANT)

    # Finalize formatting
    helper.adjust_figure(fig=fig, wspace=0.3)
    fig.subplots_adjust(top=0.7, left=0.18)
    helper.set_panel_label(label=panel, fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name=plot_name)


def plot(save: bool = True) -> None:
    plot_responses(save=save)

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as m_colors
from random import seed as rd_seed
from numpy.random import seed as np_seed
from torch import manual_seed as torch_seed
from scipy.stats import norm
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ofcsst.utils import constants, ids, keys, paths, sql
from ofcsst.simulation import agents, databases, network, scan, tasks
from ofcsst.figures import helper
from ofcsst.figures.main.f2cef_reversal import PLOT_DIR


N_STAGES = 5
N_TRIALS = 200
PATH = paths.RESULT_DIR / paths.SIMULATION_SUBDIR / 'fig_2bc_var_estimates.pkl'


def simulate(params: dict, seed: int = 0, nr_trials: int = constants.NR_TRIALS):

    # Setting random seed
    rd_seed(seed)
    np_seed(seed)
    torch_seed(seed)

    # Initialize agent and environment
    tot_nr_neurons = params[keys.NR_SIGNAL] + params[keys.NR_DISTRACTOR]
    h_actor = network.PGActor(input_size=tot_nr_neurons, learning_rate=params[keys.LEARNING_RATE_PG])
    q_actor = network.QActor(input_size=tot_nr_neurons, learning_rate=params[keys.LEARNING_RATE_Q])
    gain_modulator = network.GainModulator(representation_size=tot_nr_neurons, v_lr=params[keys.LEARNING_RATE_V])
    agent = agents.ConvexAgentGain(habit_actor=h_actor, goal_actor=q_actor, gain_modulator=gain_modulator)
    variance_estimator = network.VarianceEstimator()
    task = tasks.get_task(task_id=constants.TASK_ID, nr_distractor=params[keys.NR_DISTRACTOR],
                          nr_signal=constants.NUMBER_SIGNAL)

    # Values to monitor and store during simulation
    outcomes = np.full(shape=(nr_trials,), fill_value=np.nan)
    outcome_pred_t1 = []
    outcome_pred_t2 = []
    variance_estimates = np.full(shape=(nr_trials,), fill_value=np.nan)
    t1_trials = []
    t2_trials = []

    # Simulating trial after trial
    for trial_id in range(nr_trials):
        representation = task.init_stimuli()
        t1 = representation[0] > 0.
        action = agent.get_action(basal_input=representation)
        reward, _ = task.get_outcome(acted_upon=action)
        outcomes[trial_id] = reward
        r_pred = agent.get_r_pred(action=0)

        if t1:
            t1_trials += [trial_id]
            outcome_pred_t1 += [r_pred]
        else:
            t2_trials += [trial_id]
            outcome_pred_t2 += [r_pred]
        spe = torch.abs(reward - agent.get_r_pred())
        variance_estimates[trial_id] = variance_estimator.get()
        agent.update(reward=reward)
        variance_estimator.update(prediction_error=spe)

    return t1_trials, t2_trials, outcome_pred_t1, outcome_pred_t2, variance_estimates


def run(n_seeds: int = constants.NR_SEEDS, nr_trials: int = N_TRIALS):
    print('Running simulations for Figure 2bc')

    db_path = scan.get_db_path(scan_type=ids.ST_FINAL, simulation_type=ids.SIM_CONVEX_GAIN,
                               task_id=constants.TASK_ID, non_stationary=False)
    pg_lr, q_lr, ap_lr = sql.select(db_path=db_path, table=databases.BEST_PARAM_TABLE,
                                    get_cols=[keys.LEARNING_RATE_PG, keys.LEARNING_RATE_Q, keys.LEARNING_RATE_V])[0]
    params = {keys.NR_SIGNAL: constants.NUMBER_SIGNAL,
              keys.NR_DISTRACTOR: constants.NUMBER_DISTRACTOR,
              keys.LEARNING_RATE_PG: pg_lr,
              keys.LEARNING_RATE_Q: q_lr,
              keys.LEARNING_RATE_V: ap_lr}

    outcome_pred_t1 = np.full(shape=(n_seeds, nr_trials), fill_value=np.nan)
    outcome_pred_t2 = np.full(shape=(n_seeds, nr_trials), fill_value=np.nan)
    variance_estimates = np.zeros((n_seeds, nr_trials))
    for seed in range(n_seeds):
        print(f'\rFigure 2bc: Simulating seed {seed}/{n_seeds}', end="")
        t1_t, t2_t, pred_t1, pred_t2, variance_estimates[seed, :] = simulate(params, seed, nr_trials)
        outcome_pred_t1[seed, t1_t] = pred_t1
        outcome_pred_t2[seed, t2_t] = pred_t2
    print(f'\rFigure 2bc: Simulation is now completed!')

    data = {'pred_t1_av': np.nanmean(outcome_pred_t1, axis=0),
            'pred_t2_av': np.nanmean(outcome_pred_t2, axis=0),
            'var_av': np.mean(variance_estimates, axis=0),
            'pred_t1_std': np.nanstd(outcome_pred_t1, axis=0),
            'pred_t2_std': np.nanstd(outcome_pred_t2, axis=0),
            'var_std': np.nanstd(variance_estimates, axis=0)}

    with open(PATH, 'wb') as f:
        pickle.dump(data, f)


def plot(save: bool = True, n_seeds: int = constants.NR_SEEDS, loading: bool = True, nr_trials: int = N_TRIALS):

    # If data is unavailable, run simulation
    if not (loading and PATH.is_file()):
        run(n_seeds=n_seeds, nr_trials=nr_trials)

    with open(PATH, 'rb') as f:
        data = pickle.load(f)

    # Init reward prediction and predicted variance plot
    trials = list(range(nr_trials))
    helper.set_style()
    plot_dir = paths.MAIN_FIG_DIR / 'f2_variance'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_dist_trials = np.linspace(0, nr_trials, N_STAGES + 1, endpoint=True)
    normalize = m_colors.Normalize(vmin=1, vmax=nr_trials)
    cmap = helper.get_performance_cmap(phase='learning')
    cbar = m_colors.LinearSegmentedColormap.from_list("Custom", [cmap(i) for i in range(cmap.N)], N=N_STAGES)
    scalar_map = cm.ScalarMappable(norm=normalize, cmap=cbar)
    scalar_map.set_array([1 + p for p in plot_dist_trials])

    # Plot evolution of outcome prediction and the estimated variance
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(helper.PANEL_WIDTH, helper.PANEL_HEIGHT),
                            gridspec_kw={'width_ratios': [3, 1, 1]})
    right_edge = nr_trials * 1.9
    for y in [-1, 0., 1]:
        axs[0].plot([0, right_edge], [y, y], linestyle=(0, (6, 6)), color='grey', linewidth=helper.AXIS_WIDTH / 2.,
                    clip_on=False)
    fill_style = {'alpha': 0.4, 'linewidth': 0}
    colors = ['k', (149. / 255., 149. / 255., 149. / 255.), helper.COLOR_LN]
    axs[0].fill_between(trials, data['pred_t1_av'] - data['pred_t1_std'], data['pred_t1_av'] + data['pred_t1_std'],
                        color=colors[2], **fill_style)
    axs[0].fill_between(trials, data['pred_t2_av'] - data['pred_t2_std'], data['pred_t2_av'] + data['pred_t2_std'],
                        color=colors[2], **fill_style)
    axs[0].plot(trials, data['pred_t1_av'], color=colors[0], label=r'$\mathregular{s_1}$')
    axs[0].plot(trials, data['pred_t2_av'], color=colors[1], label=r'$\mathregular{s_2}$')
    var_col = (0.35, 0.35, 0.35)
    divider = make_axes_locatable(axs[0])
    ax_low = divider.append_axes("bottom", size="50%", pad=0.05)
    ax_low.fill_between(trials, data['var_av'] - data['var_std'], data['var_av'] + data['var_std'],
                        color=helper.COLOR_LN, **fill_style)
    ax_low.plot(trials, data['var_av'], color=var_col)
    axs[0].legend(loc="upper right", bbox_to_anchor=(1, 0.9), borderaxespad=0,
                  frameon=False)

    # Set axes
    axs[0].set_xticks([])
    ax_low.set_xticks([0, 50, 100, 150, 200], ['', '50', '100', '150', '200'])
    axs[0].set_yticks([-1, 0, 1.])
    ax_low.set_yticks([0, 0.5])
    axs[0].set_ylim([-1.05, 1.05])
    ax_low.set_ylim([0, None])
    axs[0].spines[['left']].set_bounds([constants.PUNISHMENT, constants.REWARD])
    axs[0].set_ylabel(r'$\mathregular{\hat{Q}(a_{L}, z)}$')
    ax_low.set_ylabel(r'$\mathregular{\hat{\sigma}^2}$')
    axs[0].set_xlim([0, nr_trials])
    ax_low.set_xlim([0, nr_trials])
    ax_low.set_xlabel("Trial")
    axs[0].spines[['right', 'top', 'bottom']].set_visible(False)
    ax_low.spines[['right', 'top']].set_visible(False)

    # Learning progress color bar
    cax = divider.append_axes('top', size='10%', pad=0.)
    cb = fig.colorbar(scalar_map, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cax.xaxis.set_ticks_position('top')
    tick_x = 0.6 * nr_trials / N_STAGES
    cax.set_xticks([tick_x, nr_trials - tick_x], ["Naïve", "Expert"])
    cax.tick_params('x', length=0, which='major', pad=1)
    nax = divider.append_axes('top', size='20%', pad=0.)
    nax.axis('off')

    # Init probability density plot
    y_min, y_max = -1.895, 1.68
    x_min, x_max = -0.1, 3.5
    ys = np.linspace(y_min, y_max, num=300)
    colors = cmap([i / (N_STAGES-1) for i in range(N_STAGES)])

    # Plot evolution of predictions as gaussian density functions during learning (for each stimulus)
    for sub_plot_idx in range(2):
        if sub_plot_idx == 0:
            r_pred = data['pred_t1_av']
        else:
            r_pred = data['pred_t2_av']

        # Plot outcome expectation PDFs for each stage of learning
        for t in (range(len(plot_dist_trials) - 1)):

            # Get trials for stage
            t1 = int(plot_dist_trials[t])
            t2 = int(plot_dist_trials[t + 1] - 1)

            # Compute mean and std
            prob_mean = np.mean(r_pred[t1:t2])
            prob_std = np.sqrt(np.mean(data['var_av'][t1:t2]))

            # Plot PDF
            prob = norm.pdf(prob_mean, ys, prob_std)
            axs[1 + sub_plot_idx].plot(prob, ys, color=colors[t], linewidth=helper.LINE_WIDTH)

        # Cosmetics
        axs[1 + sub_plot_idx].set_xlim([x_min, x_max])
        axs[1 + sub_plot_idx].set_xticks([])
        axs[1 + sub_plot_idx].set_ylim([y_min, y_max])
        axs[1 + sub_plot_idx].set_yticks([])
        axs[1 + sub_plot_idx].spines[['left', 'bottom', 'right', 'top']].set_visible(False)
        axs[1 + sub_plot_idx].set_facecolor((0, 0, 0, 0))
        divider = make_axes_locatable(axs[1 + sub_plot_idx])
        nax = divider.append_axes("bottom", size="10%", pad=0.)
        nax.axis('off')
    axs[1].set_xlabel(r'$\mathregular{p(R | a_L, s_1)}$', rotation=310, loc='left', labelpad=-0.5)
    axs[2].set_xlabel(r'$\mathregular{p(R | a_L, s_2)}$', rotation=310, loc='left', labelpad=-0.5)

    # Illustrate unexpected reward or punishment
    colors = [helper.COLOR_PUNISHMENT, helper.COLOR_REWARD]
    outcomes = [-1, 1]
    texts = ['Punishment', 'Reward']
    for i in range(2):
        axs[1 + i].annotate(texts[i], xy=(0., outcomes[i]), xytext=(0.5 * x_max, outcomes[i]),
                            color=colors[i], va='center', arrowprops=dict(arrowstyle="-|>", color=colors[i]),
                            bbox=dict(pad=-0.5, facecolor="none", edgecolor="none"), rotation=270)

    # Finalize figure
    helper.adjust_figure(fig=fig, hspace=0., wspace=0.2)
    fig.subplots_adjust(top=0.93)
    helper.set_panel_label(label="b", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name='b_var_pred', png=True)

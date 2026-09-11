import random
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as m_colors
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from typing import cast, Literal
from math import exp
from ofcsst.utils import constants, ids, keys, paths, sql
from ofcsst.simulation import databases, network, scan, tasks
from ofcsst.figures import helper, statistics

N_TRIALS = 200
N_TASKS = 2
PATH = paths.RESULT_DIR / paths.SIMULATION_SUBDIR / "fig_2_reversal.pkl"
PLOT_DIR = paths.MAIN_FIG_DIR / "fig2"

# Keys for recorded variables of interest
VKey = Literal[
    "confidence_av",
    "confidence_loss_av",
    "context_prediction_error_av",
    "context_prediction_error",
]
C_AV: VKey = "confidence_av"
CL_AV: VKey = "confidence_loss_av"
CPE_AV: VKey = "context_prediction_error_av"
CPE: VKey = "context_prediction_error"


class FakeLearner:
    """
    A Custom agent that simulates a prototypical learning curve for a binary task with reversals.
    The agent's performance is modeled using a sigmoid function.
    """

    def __init__(self, nr_trials=constants.NR_TRIALS):
        self.n_trials = nr_trials
        self.perf_fct = lambda v: -0.1 + (
            1.0 / (1.0 + exp(1.0 - 6 * v / nr_trials) / 0.5)
        ) ** (np.log(0.5) / np.log(1.0 / (1.0 + exp(1.0) / 0.5)))
        self.rev_perf_fct = lambda v: 1 - exp(-(v + 1) * 3 / self.n_trials)
        self.reward = torch.tensor([constants.REWARD])
        self.punishment = torch.tensor([constants.PUNISHMENT])

    def get_outcome(self, trial_id, stimulus: bool, reversal: bool = False):
        if reversal:
            performance = self.rev_perf_fct(trial_id)
        else:
            performance = self.perf_fct(trial_id)
        correct_action = bool(
            np.random.choice([0, 1], p=[1 - performance, performance])
        )
        nogo = (reversal and stimulus) or (not reversal and not stimulus)
        if correct_action and not nogo:
            return self.reward, True
        elif not correct_action and nogo:
            return self.punishment, True
        else:
            return torch.tensor([0.0]), False


def simulate(
    seed: int = 0, nr_trials_per_task: int = N_TRIALS, nr_tasks: int = N_TASKS
):
    """
    Simulate the evolution of different estimators for the expected reward, uncertainty, confidence, confidence loss,
    and context-prediction error during a prototypical learning trajectory of a binary task with a rule reversal.
    """

    # Pseudorandom seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initializations
    nr_distractor = constants.NUMBER_DISTRACTOR
    sim = FakeLearner(nr_trials=nr_trials_per_task)
    tot_nr_neurons = nr_distractor + constants.NUMBER_SIGNAL
    task = tasks.Binary2VN(
        nr_distractor=nr_distractor, nr_signal=constants.NUMBER_SIGNAL
    )
    scan_db_path = scan.get_db_path(
        scan_type=ids.ST_TD, simulation_type=ids.SIM_CONVEX_OFC, non_stationary=True
    )

    select_cols = [keys.LEARNING_RATE_Q, keys.LEARNING_RATE_V]
    q_lr, v_lr = sql.get_max(
        db_path=scan_db_path,
        table=databases.BEST_PARAM_TABLE,
        group_cols=[],
        select_cols=select_cols,
        max_col=keys.PERFORMANCE,
        maxmin=True,
    )[0]
    ofc = network.OFC(input_size=tot_nr_neurons, q_learning_rate=q_lr)
    gain_modulator = network.GainModulatorV(
        representation_size=tot_nr_neurons, v_lr=v_lr
    )

    # Define containers to track values of interest during simulation
    tot_trials = nr_tasks * nr_trials_per_task
    confidence = np.array([-666.666] * tot_trials)
    confidence_loss = np.array([-666.666] * tot_trials)
    context_prediction_error = np.array([-666.666] * tot_trials)

    # Simulating trial after trial
    for reversal_idx in range(2):
        reversal = reversal_idx % 2 != 0
        task.set_task(reversal=reversal)
        for trial in range(nr_trials_per_task):
            # Simulate trial and update the different estimators
            trial_id = reversal_idx * nr_trials_per_task + trial
            stimulus = bool(random.getrandbits(1))
            representation = task.init_stimuli(stimulus=stimulus)
            representation = representation * gain_modulator.get()
            reward, action = sim.get_outcome(
                trial_id=trial, stimulus=stimulus, reversal=reversal
            )
            gain_modulator.simulate(representation=representation)
            ofc.simulate(representation=representation, action=action)
            ofc.update(outcome=reward)
            gain_modulator.update(reward=reward)

            # Record the estimator values for the current trial
            (
                _,
                confidence[trial_id],
                confidence_loss[trial_id],
                context_prediction_error[trial_id],
            ) = ofc.get_log()

    return confidence, confidence_loss, context_prediction_error


def run(
    n_seeds: int = constants.NR_SEEDS,
    nr_trials_per_task: int = N_TRIALS,
    nr_reversals: int = N_TASKS,
):
    print("Running simulations for Figure 2ef")

    tot_nr_trials = nr_trials_per_task * nr_reversals
    confidence = np.zeros(shape=(n_seeds, tot_nr_trials))
    confidence_loss = np.zeros(shape=(n_seeds, tot_nr_trials))
    cpe = np.zeros(shape=(n_seeds, tot_nr_trials))
    for seed in range(n_seeds):
        print(f"\rFigure 2ef: Simulating seed {seed}/{n_seeds}", end="")
        confidence[seed, :], confidence_loss[seed, :], cpe[seed, :] = simulate(
            seed=seed, nr_trials_per_task=nr_trials_per_task, nr_tasks=nr_reversals
        )
    print("\rFigure 2ef: Simulation is now completed!")

    data = {
        C_AV: np.mean(confidence, axis=0),
        CL_AV: np.mean(confidence_loss, axis=0),
        CPE_AV: np.mean(cpe, axis=0),
        CPE: cpe,
    }

    with open(PATH, "wb") as f:
        pickle.dump(data, f)

    return data


def get_data(n_seeds=constants.NR_SEEDS, loading: bool = True):
    nr_reversals = N_TASKS
    nr_trials_per_task = N_TRIALS
    if loading and PATH.is_file():
        with open(PATH, "rb") as f:
            data = pickle.load(f)
    else:
        data = run(
            n_seeds=n_seeds,
            nr_trials_per_task=nr_trials_per_task,
            nr_reversals=nr_reversals,
        )
    return data


def init_panel():
    nr_reversals = N_TASKS
    nr_trials_per_task = N_TRIALS
    tot_nr_trials = nr_trials_per_task * nr_reversals
    trials = list(range(1, 1 + tot_nr_trials))
    x_ticks = [0, 100, 200, 300, 400]
    x_tick_labels = ["", "100", "200", "300", ""]
    return (
        nr_reversals,
        nr_trials_per_task,
        tot_nr_trials,
        trials,
        x_ticks,
        x_tick_labels,
    )


def plot_box_plots(
    data,
    panel: str,
    y_label: str,
    y_lim: tuple[float, float],
    plot_name: str,
    save: bool = True,
    stim_w: bool = False,
    verbose: bool = False,
) -> None:
    # Init figure
    helper.set_style()
    fig, axs = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(helper.PANEL_WIDTH, helper.PANEL_HEIGHT),
        squeeze=False,
    )
    fig = cast(plt.Figure, fig)
    axs = axs[0, 0]
    y_min = y_lim[0] - 0.1 * (y_lim[1] - y_lim[0])

    # Plot box-plots
    phases = list(range(4))
    xs = [-1.5, -0.6, 0.6, 1.5]
    phase_colors = [helper.COLOR_LN, helper.COLOR_LE, helper.COLOR_RN, helper.COLOR_RE]
    tls = axs.get_xticklabels()
    bp_props = {
        "vert": True,
        "widths": 0.6,
        "patch_artist": True,
        "showfliers": False,
        "boxprops": dict(linewidth=0),
        "capprops": dict(linewidth=helper.LINE_WIDTH),
        "whiskerprops": dict(linewidth=helper.LINE_WIDTH, solid_capstyle="butt"),
        "flierprops": dict(
            marker="o", markersize=helper.MARKER_SIZE, linewidth=helper.LINE_WIDTH
        ),
        "medianprops": dict(
            color="k", solid_capstyle="butt", linewidth=helper.LINE_WIDTH
        ),
    }
    for phase in phases:
        b_plots = axs.boxplot(x=data[phase], positions=[xs[phase]], **bp_props)
        for patch in b_plots["boxes"]:
            patch.set_facecolor(phase_colors[phase])
        for median in b_plots["medians"]:
            if phase == 1:
                median.set_color("white")
            else:
                median.set_color("black")
        tls[phase].set_color(phase_colors[phase])
        tls[phase].set_fontweight("bold")

    # Axis cosmetics
    axs.spines[["right", "top", "bottom"]].set_visible(False)
    axs.set_xticks(xs, helper.PHASES)
    axs.tick_params(axis="x", which="major", pad=-1)
    [label.set_fontweight("bold") for label in tls]
    axs.tick_params(bottom=False)
    axs.set_ylim([y_min, y_lim[1]])
    axs.spines[["left"]].set_bounds(y_lim)
    axs.set_ylabel(y_label)

    # Rule switch cosmetics
    axs.text(
        -0.1,
        y_lim[1] - 0.25 * (y_lim[1] - y_lim[0]),
        rotation=90,
        clip_on=False,
        **helper.FONT_RULE_SWITCH,
    )
    y_rule_switch = [
        y_lim[0] + (y_lim[1] - y_lim[0]) * 0.05,
        y_lim[0] + (y_lim[1] - y_lim[0]) * 0.95,
    ]
    axs.plot([0, 0], y_rule_switch, **helper.STYLE_RULE_SWITCH)

    # Learning phase cosmetics
    cmap = helper.get_performance_cmap(phase="both")
    normalize = m_colors.Normalize(vmin=0, vmax=1)
    scalar_map = ScalarMappable(norm=normalize, cmap=cmap)
    cb = plt.colorbar(
        scalar_map,
        ax=axs,
        orientation="horizontal",
        anchor=(0, 0.0),
        pad=-0.17,
        ticks=[],
        drawedges=False,
        aspect=15,
    )
    cb.outline.set_visible(False)
    cb.ax.zorder = -1
    y = y_lim[0] - (y_lim[1] - y_lim[0]) * 0.04
    naive_expert_style = {"ha": "center", "color": "white", "va": "center"}
    axs.text(xs[0], y, "Naïve", **naive_expert_style)
    axs.text(xs[1], y, "Expert", **naive_expert_style)
    axs.text(xs[2], y, "Naïve", **naive_expert_style)
    axs.text(xs[3], y, "Expert", **naive_expert_style)
    axs.patch.set_visible(False)

    # Significant difference in statistics
    ys = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.97
    yst = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.98
    eps = 0.05
    n_tests = 3
    groups = (
        (helper.PHASES[0], helper.PHASES[1]),
        (helper.PHASES[1], helper.PHASES[2]),
        (helper.PHASES[2], helper.PHASES[3]),
    )
    for p in range(n_tests):
        pv = statistics.welch_t_bonferroni(
            data[p],
            data[p + 1],
            n_tests=n_tests,
            verbose=verbose,
            outcome="the activity",
            group1=groups[p][0],
            group2=groups[p][1],
        )
        axs.plot([xs[p] + eps, xs[p + 1] - eps], [ys, ys], **helper.STYLE_SIGNIFICANT)
        if pv < constants.SIGNIFICANCE_THRESHOLD:
            axs.text(
                x=0.5 * (xs[p] + xs[p + 1]),
                y=ys,
                s=helper.get_significance(pv),
                **helper.FONT_SIGNIFICANT,
            )
        else:
            axs.text(x=0.5 * (xs[p] + xs[p + 1]), y=yst, **helper.FONT_NON_SIGNIFICANT)

    # Annotate stimulus window
    if stim_w:
        box_y0 = 0.609
        box_l = 0.104
        box_w = 0.035
        box_x = 1.01
        b_cols = [helper.COLOR_BASELINE, helper.COLOR_STIMULUS, helper.COLOR_OUTCOME]
        box_durations = [2, 1, 2]
        for b in range(len(b_cols)):
            rect = patches.Rectangle(
                (box_x, box_y0 - sum(box_durations[:b]) * box_l),
                box_w,
                -box_durations[b] * box_l,
                linewidth=0,
                transform=axs.transAxes,
                facecolor=b_cols[b],
                clip_on=False,
            )
            axs.add_patch(rect)
        annotate_x = 0.02 + box_w
        axs.annotate(
            "Stimulus",
            xy=(box_x + annotate_x, box_y0 - 2.5 * box_l),
            xytext=(box_x + annotate_x + 0.033, box_y0 - 2.5 * box_l),
            xycoords="axes fraction",
            transform=axs.transAxes,
            fontsize=helper.FONT_SIZE,
            ha="center",
            rotation=270,
            va="center",
            bbox=dict(boxstyle="square, pad=0.1", fc="white", ec="none"),
            arrowprops=dict(
                arrowstyle="-[, widthB=0.9, lengthB=0.3",
                lw=helper.AXIS_WIDTH,
                color="k",
            ),
        )

    # Finalize formatting
    helper.adjust_figure(fig=fig, wspace=0.0)
    fig.subplots_adjust(left=0.23, bottom=0.1, right=0.9)
    helper.set_panel_label(label=panel, fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name=plot_name)


def panel_c_confidence(
    n_seeds=constants.NR_SEEDS, save: bool = True, loading: bool = True
):
    # Init plots
    nr_reversals, nr_trials_per_task, tot_nr_trials, trials, x_ticks, x_tick_labels = (
        init_panel()
    )
    helper.set_style()
    fig, axs = plt.subplots(
        nrows=2, ncols=1, figsize=(helper.PANEL_WIDTH, helper.PANEL_HEIGHT)
    )
    fig = cast(plt.Figure, fig)

    # Plot
    colors = [helper.COLOR_LE, helper.COLOR_SST, helper.COLOR_VIP]
    labels = ["Confidence", "CPE", "CL"]
    data_key = [C_AV, CPE_AV, CL_AV]
    ax_ids = [0, 0, 1]
    data = get_data(n_seeds=n_seeds, loading=loading)
    lns = [
        axs[ax_ids[i]].plot(
            trials, data[data_key[i]], color=colors[i], label=labels[i], clip_on=False
        )
        for i in range(3)
    ]

    # Set axes and cosmetics
    cmap = helper.get_performance_cmap(phase="both")
    normalize = m_colors.Normalize(vmin=0, vmax=1)
    scalar_map = ScalarMappable(norm=normalize, cmap=cmap)
    cb = plt.colorbar(
        scalar_map,
        ax=axs[0],
        orientation="horizontal",
        location="top",
        ticks=[],
        drawedges=False,
        fraction=0.4,
        pad=0.01,
        aspect=17,
    )
    cb.outline.set_visible(False)
    cb.ax.zorder = -1
    xn, xe, y = 30, 35, 1.08
    naive_expert_style = {"ha": "center", "color": "white"}
    axs[0].text(xn, y, "Naïve", **naive_expert_style)
    axs[0].text(N_TRIALS - xe, y, "Expert", **naive_expert_style)
    axs[0].text(N_TRIALS + xn, y, "Naïve", **naive_expert_style)
    axs[0].text(2 * N_TRIALS - xe, y, "Expert", **naive_expert_style)
    ne_x, ne_y = 29, 1.3
    axs[0].text(
        ne_x, ne_y, helper.PHASES[0], ha="center", color=helper.COLOR_LN, weight="bold"
    )
    axs[0].text(
        N_TRIALS - ne_x,
        ne_y,
        helper.PHASES[1],
        ha="center",
        color=helper.COLOR_LE,
        weight="bold",
    )
    axs[0].text(
        N_TRIALS + ne_x,
        ne_y,
        helper.PHASES[2],
        ha="center",
        color=helper.COLOR_RN,
        weight="bold",
    )
    axs[0].text(
        2 * N_TRIALS - ne_x,
        ne_y,
        helper.PHASES[3],
        ha="center",
        color=helper.COLOR_RE,
        weight="bold",
    )
    axs[0].patch.set_visible(False)
    axs[1].text(
        N_TRIALS * 0.95, 1, rotation=90, clip_on=False, **helper.FONT_RULE_SWITCH
    )
    axs[1].plot(
        [N_TRIALS, N_TRIALS],
        [0, 3.1],
        zorder=0,
        clip_on=False,
        **helper.STYLE_RULE_SWITCH,
    )
    for a in range(2):
        axs[a].set_xlim([0, nr_trials_per_task * nr_reversals])
        axs[a].set_yticks([0.0, 1.0])
        axs[a].spines.right.set_visible(False)
        axs[a].spines.top.set_visible(False)
        axs[a].tick_params(axis="y", which="major", pad=1)

    axs[0].set_xticks([])
    axs[0].spines.bottom.set_visible(False)
    axs[0].set_ylim([0, 1.0])
    axs[1].set_xticks(x_ticks, x_tick_labels)
    axs[1].set_ylim([-0.05, np.max(data[CL_AV]).item() * 1.2])
    axs[1].spines.left.set_bounds([0, 1])
    axs[1].set_xlabel(r"Trial", labelpad=1.7)
    lns = lns[0] + lns[1] + lns[2]
    axs[1].legend(
        handles=lns,
        bbox_to_anchor=(0.0, 0.6, 0.5, 0.5),
        loc="upper left",
        handlelength=1,
        frameon=False,
    )
    rect = patches.Rectangle(
        (-87, -0.4),
        10,
        1.5,
        linewidth=0,
        edgecolor="r",
        facecolor=helper.COLOR_STIMULUS,
        clip_on=False,
    )
    axs[0].add_patch(rect)
    axs[0].text(
        -98, 0.35, s="Stimulus", rotation=90, clip_on=False, ha="center", va="center"
    )
    rect = patches.Rectangle(
        (-87, -0.1),
        10,
        1.5,
        linewidth=0,
        edgecolor="r",
        facecolor=helper.COLOR_OUTCOME,
        clip_on=False,
    )
    axs[1].add_patch(rect)
    axs[1].text(
        -98, 0.7, s="Outcome", rotation=90, clip_on=False, ha="center", va="center"
    )
    axs[0].arrow(
        -70, 1, 0, -2.5, color="k", head_width=5, head_length=0.2, clip_on=False
    )
    axs[0].text(
        -55, -0.25, s="Trial time", rotation=90, clip_on=False, ha="center", va="center"
    )
    axs[1].yaxis.set_label_coords(-0.06, 0.4)

    # Final adjustments and annotations
    helper.adjust_figure(fig=fig)
    fig.subplots_adjust(bottom=0.18, hspace=0.1, left=0.23, top=0.97)
    helper.set_panel_label(label="c", fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name="c_confidence")


def panel_f_cpe(save: bool = True, loading: bool = True, verbose: bool = False):
    nr_trials_per_task = N_TRIALS
    tot_nr_trials = 2 * N_TRIALS
    phase_width = int(nr_trials_per_task / 2)
    phase_t_ranges = [
        [0, phase_width - 1],
        [nr_trials_per_task - phase_width, nr_trials_per_task - 1],
        [nr_trials_per_task, nr_trials_per_task + phase_width - 1],
        [tot_nr_trials - phase_width, tot_nr_trials - 1],
    ]
    data = get_data(n_seeds=constants.NR_SEEDS, loading=loading)
    data = [
        data[CPE][:, phase_t_ranges[p][0] : phase_t_ranges[p][1] + 1].flatten()
        for p in range(4)
    ]
    plot_box_plots(
        data=data,
        panel="f",
        y_label="Context prediction error (CPE)",
        y_lim=(-0.1, 0.4),
        plot_name="f_cpe",
        save=save,
        verbose=verbose,
    )


def plot(save: bool = True):
    panel_c_confidence(save=save)
    panel_f_cpe(save=save)

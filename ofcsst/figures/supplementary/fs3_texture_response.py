import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import cast
from ofcsst.utils import paths, constants
from ofcsst.figures import helper, statistics
from ofcsst.figures.main.f3bc_s1_response import get_path, WITH_OFC


PLOT_DIR = paths.SUPPLEMENTARY_FIG_DIR / "figs3"


def plot(save: bool = True, verbose: bool = False):
    ps = np.array([0, 1])
    n_phases = len(ps)
    n_trial_types = 2

    # Get data
    data = np.load(get_path(sim_type=WITH_OFC))
    y_max = 105
    y_min_s = 60
    y_min = 60
    y_max_s = 100
    ylabel = "Sensory response ($|z|$)"
    trial_type = [r"$\mathregular{s_1}$", r"$\mathregular{s_2}$"]
    panel = "b"
    plot_name = "b_learning_z_responses"

    # Init plot
    helper.set_style()
    fig, axs = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(0.5 * helper.PANEL_WIDTH, 0.8 * helper.PANEL_HEIGHT),
        squeeze=False,
    )
    fig = cast(plt.Figure, fig)
    axs = axs[0, 0]
    phase_colors = [helper.COLOR_LN, helper.COLOR_LE]
    x_ticks = 3 * ps + 0.5
    b_plots = [{}, {}]
    box_width = 0.85
    bp_props = {
        "vert": True,
        "widths": box_width,
        "patch_artist": True,
        "showfliers": False,
        "boxprops": dict(linewidth=helper.AXIS_WIDTH),
        "capprops": dict(linewidth=helper.AXIS_WIDTH),
        "whiskerprops": dict(linewidth=helper.AXIS_WIDTH, solid_capstyle="butt"),
        "flierprops": dict(
            marker="o", markersize=helper.MARKER_SIZE, linewidth=helper.AXIS_WIDTH
        ),
        "medianprops": dict(
            color="k", solid_capstyle="butt", linewidth=helper.AXIS_WIDTH
        ),
    }
    for tt in range(n_trial_types):
        pos = 3 * ps + tt
        y = [
            data[:, phase, tt][~np.isnan(data[:, phase, tt])]
            for phase in range(n_phases)
        ]
        b_plots[tt] = axs.boxplot(x=y, positions=pos, **bp_props)
        for median, patch, color in zip(
            b_plots[tt]["medians"], b_plots[tt]["boxes"], phase_colors
        ):
            if tt == 0:  # Stimulus 1
                patch.set(linewidth=0)
                patch.set_facecolor(color)
                if color == phase_colors[1]:
                    median.set_color("white")
            else:  # Stimulus 2
                patch.set(color=color)
                patch.set_facecolor("white")
    axs.set_ylim([y_min, y_max])
    axs.set_xticks(x_ticks, [helper.PHASES[p] for p in ps])
    axs.set_xlim([-1, 5])
    axs.spines[["right", "top", "bottom"]].set_visible(False)
    axs.spines.left.set_bounds([y_min_s, y_max_s])
    axs.tick_params(bottom=False)
    tls = axs.get_xticklabels()
    for p in ps:
        tls[p].set_color(phase_colors[p])
        tls[p].set_fontweight("bold")
    axs.set_ylabel(ylabel)
    hit_patch = patches.Patch(color="black", label=trial_type[0])
    cr_patch = patches.Patch(facecolor="white", ec="black", label=trial_type[1])
    axs.legend(
        loc="lower right",
        handles=[hit_patch, cr_patch],
        handlelength=box_width,
        frameon=False,
        bbox_to_anchor=(1, 1.0),
        borderpad=0,
    )

    # Significant difference statistics
    x = 3 * ps
    ymin, y_max = axs.get_ylim()
    y_sig = ymin + 0.9 * (y_max - ymin)
    y_sig_ns = ymin + 0.92 * (y_max - ymin)
    pn = ["naïve", "expert"]
    for p in ps:
        d1 = data[:, p, 0]
        d2 = data[:, p, 1]
        pv = statistics.welch_t_bonferroni(
            d1[~np.isnan(d1)],
            d2[~np.isnan(d2)],
            n_tests=n_phases,
            outcome=f"{pn[p]} sensory response",
            group1=r"$\stim_1$",
            group2=r"$\stim_2$",
            verbose=verbose,
        )
        axs.plot([x[p], x[p] + 1], [y_sig, y_sig], **helper.STYLE_SIGNIFICANT)
        if pv < constants.SIGNIFICANCE_THRESHOLD:
            axs.text(
                x=x[p] + 0.5,
                y=0.99 * y_sig_ns,
                s=helper.get_significance(pv),
                **helper.FONT_SIGNIFICANT,
            )
        else:
            axs.text(x=x[p] + 0.5, y=y_sig_ns, **helper.FONT_NON_SIGNIFICANT)

    # Finalize formatting
    helper.adjust_figure(fig=fig, wspace=0.3)
    fig.subplots_adjust(top=0.7)
    helper.set_panel_label(label=panel, fig=fig)

    # Save or display
    helper.save_or_show(save=save, fig=fig, plot_dir=PLOT_DIR, plot_name=plot_name)

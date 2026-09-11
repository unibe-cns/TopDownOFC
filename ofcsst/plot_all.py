from copy import copy
from pypdf import PdfWriter, PdfReader, PageObject, Transformation
from ofcsst.utils import paths
from ofcsst.figures import main, supplementary


def add_page(
    panels: list[str],
    pages: list[int],
    merged_page: PageObject,
    writer: PdfWriter,
    xs: list[float],
    ys: list[float],
) -> None:
    """Assemble the panels into a figure and add it to the final pdf."""

    for path, page_idx, x, y in zip(panels, pages, xs, ys):
        page = copy(PdfReader(path).pages[page_idx])
        transformation = Transformation().translate(tx=x, ty=y)
        merged_page.merge_transformed_page(page, transformation)

    out_page = writer.add_page(merged_page)
    out_page.compress_content_streams()


def assemble_figures():
    """Assemble all figure panels into a single pdf containing all the arranged figures."""

    print("Assembling figures...", end="")

    output_path = str(paths.PLOT_DIR / "figures.pdf")
    writer = PdfWriter()
    pt = 28.3464566929
    width = 17.8 * pt
    panel_height = 4.5 * pt
    schematic_pdf = str(paths.PLOT_DIR / "schematics.pdf")

    # Figure 1
    height = 3 * panel_height
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [
        0.0,
        width / 5.5,
        width / 3.0,
        width * 2.0 / 3.0,
        width / 3.0,
        width * 2.0 / 3.0,
    ]
    ys = [0.0, panel_height * 1.3, panel_height, panel_height, 0.0, 0.0]
    pages = [0, 0, 0, 0, 0, 0]
    pdf_names = [
        "d_gain_inset.pdf",
        "e_distractor_landscape.pdf",
        "f_vary_snr.pdf",
        "h_distractor_dependence.pdf",
        "i_relative_gain.pdf",
    ]
    panels = [str(paths.MAIN_FIG_DIR / "fig1" / p) for p in pdf_names]
    panels.insert(0, schematic_pdf)
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure 2
    height = 2 * panel_height
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [width / 3.0, width * 2.0 / 3.0, width / 3.0, width * 2.0 / 3.0, 0.0]
    ys = [panel_height, panel_height, 0.0, 0.0, 0.0]
    pages = [0, 0, 0, 0, 1]
    pdf_names = ["b_var_pred.pdf", "c_confidence.pdf", "e_ofc_xp.pdf", "f_cpe.pdf"]
    panels = [str(paths.MAIN_FIG_DIR / "fig2" / p) for p in pdf_names]
    panels.insert(4, schematic_pdf)
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure 3
    height = 1.8 * panel_height
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [
        0.8 * width / 3,
        1.9 * width / 3,
        0.8 * width / 3.0,
        1.3 * width / 3.0,
        2.1 * width / 3.0,
        0.0,
    ]
    ys = [panel_height, panel_height, 0.0, 0.0, 0.0, 0.0]
    pages = [0, 0, 0, 0, 0, 2]
    pdf_names = [
        "b_s1_response.pdf",
        "c_z_response.pdf",
        "e_performance_bars.pdf",
        "f_expertise_cumulative_distribution.pdf",
        "g_expert_time.pdf",
    ]
    panels = [str(paths.MAIN_FIG_DIR / "fig3" / p) for p in pdf_names]
    panels.insert(5, schematic_pdf)
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure 4
    height = 10.435 * pt
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [0.0, 0.0, width / 3.0, 2.0 * width / 3.0, 0.0]
    ys = [1.35 * panel_height, 0.0, 10.435 * pt - 0.6 * panel_height, 0.0, 0.0]
    pages = [0, 0, 0, 0, 3]
    pdf_names = [
        "b_mice_perf.pdf",
        "c_traces.pdf",
        "d_expertise.pdf",
        "g_expert_times.pdf",
    ]
    panels = [str(paths.MAIN_FIG_DIR / "fig4" / p) for p in pdf_names]
    panels.insert(5, schematic_pdf)
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure 5
    height = 7.85 * pt
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [0.0, 0.0]
    ys = [2.45 * pt, 0.0]
    pages = [0, 4]
    pdf_names = ["a_performance_vip_cl.pdf"]
    panels = [str(paths.MAIN_FIG_DIR / "fig5" / p) for p in pdf_names]
    panels.insert(2, schematic_pdf)
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure S1
    height = panel_height
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [0, width / 3.0, width * 2.0 / 3.0]
    ys = [0.0, 0.0, 0.0]
    pages = [0, 0, 0]
    pdf_names = [
        "a_distractor_landscape.pdf",
        "b_vary_snr.pdf",
        "c_distractor_dependence.pdf",
    ]
    panels = [str(paths.SUPPLEMENTARY_FIG_DIR / "figs1" / p) for p in pdf_names]
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure S2
    height = 2 * panel_height
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [0, width / 3.0, width * 2.0 / 3.0, width / 3.0]
    ys = [panel_height, panel_height, panel_height, 0.0]
    pages = [0, 0, 0, 0]
    pdf_names = ["a.pdf", "b.pdf", "c.pdf", "d_prob_opt.pdf"]
    panels = [str(paths.SUPPLEMENTARY_FIG_DIR / "figs2" / p) for p in pdf_names]
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure S3
    height = 0.8 * panel_height
    merged_page = PageObject.create_blank_page(None, width / 3.0, height)
    xs = [0.0, width / 6.0, 0.0]
    ys = [0.0, 0.0, 0.0]
    pages = [0, 0, 5]
    pdf_names = ["a_learning_s1.pdf", "b_learning_z_responses.pdf"]
    panels = [str(paths.SUPPLEMENTARY_FIG_DIR / "figs3" / p) for p in pdf_names]
    panels.insert(2, schematic_pdf)
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure S4
    height = panel_height
    merged_page = PageObject.create_blank_page(None, width / 3.0, height)
    xs = [0.0]
    ys = [0.0]
    pages = [0]
    pdf_names = ["_distractor_dependence.pdf"]
    panels = [str(paths.SUPPLEMENTARY_FIG_DIR / "figs4" / p) for p in pdf_names]
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure S5
    height = panel_height
    merged_page = PageObject.create_blank_page(None, width * 2.0 / 3.0, height)
    xs = [0, width / 3.0]
    ys = [0.0, 0.0]
    pages = [0, 0]
    pdf_names = ["a_snr_2d.pdf", "b_snr_1d.pdf"]
    panels = [str(paths.SUPPLEMENTARY_FIG_DIR / "figs5" / p) for p in pdf_names]
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure S6
    height = 2 * panel_height
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [0.0, width / 3.0, 2 * width / 3.0, 0.0, width / 3.0, 2 * width / 3.0]
    ys = [panel_height, panel_height, panel_height, 0.0, 0.0, 0.0]
    pages = [0, 0, 0, 0, 0, 0]
    pdf_names = [
        "a_schematic.pdf",
        "b_expert_time_pyramid.pdf",
        "c_time_to_expert.pdf",
        "d_distractor_landscape.pdf",
        "e_vary_snr.pdf",
        "f_distractor_dependence.pdf",
    ]
    panels = [str(paths.SUPPLEMENTARY_FIG_DIR / "figs6" / p) for p in pdf_names]
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure S7
    merged_page = PageObject.create_blank_page(None, width, panel_height)
    xs = [0.0, width / 3.0]
    ys = [0.0, 0.0]
    pages = [0, 0]
    pdf_names = ["a_schematic.pdf", "b_perf_traces.pdf"]
    panels = [str(paths.SUPPLEMENTARY_FIG_DIR / "figs7" / p) for p in pdf_names]
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure S8
    merged_page = PageObject.create_blank_page(None, width, panel_height)
    xs = [0.0, width / 6.0, 1.3 / 3.0 * width]
    ys = [0.0, 0.0, 0.0]
    pages = [0, 0, 0]
    pdf_names = [
        "a_performance_bars.pdf",
        "b_expertise_cumulative_distribution.pdf",
        "c_expert_time.pdf",
    ]
    panels = [str(paths.SUPPLEMENTARY_FIG_DIR / "figs8" / p) for p in pdf_names]
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure S9
    merged_page = PageObject.create_blank_page(None, width, panel_height)
    xs = [0.0, width / 2.0]
    ys = [0.0, 0.0]
    pages = [0, 0]
    pdf_names = ["a_d_traces.pdf", "b_kaplan_meier.pdf"]
    panels = [str(paths.SUPPLEMENTARY_FIG_DIR / "figs9" / p) for p in pdf_names]
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure S10
    merged_page = PageObject.create_blank_page(None, width * 0.6, panel_height)
    xs = [0.0, width / 3]
    ys = [0.0, panel_height / 2]
    pages = [0, 0]
    pdf_names = ["a_pi_weight.pdf", "b_expert_time.pdf"]
    panels = [str(paths.SUPPLEMENTARY_FIG_DIR / "figs10" / p) for p in pdf_names]
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure S11
    merged_page = PageObject.create_blank_page(None, width, 1.5 * panel_height)
    xs = [0.0, 2 * width / 3]
    ys = [0.0, 0]
    pages = [0, 0]
    pdf_names = ["a_traces.pdf", "b_ofc_bars.pdf"]
    panels = [str(paths.SUPPLEMENTARY_FIG_DIR / "figs11" / p) for p in pdf_names]
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Figure S12
    merged_page = PageObject.create_blank_page(None, width, 2.2 * panel_height)
    xs = [0.0, 1.3 * width / 3, 2.15 * width / 3, 0.0, 0.5 * width, 0.0]
    ys = [panel_height, panel_height, panel_height, 0.0, 0.0, 0.0]
    pages = [0, 0, 0, 0, 0, 6]
    pdf_names = [
        "a_performance_vip_cl.pdf",
        "b_slope_stats.pdf",
        "c_intercept_stats.pdf",
        "d_kaplan_meier.pdf",
        "e_kaplan_meier.pdf",
    ]
    panels = [str(paths.SUPPLEMENTARY_FIG_DIR / "figs12" / p) for p in pdf_names]
    panels.insert(5, schematic_pdf)
    add_page(panels, pages, merged_page, writer, xs, ys)

    # Save everything
    with open(output_path, "wb") as fileobj:
        writer.write(fileobj)

    print("\rComplete figures assembled successfully!")


def plot():
    """Plot and save all figure panels and assemble them into complete figures within a single pdf."""

    # Plot main figure panels
    main_panels = [
        "f1d_gain_inset",
        "f1e_explore_distractor",
        "f1f_explore_gain",
        "f1h_gain_vs_no_gain",
        "f1i_relative_gain",
        "f2b_predicted_variance",
        "f2cef_reversal",
        "f3bc_s1_response",
        "f3efg_ofc_sst",
        "f4bcd_double_reversal",
        "f4g_multicontext",
        "f5a_vip_cl",
    ]
    n_panels = len(main_panels)
    for f in range(n_panels):
        print(f"\rPlotting main panels {100 * f / n_panels:0.1f}% completed", end="")
        getattr(main, main_panels[f]).plot()
    print("\rPlotting main panels is now complete!")

    # Plot supplementary figure panels
    sup_panels = [
        "fs1_q_policy",
        "fs2d_paopt",
        "fs3_texture_response",
        "fs4_control_agents",
        "fs5_snr",
        "fs6bc",
        "fs6def",
        "fs7_reset",
        "fs8_optimization",
        "fs9_ofc_inhibited_perf",
        "fs10_multi_rev_no_gain",
        "fs11_multi_rev_ofc",
        "fs12_vip_cl_stats",
    ]
    n_panels = len(sup_panels)
    for f in range(n_panels):
        print(
            f"\rPlotting supplementary panels {100 * f / n_panels:0.1f}% completed",
            end="",
        )
        getattr(supplementary, sup_panels[f]).plot()
    print("\rPlotting supplementary panels is now complete!")

    # Assemble panels into complete figures
    assemble_figures()


def stats():
    """Run all statistical tests and print the results."""
    print("Figure 2f")
    main.f2cef_reversal.panel_f_cpe(save=True, verbose=True, loading=True)
    print("Figure 3c")
    main.f3bc_s1_response.plot_responses(save=True, verbose=True)
    print("Figure 3e")
    main.f3efg_ofc_sst.panel_e_performance_bars(save=True, verbose=True)
    print("Figure 4d")
    main.f4bcd_double_reversal.panel_d_expert_times(save=True, verbose=True)
    print("Figure 4g")
    main.f4g_multicontext.panel_g_expert_times(save=True, verbose=True)
    print("Figure S3b")
    supplementary.fs3_texture_response.plot(save=True, verbose=True)
    print("Figure S8a")
    supplementary.fs8_optimization.panel_a_performance_bars(save=True, verbose=True)
    print("Figure S9b")
    supplementary.fs9_ofc_inhibited_perf.panel_b_kaplan_meier(save=True, verbose=True)
    print("Figure S10b")
    supplementary.fs10_multi_rev_no_gain.panel_b_expert_times(save=True, verbose=True)
    print("Figure S12bc")
    supplementary.fs12_vip_cl_stats.panel_bc_stats(save=True, verbose=True)
    print("Figure S12de")
    supplementary.fs12_vip_cl_stats.panel_de_kaplan_meier(save=True, verbose=True)

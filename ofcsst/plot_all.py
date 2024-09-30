import ofcsst
from pypdf import PdfWriter, PdfReader, PageObject, Transformation


def assemble_figures():
    print('Assembling complete figures...', end='')

    output_path = str(ofcsst.utils.paths.PLOT_DIR / 'figures.pdf')
    writer = PdfWriter(output_path)
    pt = 28.3464566929
    width = 17.8 * pt
    panel_height = 4.5 * pt
    schematic_pdf = str(ofcsst.utils.paths.PLOT_DIR / 'schematics.pdf')

    # Figure 1
    height = 3 * panel_height
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [0., width / 3., width * 2. / 3., width / 3., width * 2. / 3.]
    ys = [0., panel_height, panel_height, 0., 0.]
    pages = [0, 0, 0, 0, 0]
    pdf_names = ['e_distractor_landscape.pdf', 'f_vary_snr.pdf', 'h_distractor_dependence.pdf', 'i_relative_gain.pdf']
    panels = [str(ofcsst.utils.paths.MAIN_FIG_DIR / 'fig1' / p) for p in pdf_names]
    panels.insert(0, schematic_pdf)
    for p in range(len(panels)):
        panel = PdfReader(str(panels[p])).pages[pages[p]]
        transformation = Transformation().translate(tx=xs[p], ty=ys[p])
        panel.add_transformation(transformation, expand=True)
        merged_page.merge_page(panel)
    writer.add_page(merged_page)

    # Figure 2
    height = 2 * panel_height
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [width / 3., width * 2. / 3., width / 3., width * 2. / 3., 0.]
    ys = [panel_height, panel_height, 0., 0., 0.]
    pages = [0, 0, 0, 0, 1]
    pdf_names = ['b_var_pred.pdf', 'c_confidence.pdf', 'e_ofc_xp.pdf', 'f_cpe.pdf']
    panels = [str(ofcsst.utils.paths.MAIN_FIG_DIR / 'fig2' / p) for p in pdf_names]
    panels.insert(4, schematic_pdf)
    for p in range(len(panels)):
        panel = PdfReader(str(panels[p])).pages[pages[p]]
        transformation = Transformation().translate(tx=xs[p], ty=ys[p])
        panel.add_transformation(transformation, expand=True)
        merged_page.merge_page(panel)
    writer.add_page(merged_page)

    # Figure 3
    height = 1.8 * panel_height
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [0.8 * width / 3, 1.9 * width / 3, 0.8 * width / 3., 1.3 * width / 3., 2.1 * width / 3., 0.]
    ys = [panel_height, panel_height, 0., 0., 0., 0.]
    pages = [0, 0, 0, 0, 0, 2]
    pdf_names = ['b_s1_response.pdf', 'c_z_response.pdf', 'e_performance_bars.pdf',
                 'f_expertise_cumulative_distribution.pdf', 'g_expert_time.pdf']
    panels = [str(ofcsst.utils.paths.MAIN_FIG_DIR / 'fig3' / p) for p in pdf_names]
    panels.insert(5, schematic_pdf)
    for p in range(len(panels)):
        panel = PdfReader(str(panels[p])).pages[pages[p]]
        transformation = Transformation().translate(tx=xs[p], ty=ys[p])
        panel.add_transformation(transformation, expand=True)
        merged_page.merge_page(panel)
    writer.add_page(merged_page)

    # Figure 4
    height = 10.435 * pt
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [0., 0., width / 3., width / 3., 2. * width / 3., 0.]
    ys = [1.35 * panel_height, 0., 10.435 * pt - 0.6 * panel_height, 10.435 * pt - 0.6 * panel_height, 0., 0.]
    pages = [0, 0, 0, 0, 0, 3]
    pdf_names = ['b_mice_perf.pdf', 'c_traces.pdf', 'd_expertise.pdf', 'd_xp_data.pdf', 'g_expert_times.pdf']
    panels = [str(ofcsst.utils.paths.MAIN_FIG_DIR / 'fig4' / p) for p in pdf_names]
    panels.insert(5, schematic_pdf)
    for p in range(len(panels)):
        panel = PdfReader(str(panels[p])).pages[pages[p]]
        transformation = Transformation().translate(tx=xs[p], ty=ys[p])
        panel.add_transformation(transformation, expand=True)
        merged_page.merge_page(panel)
    writer.add_page(merged_page)

    # Figure 5
    height = 7.85 * pt
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [0., 0., 0.]
    ys = [2.45 * pt, 2.45 * pt, 0.]
    pages = [0, 0, 4]
    pdf_names = ['a_performance_vip_cl.pdf', 'a_xp_data.pdf']
    panels = [str(ofcsst.utils.paths.MAIN_FIG_DIR / 'fig5' / p) for p in pdf_names]
    panels.insert(2, schematic_pdf)
    for p in range(len(panels)):
        panel = PdfReader(str(panels[p])).pages[pages[p]]
        transformation = Transformation().translate(tx=xs[p], ty=ys[p])
        panel.add_transformation(transformation, expand=True)
        merged_page.merge_page(panel)
    writer.add_page(merged_page)

    # Figure S1
    height = 2 * panel_height
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [0, width / 3., width * 2. / 3., width / 3.]
    ys = [panel_height, panel_height, panel_height, 0.]
    pages = [0, 0, 0, 0, 0, 0]
    pdf_names = ['a.pdf', 'b.pdf', 'c.pdf', 'd_prob_opt.pdf']
    panels = [str(ofcsst.utils.paths.SUPPLEMENTARY_FIG_DIR / 'figs1' / p) for p in pdf_names]
    for p in range(len(panels)):
        panel = PdfReader(str(panels[p])).pages[pages[p]]
        transformation = Transformation().translate(tx=xs[p], ty=ys[p])
        panel.add_transformation(transformation, expand=True)
        merged_page.merge_page(panel)
    writer.add_page(merged_page)

    # Figure S2
    height = 0.8 * panel_height
    merged_page = PageObject.create_blank_page(None, width / 3., height)
    xs = [0., width / 6., 0.]
    ys = [0., 0., 0.]
    pages = [0, 0, 5]
    pdf_names = ['a_learning_s1.pdf', 'b_learning_z_responses.pdf']
    panels = [str(ofcsst.utils.paths.SUPPLEMENTARY_FIG_DIR / 'figs2' / p) for p in pdf_names]
    panels.insert(2, schematic_pdf)
    for p in range(len(panels)):
        panel = PdfReader(str(panels[p])).pages[pages[p]]
        transformation = Transformation().translate(tx=xs[p], ty=ys[p])
        panel.add_transformation(transformation, expand=True)
        merged_page.merge_page(panel)
    writer.add_page(merged_page)

    # Figure S3
    height = panel_height
    merged_page = PageObject.create_blank_page(None, width * 2. / 3., height)
    xs = [0, width / 3.]
    ys = [0., 0.]
    pages = [0, 0, 0, 0, 0, 0]
    pdf_names = ['a_snr_2d.pdf', 'b_snr_1d.pdf']
    panels = [str(ofcsst.utils.paths.SUPPLEMENTARY_FIG_DIR / 'figs3' / p) for p in pdf_names]
    for p in range(len(panels)):
        panel = PdfReader(str(panels[p])).pages[pages[p]]
        transformation = Transformation().translate(tx=xs[p], ty=ys[p])
        panel.add_transformation(transformation, expand=True)
        merged_page.merge_page(panel)
    writer.add_page(merged_page)

    # Figure S4
    height = 2 * panel_height
    merged_page = PageObject.create_blank_page(None, width, height)
    xs = [0., width / 3., 2 * width / 3., 0., width / 3., 2 * width / 3.]
    ys = [panel_height, panel_height, panel_height, 0., 0., 0.]
    pages = [0, 0, 0, 0, 0, 0]
    pdf_names = ['a_schematic.pdf', 'b_expert_time_pyramid.pdf', 'c_time_to_expert.pdf', 'd_distractor_landscape.pdf',
                 'e_vary_snr.pdf', 'f_distractor_dependence.pdf']
    panels = [str(ofcsst.utils.paths.SUPPLEMENTARY_FIG_DIR / 'figs4' / p) for p in pdf_names]
    for p in range(len(panels)):
        panel = PdfReader(str(panels[p])).pages[pages[p]]
        transformation = Transformation().translate(tx=xs[p], ty=ys[p])
        panel.add_transformation(transformation, expand=True)
        merged_page.merge_page(panel)
    writer.add_page(merged_page)

    # Figure S5
    merged_page = PageObject.create_blank_page(None, width, panel_height)
    xs = [0., width / 3.]
    ys = [0., 0.]
    pages = [0, 0]
    pdf_names = ['a_schematic.pdf', 'b_perf_traces.pdf']
    panels = [str(ofcsst.utils.paths.SUPPLEMENTARY_FIG_DIR / 'figs5' / p) for p in pdf_names]
    for p in range(len(panels)):
        panel = PdfReader(str(panels[p])).pages[pages[p]]
        transformation = Transformation().translate(tx=xs[p], ty=ys[p])
        panel.add_transformation(transformation, expand=True)
        merged_page.merge_page(panel)
    writer.add_page(merged_page)

    # Figure S6
    merged_page = PageObject.create_blank_page(None, width, panel_height)
    xs = [0., width / 6., 1.3 / 3. * width]
    ys = [0., 0., 0.]
    pages = [0, 0, 0]
    pdf_names = ['a_performance_bars.pdf', 'b_expertise_cumulative_distribution.pdf', 'c_expert_time.pdf']
    panels = [str(ofcsst.utils.paths.SUPPLEMENTARY_FIG_DIR / 'figs6' / p) for p in pdf_names]
    for p in range(len(panels)):
        panel = PdfReader(str(panels[p])).pages[pages[p]]
        transformation = Transformation().translate(tx=xs[p], ty=ys[p])
        panel.add_transformation(transformation, expand=True)
        merged_page.merge_page(panel)
    writer.add_page(merged_page)

    # Figure S7
    merged_page = PageObject.create_blank_page(None, width * 0.6, panel_height)
    xs = [0., width / 3]
    ys = [0., panel_height / 2]
    pages = [0, 0]
    pdf_names = ['a_pi_weight.pdf', 'b_expert_time.pdf']
    panels = [str(ofcsst.utils.paths.SUPPLEMENTARY_FIG_DIR / 'figs7' / p) for p in pdf_names]
    for p in range(len(panels)):
        panel = PdfReader(str(panels[p])).pages[pages[p]]
        transformation = Transformation().translate(tx=xs[p], ty=ys[p])
        panel.add_transformation(transformation, expand=True)
        merged_page.merge_page(panel)
    writer.add_page(merged_page)

    # Figure S8
    merged_page = PageObject.create_blank_page(None, width, 1.5 * panel_height)
    xs = [0., 2 * width / 3]
    ys = [0., 0]
    pages = [0, 0]
    pdf_names = ['a_traces.pdf', 'b_ofc_bars.pdf']
    panels = [str(ofcsst.utils.paths.SUPPLEMENTARY_FIG_DIR / 'figs8' / p) for p in pdf_names]
    for p in range(len(panels)):
        panel = PdfReader(str(panels[p])).pages[pages[p]]
        transformation = Transformation().translate(tx=xs[p], ty=ys[p])
        panel.add_transformation(transformation, expand=True)
        merged_page.merge_page(panel)
    writer.add_page(merged_page)

    # Save everything
    with open(output_path, 'wb') as fileobj:
        writer.write(fileobj)

    print('\rComplete figures assembled successfully!')


def plot():
    # Plot main figure panels
    main_panels = ['f1e_explore_distractor', 'f1f_explore_gain', 'f1h_gain_vs_no_gain', 'f1i_relative_gain',
                   'f2b_predicted_variance', 'f2cef_reversal', 'f3b_s1_xp', 'f3efg_ofc_sst', 'f4bcd_double_reversal',
                   'f4g_multicontext', 'f5a_VIP_CL']
    n_panels = len(main_panels)
    for f in range(n_panels):
        print(f"\rPlotting main panels {100 * f / n_panels:0.1f}% completed", end="")
        getattr(ofcsst.figures.main, main_panels[f]).plot()
    print(f"\rPlotting main panels is now complete!")

    # Plot supplementary figure panels
    sup_panels = ['fs1d_paopt', 'fs2_texture_response', 'fs3_snr', 'fs4bc', 'fs4def', 'fs5_reset', 'fs6_optimization',
                  'fs7_multi_rev_no_gain', 'fs8_multi_rev_ofc']
    n_panels = len(sup_panels)
    for f in range(n_panels):
        print(f"\rPlotting supplementary panels {100 * f / n_panels:0.1f}% completed", end="")
        getattr(ofcsst.figures.supplementary, sup_panels[f]).plot()
    print(f"\rPlotting supplementary panels is now complete!")

    # Assemble panels into complete figures
    assemble_figures()

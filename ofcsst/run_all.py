import ofcsst


def run():

    # Run simulations for main figures
    main_sims = ['f1i_relative_gain', 'f2b_predicted_variance', 'f2cef_reversal', 'f3b_s1_xp', 'f3efg_ofc_sst',
                 'f4bcd_double_reversal', 'f4g_multicontext', 'f5a_VIP_CL']
    n_sims = len(main_sims)
    for s in range(n_sims):
        print(f"\rRunning main simulations {s} / {n_sims} completed: ", end="")
        getattr(ofcsst.figures.main, main_sims[s]).run()
    print(f"\rMain simulations complete!")

    # Run simulations for supplementary figures
    sup_sims = ['fs4bc', 'fs5_reset', 'fs6_optimization', 'fs7_multi_rev_no_gain']
    n_sims = len(sup_sims)
    for s in range(n_sims):
        print(f"\rRunning supplementary simulations {s} / {n_sims} completed: ", end="")
        getattr(ofcsst.figures.supplementary, sup_sims[s]).run()
    print(f"\rSupplementary simulations complete!")

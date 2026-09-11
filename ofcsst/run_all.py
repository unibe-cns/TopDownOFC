from ofcsst.figures import main, supplementary


def run():
    """Run all simulations based on the optimized hyperparameters."""

    # Run simulations for main figures
    main_sims = (
        "f1i_relative_gain",
        "f2b_predicted_variance",
        "f2cef_reversal",
        "f3bc_s1_response",
        "f3efg_ofc_sst",
        "f4bcd_double_reversal",
        "f4g_multicontext",
        "f5a_vip_cl",
    )
    n_sims = len(main_sims)
    for s in range(n_sims):
        print(f"\rRunning main simulations {s} / {n_sims} completed: ", end="")
        getattr(main, main_sims[s]).run()
    print("\rMain simulations complete!")

    # Run simulations for supplementary figures
    sup_sims = (
        "fs6bc",
        "fs7_reset",
        "fs8_optimization",
        "fs10_multi_rev_no_gain",
    )
    n_sims = len(sup_sims)
    for s in range(n_sims):
        print(f"\rRunning supplementary simulations {s} / {n_sims} completed: ", end="")
        getattr(supplementary, sup_sims[s]).run()
    print("\rSupplementary simulations complete!")

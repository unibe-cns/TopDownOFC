from ofcsst.figures import main, supplementary


def scan():
    """Run all hyperparameter optimization searches."""

    # Run parameter scans for main figures
    main_sims = ["f1e_explore_distractor", "f1h_gain_vs_no_gain", "f3efg_ofc_sst"]
    n_sims = len(main_sims)
    for s in range(n_sims):
        print(f"\rRunning main scans {s} / {n_sims} completed: ", end="")
        getattr(main, main_sims[s]).scan()
    print("\rMain scans complete!")

    # Run parameter scans for supplementary figures
    sup_sims = [
        "fs1_q_policy",
        "fs4_control_agents",
        "fs6bc",
        "fs6def",
        "fs8_optimization",
    ]
    n_sims = len(sup_sims)
    for s in range(n_sims):
        print(f"\rRunning supplementary scans {s} / {n_sims} completed: ", end="")
        getattr(supplementary, sup_sims[s]).scan()
    print("\rSupplementary scans complete!")

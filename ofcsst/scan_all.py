import ofcsst


def scan():

    # Run parameter scans for main figures
    main_sims = ['f1e_explore_distractor', 'f1h_gain_vs_no_gain', 'f3efg_ofc_sst']
    n_sims = len(main_sims)
    for s in range(n_sims):
        print(f"\rRunning main scans {s} / {n_sims} completed: ", end="")
        getattr(ofcsst.figures.main, main_sims[s]).scan()
    print(f"\rMain scans complete!")

    # Run parameter scans for supplementary figures
    sup_sims = ['fs4bc', 'fs4def', 'fs6_optimization']
    n_sims = len(sup_sims)
    for s in range(n_sims):
        print(f"\rRunning supplementary scans {s} / {n_sims} completed: ", end="")
        getattr(ofcsst.figures.supplementary, sup_sims[s]).scan()
    print(f"\rSupplementary scans complete!")

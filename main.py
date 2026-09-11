import sys
import logging
import ofcsst


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)

    if len(sys.argv) == 1:
        ofcsst.plot_all.plot()
    else:
        if str(sys.argv[1]) == "plot":
            ofcsst.plot_all.plot()

        elif str(sys.argv[1]) == "simulate":
            ofcsst.run_all.run()

        elif str(sys.argv[1]) == "scan":
            ofcsst.scan_all.scan()

        elif str(sys.argv[1]) == "stats":
            ofcsst.plot_all.stats()

        # Explain correct usage syntax
        else:
            print(
                "Proper usage requires one of the arguments 'plot', 'simulate', "
                "'scan' or 'stats':\n"
                ">> python main.py plot      # plot all panels and assemble figures.pdf\n"
                ">> python main.py simulate  # rerun all simulations\n"
                ">> python main.py scan      # rerun all hyperparameter scans\n"
                ">> python main.py stats     # recompute the reported statistics"
            )

import sys
import logging
import ofcsst


if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL)

    if len(sys.argv) == 1:
        ofcsst.plot_all.plot()
    else:
        if str(sys.argv[1]) == 'plot':
            ofcsst.plot_all.plot()

        elif str(sys.argv[1]) == 'simulate':
            ofcsst.run_all.run()

        elif str(sys.argv[1]) == 'scan':
            ofcsst.scan_all.scan()

        # Explain correct usage syntax
        else:
            print("Proper usage requires argument 'plot' or 'simulate'. To plot all the panels enter: \n"
                  ">> python main.py plot\n"
                  "\nTo run all the simulations enter:\n"
                  ">> python main.py simulate")

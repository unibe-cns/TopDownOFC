from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
RESULT_DIR = ROOT_DIR / Path('results')
SCAN_SUBDIR = 'scan'
SIMULATION_SUBDIR = 'simulation'
SUPPLEMENTARY_SIMULATION_DIR = RESULT_DIR / SIMULATION_SUBDIR / 'supplementary'
PLOT_DIR = ROOT_DIR / 'figures'
MAIN_FIG_DIR = PLOT_DIR / 'main'
SUPPLEMENTARY_FIG_DIR = PLOT_DIR / 'supplementary'
ICON_DIR = PLOT_DIR / 'icons'
SVG = "svg"
STATIONARY_SUBDIR = "stationary"
SWITCH_SUBDIR = "non-stationary"

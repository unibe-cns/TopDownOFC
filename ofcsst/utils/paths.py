from pathlib import Path

ROOT_DIR = Path(
    __file__
).parent.parent.parent.resolve()  # Root directory of the project
RESULT_DIR = ROOT_DIR / Path("results")  # Directory for results
SCAN_SUBDIR = "scan"  # Subdirectory for scan results
SIMULATION_SUBDIR = "simulation"  # Subdirectory for simulation results
EXPERIMENT_DIR = RESULT_DIR / "experimental"  # Directory for experimental data
SUPPLEMENTARY_SIMULATION_DIR = (
    RESULT_DIR / SIMULATION_SUBDIR / "supplementary"
)  # Directory for supplementary simulation results
PLOT_DIR = ROOT_DIR / "figures"  # Directory for figures
MAIN_FIG_DIR = PLOT_DIR / "main"  # Directory for main figures
SUPPLEMENTARY_FIG_DIR = (
    PLOT_DIR / "supplementary"
)  # Directory for supplementary figures
SVG = "svg"  # Subdirectory for SVG files
STATIONARY_SUBDIR = "stationary"  # Subdirectory for stationary results
SWITCH_SUBDIR = "non-stationary"  # Subdirectory for non-stationary results

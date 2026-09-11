# Hierarchy of prediction errors shapes the learning of context-dependent sensory representations

## Description

This program runs the simulations and plots the figures for the manuscript '[Hierarchy of prediction errors shapes the learning of context-dependent sensory representations](https://www.biorxiv.org/content/10.1101/2024.09.30.615819v1)'. This means simulating models of mice with S1 L2/3 sensory pyramidal neurons. learning a go/no-go sensory discrimination task with possible rule reversals.

### Installation

These are instructions to set up a functioning conda environment (~10 min). If conda is not installed yet, do so following these [instructions](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) (linux recommended). Then make sure conda is up-to-date.
```
conda update conda --all
```

Create the conda environment `ofcsst`
```
conda env create -f environment.yml
```

Activate the conda environment 
```
conda activate ofcsst
```

### Execution

Before rerunning the simulations, you can test whether the installation was successful (circa 1-2 min run time). Just enter
```
python main.py plot
```

To recompute the statistical tests and output the details, run
```
python main.py stats
```

This will plot all the simulation panels (in the folder `figures`) and assemble all the panels into the full figures (`figures/figures.pdf`). To rerun all the simulations (circa 1 hour run time), enter
```
python main.py simulate
```

This will overwrite the current simulation outcomes in the `results/simulation` folder, which are the files used for the plots. These simulations use the parameter values optimized from parameter scans. These optimized values are in database files in the folder `results/scan`. These scan database files mostly only contain the final optimized parameters. The full scans can be reproduced by running the command:
```
python main.py scan
```

However, be warned! This is computationally extensive and could take a very long time. Also starting them and interrupting the scanning before completion might corrupt the files in `results/scan`.

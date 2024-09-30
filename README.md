# Hierarchy of prediction errors shapes the learning of context-dependent sensory representations

## Description

This program runs the simulations and plots the figures for the manuscript 'Hierarchy of prediction errors shapes the learning of context-dependent sensory representations'. This means simulating models of mice with S1 L2/3 sensory pyramidal neurons. learning a go/no-go sensory discrimination task with possible rule reversals.

### Installation

If conda is not installed yet, do so following these [instructions](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) (linux recommended). Then make sure conda is up-to-date.
```
conda update conda --all
```

Create a conda environment with your favorite name, here `myenv`
```
conda create -n myenv python=3.10.8
```

Activate the conda environment 
```
conda activate myenv
```

Install the required libraries
```
pip install -r requirements.txt
```

Setup the simulation library
```
python setup.py install
```

### Execution

Before rerunning the simulations, you can test whether the installation was successful. Just enter
```
python main.py plot
```

This will plot all the simulation panels (in the folder './figures') and assemble all the panels into the full figures ('./figures/figures.pdf'). To rerun all the simulations, enter
```
python main.py simulate
```

This will overwrite the current simulation outcomes in the './results/simulation' folder, which are the files used for the plots. These simulations use the parameter values optimized from parameter scans. These optimized values are in database files in the folder '.results/scan'. These scan database files mostly only contain the final optimized parameters. The full scans can be reproduced by running the command:
```
python main.py scan
```

However, be warned! This is computationally extensive and could take a very long time. Also starting them and interrupting the scanning before completion might corrupt the files in './results/scan'. We therefore strongly advise to avoid this command unless you know what you are doing. If you mess up and corrupt your files, you can recover them from the github.

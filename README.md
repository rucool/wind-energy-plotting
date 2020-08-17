# Wind Energy Plotting
Tools for plotting wind energy data.

Rutgers University Center for Ocean Observing Leadership


## Installation Instructions
Add the channel conda-forge to your .condarc. You can find out more about conda-forge from their website: https://conda-forge.org/

`conda config --add channels conda-forge`

Clone the wind-energy-plotting repository.

`git clone https://github.com/rucool/wind-energy-plotting.git`

Change your current working directory to the location that you downloaded wind-energy-plotting. 

`cd /Users/lgarzio/Documents/repo/wind-energy-plotting/`

Create conda environment from the included environment.yml file:

`conda env create -f environment.yml`

Once the environment is done building, activate the environment:

`conda activate wind-energy-plotting`

Install the toolbox to the conda environment from the root directory of the wind-energy-plotting toolbox:

`pip install .`

The toolbox should now be installed to your conda environment.
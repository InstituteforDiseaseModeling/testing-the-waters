# testing-the-waters

This repository contains the code used for the King County, WA school reopening analysis presented in the [Testing the waters: time to go back to school?](https://covid.idmod.org/data/Testing_the_waters_time_to_go_back_to_school.pdf) modeling report. This report used the agent-based model Covasim, which can be downloaded from [GitHub](https://github.com/InstituteforDiseaseModeling/covasim) and used for other COVID-19 disease modeling.

* Code to implement schools in Covasim can be found in `covasim_schools`.
* Various tests to ensure functionality are in `tests`.
* Scripts to conduct the analysis are in `testing_the_waters`.


## Installation


### Requirements

Python >=3.6 (64-bit). (Note: Python 2 is not supported, and only Python >=3.8 has been tested.)


### Steps

1. If desired, create a virtual environment.

	- For example, using [conda](https://www.anaconda.com/products/individual):

	  ```
	  conda create -n covaschool python=3.8
	  conda activate covaschool
	  ```

	- Or using [venv](https://docs.python.org/3/library/venv.html):

	  ```
	  python3 -m venv covaschool
	  covaschool\Scripts\activate
	  ```

2. Install [SynthPops](https://github.com/InstituteforDiseaseModeling/synthpops), a package to create synthetic populations, in a folder of your choice. Note that `pip` installation does not currently include required Seattle data files:

   ```
   git clone https://github.com/InstituteforDiseaseModeling/synthpops
   cd synthpops
   python setup.py develop
   ```

3. Install this package, CovasimSchools (which will also install [Covasim](https://covasim.org) and [Optuna](https://optuna.org)):

   ```
   git clone https://github.com/InstituteforDiseaseModeling/testing-the-waters
   cd testing-the-waters
   python setup.py develop
   ```

4. If desired, verify by running the main test script:

   ```
   cd tests
   python test_schools.py
   ```

   If everything installed correctly, this will bring up a plot.


## Usage

Scripts in the `testing_the_waters` folder produce the results presented in the [Testing the waters: time to go back to school?](https://covid.idmod.org/data/Testing_the_waters_time_to_go_back_to_school.pdf) modeling report.  Results come from the following high-level steps:

1. Generate the population files.
1. Calibrate the model.
1. Commission the simulations.
1. Regenerate the figures.

Since the populations, calibrated values, and key results are already included in the repository, each step is optional. Note that the calibration step is non-deterministic, so if the model is recalibrated, slight differences with the published results may be observed. However, as long as calibration is run to completion, the differences should be well within the published uncertainty bounds.


> **Note:** The calibration and commissioning scripts are computationally intensive. They are intended to be run on an HPC of at least 16 cores. Although they may be possible to run on a home computer, it is not recommended and may take several days to complete.  For the results presented in the paper, we used the top 30 parameter configurations identified by the calibration step.  Running 30 configurations for each of many school reopening scenarios and each of many diagnostic screening scenarios results in a large number of simulations.  To learn how the scripts work and run the process end-to-end, you can modify the number of parameter configurations (variable named `par_inds` in the scripts) to run only the one or two best parameter configuration, thereby reducing the number of simulations at the cost of increased noise.  It is also possible to select a subset of the school reopening and diagnostic screening scenarios. The script `run_testing_scenarios.py` has a boolean variable named `test_run` that implements both of these changes.

To regenerate the results that the figures are based on:

1. Acquire the synthetic populations in one of two ways:

   - Under the v20201019/inputs folder, see the `pop` file provided. 

   - Generate the synthetic populations yourself:

     ```
     cd testing_the_waters
     python create_sp_pop.py
     ```

1. Run the following scripts to produce Covasim objects with sims or msim extension that contain the individual simulation results:

	1. Run `run_testing_scenarios.py` to generate the main scenario results.
	1. Run `run_sensitivity_scenarios.py` to generate the sensitivity analyses.
	1. Run `run_countermeasure_scenarios.py` to generate the countermeasure analyses.

Because the resulting output files can be large, and the files can be generated by the above scripts, we have only included the results from the first script in the repository (which are the key results used in the report).


### Reproducing figures

To reproduce the figures from the report, run the following scripts:

1. Fig. 1: `plot_testing.py`
1. Fig. 2: `plot_testing.py`
1. Fig. 3: `pcr_days_sweep.py`
1. Fig. 4: `plot_testing.py`
1. Fig. 5: `plot_sensitivity.py`
1. Fig. 6: `plot_countermeasures.py`
1. Fig. 7: `plot_testing.py`
1. Fig. 8: Generated by a different model (RAINIER); not included in this repository. For more information on the RAINIER methodology, see this [technical report](https://covid.idmod.org/data/Sustained_reductions_in_transmission_have_led_to_declining_COVID_19_prevalence_in_King_County_WA.pdf). 
1. Fig. 9: `run_and_plot_calib.py`
1. Fig. 10: `pcr_vs_ag_delay.py`

For more information, see the documentation in the individual files.

## Disclaimer

The code in this repository was developed by IDM to support our research in disease transmission and managing epidemics. We've made it publicly available under the Creative Commons Attribution-ShareAlike 4.0 International License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as contemplated under the Creative Commons Attribution-ShareAlike 4.0 International License.
# Analysis code for the pRFExpect experiment
This repository contains the analysis code for the pRFExpect experiment.

## Requirements
OS: I ran this script on Rocky Linux 8.10 (Green Obsidian) and macOS 15 (Apple Silicon and Intel). Other OS should be also fine.
Python: 3.9–3.13 recommended. It takes a few minutes to install all the enviorments and this package.

### For fMRI data analysis
- Python 3.9 or higher
- ipykernel
- ipympl
- prfpy: [https://github.com/VU-Cog-Sci/prfpy](https://github.com/VU-Cog-Sci/prfpy)

### For eyemovement data analysis
I'd like to suggest using a separate conda environment for eyemovement data analysis, as it requires some packages that may conflict with the fMRI analysis environment.
- hedpy: [https://github.com/tknapen/hedfpy](https://github.com/tknapen/hedfpy)

## Installation
1. Clone this repository to local folder.
2. Install the required packages using pip.
3. Install this repository as a package using pip.

## Contents
### General files
0. README.md
    - This file.

1. prf_expect/settings.yml
    - This file contains the settings for the analysis, including paths to data directories and parameters for preprocessing and analysis.
    **Before you run any analysis**, please make sure to update the paths in this file to point to your local data directories. Simply open the file in a text editor and modify the path:
    ```yaml
    general:
      data_dir: /path/to/your/parent/folder/of/your/data
    ```

2. prf_expect/utils/*
    - This directory contains utility functions for data input/output, pRF fitting, and visualization.

### preprocessing scripts
1. prf_expect/preproc/functional/highpass.py
    - This script implements a high-pass filter for the fMRI data. Raw data and filtered data are not shared due to data privacy concerns.

2. prf_expect/preproc/functional/psc.py
    - This script computes the percent signal change (PSC) for the fMRI data. PSC data is shared in the dataset.

3. prf_expect/preproc/dms/con_dms.py
    - This script constructs design matrices from screenshots of the experiment (per TR). The screenshots are not shared. The design matrices are shared in the dataset. Because the screen in the scanner was up-side-down, the design matrices are also flipped upside-down in this script.

4. prf_expect/preproc/dms/cut_dms.py
    - In our 7T MRI setup, the small bore resulted in partial occlusion of the display, with approximately one-tenth of the upper part of the screen being blocked. At the beginning of the experiment, participants were asked to report the proportion of the screen they perceived as occluded. Based on these reports, we set the corresponding portion of the design matrix to zero, ensuring that the modeled stimulus representation matched the actually visible display area.

5. prf_expect/preproc/functional/segment_into_subruns.ipynb
    - This notebook contains the code to preprocess the functional data by segmenting it into smaller subruns for easier analysis. For standard pRF runs, it also averages the data across subruns.


### Eyemovement analysis and visualization scripts
1. prf_expect/analysis/eyemovement/eyemovement_figure.ipynb
    - This notebook visualizes the eyemovement data under different conditions.

### pRF fit scripts
1. prf_expect/analysis/prf_fit/pRF_fits.py
    - This script fits the divisive normalization model to the fMRI data using the prfpy package. It takes very long time to run (several hours per subject).

2. prf_expect/analysis/prf_fit/timecourse_predictions.py
    - This script generates timecourse predictions based on the fitted pRF models.

### Calculate TRMI indices scripts
1. prf_expect/analysis/GLMs/calc_TRMI_type*.py
    - These scripts calculate different types of TRMI indices. It only takes a few seconds to run these scripts.

2. prf_expect/analysis/GLMs/calc_TRMI_type1_simulate_TRMI-*.py
    - These scripts simulate data for TRMI indices, then calculate TRMI indices on the simulated data.

### Data analysis and visualization scripts
1. prf_expect/analysis/viz/make_colorbars.ipynb
    - This notebook generates colorbars for visualizations.

2. prf_expect/analysis/viz/vis_TRMI_on_visual_hierarchy.ipynb
    - This notebook visualizes TRMI indices across the visual hierarchy. If you want to run the entire analysis pipeline by yourself, make sure to run vis_TRMI_on_visual_hierarchy.ipynb once first, it will generate the a few .npy files needed in other jupyter notebooks. But anyway these .npy files are also shared in the dataset. Also don't forget to change the glm_analysis_type variable in the code cell to either "TRMI-type1" or "TRMI-type4", depending on which type of TRMI you want to visualize. 

3. prf_expect/analysis/viz/viz_prf_make_tc_figure.ipynb
    - This notebook generates figures to visualize the pRF model fits and timecourse predictions.

4. prf_expect/analysis/viz/ipympl_view_prf_parameters.ipynb
    - This notebook generates interactive visualizations of the pRF parameters on the cortical surface.

5. prf_expect/analysis/viz/viz_violation_make_tc_figure.ipynb
    - This notebook generates figures to visualize violation responses with different TRMI values.

6. prf_expect/analysis/viz/webview_pRF_and_TRMI_results.ipynb
    - This notebook generates webviews to visualize the pRF and TRMI results on the cortical surface. If you want to run the entire analysis pipeline by yourself, make sure to run vis_TRMI_on_visual_hierarchy.ipynb once first, it will generate the .npy files needed in this notebook.

7. prf_expect/analysis/viz/vis_TRMI_barcharts.ipynb
    - This notebook also generates bar charts to visualize TRMI across ROIs. It is an alternative to the bar charts in vis_TRMI_on_visual_hierarchy.ipynb. Basically no difference, but this one is ... hmmm, shorter, and quicker. So I use this figure to generate the bar charts of different TRMI types in the supplementary materials.

8. prf_expect/analysis/viz/vis_TRMI_B_and_D.ipynb
    - This notebook runs a per-subject GLM regressing TRMI on B, D, eccentricity, and pRF size, tests group-level effects, and visualizes the beta weights.

9. prf_expect/analysis/viz/webview_GLM_simulation.ipynb
    - This notebook generates webviews to visualize the TRMI simulation results on the cortical surface.

### Feature-tuned delayed normalization model scripts
1. prf_expect/analysis/feature_tuned_normalization.ipynb
    - This notebook implements the feature-tuned delayed normalization model, and generates the figure we used in the manuscript to illustrate the model.

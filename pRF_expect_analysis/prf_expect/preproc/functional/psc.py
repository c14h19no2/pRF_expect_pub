import numpy as np
import json
import argparse
import glob
from nilearn.glm.first_level.hemodynamic_models import spm_hrf
import os
from pathlib import Path
from prf_expect.utils import io, mri

# Load settings from json file
settings = io.load_settings()

parser = argparse.ArgumentParser(
    description="Script to change the unzscored pybest output to %-signal changed"
)

# Need subject nr., session nr. (by default will expect ses-1) to work out where the pybest data is, then loop over this
parser.add_argument("-s", "--subject", type=str, help="Subject ID")
parser.add_argument("-n", "--session", type=str, default="ses-1", help="Session ID")
# parser.add_argument('-t', '--task', type=str, default='pRF', help='Session ID')
parser.add_argument("-d", "--data", type=str, default="data", help="data or pilot_data")
parser.add_argument(
    "-fs",
    "--space",
    type=str,
    default="fsaverage",
    help="space in which the data resides",
)
args = parser.parse_args()

# Retrieve the arguments and get all necessary variables
subject = args.subject
session = args.session
# task = args.task
d_or_p = args.data
space = args.space

if subject is None:
    raise ValueError("Please provide a subject number")

runs = settings["design"]["runs"]
# print(runs)
tasks = settings["design"]["tasks"]
project_dir = settings["general"]["data_dir"]
runs_per_task = settings["design"]["runs_per_task"]
task_runs = []
for run_p_t in runs_per_task:
    for task in tasks:
        task_runs.append(f"task-{task}_{run_p_t}")

hemis = ["L", "R"]

# Define and make output directory to store psc data in
output_dir = "{}/{}/derivatives/prf_data/{}/{}/psc".format(
    settings["general"]["data_dir"], d_or_p, subject, session
)
os.makedirs(output_dir, exist_ok=True)

for i, run in enumerate(runs):

    # get tsv run files and calculate the baseline indices based off convoluting the stimuli with an hrf
    tsv_file = Path(__file__).parents[2] / "run_list" / f"{subject}_{run}.tsv"

    stimuli = np.loadtxt(tsv_file, delimiter="\t", skiprows=1)
    # Select the first column of the data
    stimulus_col = stimuli[:, 1]

    # Create a new regressor array with ones for all values in the first column of the data that are not equal to -1 (-1 == no stimuli)
    regressor = np.where(stimulus_col != -1, 1, 0)

    # Define the standard HRF function
    hrf = spm_hrf(1.6, oversampling=1)

    # Convolve the regressor with the standard HRF function
    conv_regressor = np.convolve(regressor, hrf)[: len(regressor)]

    # Get the real baseline TRs from the convolved regressor
    baseline_array = np.where(conv_regressor == 0, 1, 0)
    baseline = np.where(baseline_array == 1)[0]

    for hemi in hemis:

        # Get input files that are to be %-signal changed and load them into a numpy array
        npy_input_file = f"{project_dir}/{d_or_p}/derivatives/highpass/{subject}/{session}/{subject}_{session}_{task_runs[i]}_space-{space}_hemi-{hemi}_desc-denoised_bold.npy"
        data = np.load(npy_input_file)
        print(data.shape)
        # %-signal change the data using the percent_change function from the linescanning pipeline
        print("Performing %-signal change")
        psc_data = mri.percent_change(data, ax=0, baseline=baseline)

        # Define output file name based on input file
        output_file_name = npy_input_file.split("/")[-1].split(".npy")[0]
        output_file = f"{output_dir}/{output_file_name}_psc.npy"

        print(f"Saving original data to psc data in {output_file}")
        np.save(output_file, psc_data)

# highpass
import os
import numpy as np
import json
import argparse
import glob
from nilearn.surface import load_surf_data
from nilearn.glm.first_level.hemodynamic_models import spm_hrf
from prf_expect.utils.mri import highpass_dct
from prf_expect.utils.io import load_settings

settings = load_settings()

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

TR = settings["mri"]["TR"]
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
output_dir = "{}/{}/derivatives/highpass/{}/{}/".format(
    settings["data_dir"], d_or_p, subject, session
)
os.makedirs(output_dir, exist_ok=True)

for i, run in enumerate(runs):
    for hemi in hemis:
        # Get input files that are to be %-signal changed and load them into a numpy array
        surf_file = f"{project_dir}/{d_or_p}/derivatives/del4_1/{subject}/ses-1/func/{subject}_ses-1_{task_runs[i]}_space-{space}_hemi-{hemi}.func.gii"
        data = load_surf_data(surf_file)
        print(f"data shape: {data.shape}")
        # %-signal change the data using the percent_change function from the linescanning pipeline
        print("Performing highpass filtering on data...")
        highpass_data, _ = highpass_dct(
            data, settings["preprocessing"]["highpass"]["threshold"], TR
        )
        print(f"highpass_data shape: {highpass_data.shape}")
        # Define output file name based on input file
        output_file_name = surf_file.split("/")[-1].split(".npy")[0]
        output_file = f"{output_dir}/{subject}_{session}_{task_runs[i]}_space-{space}_hemi-{hemi}_desc-denoised_bold.npy"

        print(f"Saving original data to highpass data in {output_file}")
        np.save(output_file, highpass_data.T)

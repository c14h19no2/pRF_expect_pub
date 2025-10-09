"""
-----------------------------------------------------------------------------------------
pRF_fits.py
-----------------------------------------------------------------------------------------
Goal of the script:
Fit models to prf data (from prfexpect experiment)
Fitted models: 2d Gaussian and Divisive normalization model from the prfpy package
    Both grid-fit and iterative fit.
Use the linescanning pipeline to fit the data
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: subject_name

-----------------------------------------------------------------------------------------
Output(s):
# pickle file with the resulting parameters of the fit
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd ~analysis_code/analysis/
2. run python command
python prf_fits.py [subject name]
-----------------------------------------------------------------------------------------
Exemple:
python pRF_fits.py sub-001
-----------------------------------------------------------------------------------------
Written by Ningkai Wang (n.wang@vu.nl)
current prfpy version: 8342403
-----------------------------------------------------------------------------------------
"""

# General imports
import numpy as np
import os
from scipy import io
import glob
import argparse
from prf_expect.utils.fit import PRFModel
from prf_expect.utils import io


parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "subject",
    default=None,
    nargs="?",
    help="the subject of the experiment, as a zero-filled integer, such as 001, or 04.",
)
parser.add_argument(
    "slurm",
    default=None,
    type=int,
    nargs="?",
    help="slurm or not. 1 or 0",
)
cmd_args = parser.parse_args()
subject, slurm = cmd_args.subject, cmd_args.slurm
# Import settings data from json file

settings = io.load_settings()

# Define paths and data exp parameters
if slurm == 1:
    data_dir = os.path.join(settings["slurm"]["data_dir"], "data")
elif slurm == 0:
    data_dir = os.path.join(settings["general"]["data_dir"], "data")
elif slurm == None:
    print("No slurm option given, using general data dir")
    data_dir = os.path.join(settings["general"]["data_dir"], "data")
else:
    raise ValueError("slurm value should be 0 or 1")

tasks = settings["design"]["tasks"]
cpus = settings["slurm"]["cpus"]
space = settings["mri"]["space"]

# make output dir
output_dir = (
    f"{data_dir}/derivatives/prf_data/{subject}/ses-1/prf_fits/linescanning_params"
)
os.makedirs(output_dir, exist_ok=True)

# load data, first left, then right hemisphere, then concatenate

prf_data = np.concatenate(
    [
        np.load(file).T
        for LR in ["L", "R"]
        for file in sorted(
            glob.glob(
                f"{data_dir}/derivatives/prf_data/{subject}/ses-1/cut_and_averaged/{subject}_ses-1_task-pRF_space-{space}_hemi-{LR}_desc-denoised_bold_psc_mean.npy"
            )
        )
    ]
)

# Load design matrix
prf_dm = np.load(
    f"{data_dir}/derivatives/prf_data/{subject}/ses-1/dms/dm_task-pRF_run-01.npy"
)

# Create a model fitter object
model = PRFModel()
model.get_dm(prf_dm)
model.get_data(prf_data)
model.fit(
    rsq_threshold=settings["prf"]["rsq_threshold"],
    model=settings["prf"]["model_type"],
    output_dir=output_dir,
    output_base=f"{subject}_ses-1_final-fit_space-{space}",
    n_jobs=cpus,
    constraints=["tc", "tc"],
)

# Import the necessary packages
import numpy as np
import pandas as pd
import sys
import glob
import os
import time
from time import strftime, localtime
from prf_expect.utils import io
from prf_expect.utils.fit import PRFModel


def predict_prf(
    subject,
    space,
    prf_data,
    prf_dm,
    prf_params_dir,
    output_dir,
    filename,
    model="norm",
    stage="iter",
    broaden_factor=None,
):
    # load in the model parameters from a pickle file
    params_tsv_name = os.path.join(
        prf_params_dir,
        f"{subject}_ses-1_final-fit_space-{space}_model-{model}_stage-{stage}_desc-prf_params.tsv",
    )
    params_df = pd.read_csv(params_tsv_name, sep="\t")
    if broaden_factor is not None:
        params_df["prf_size"] = params_df["prf_size"] * broaden_factor
    else:
        raise ValueError("Broaden factor must be specified to sharpen the predictions.")
    # Define the objects for the prf fits
    prf_obj = PRFModel()
    prf_obj.get_dm(prf_dm)
    prf_obj.get_data(prf_data)
    prf_obj.load_params(params_df, model=model, stage=stage)
    prf_preds = prf_obj.make_predictions(model="norm")
    np.save(
        os.path.join(
            output_dir,
            filename,
        ),
        prf_preds,
    )


def load_psc_pe_data(subject, space, run, psc_dir, base_name):
    data = np.concatenate(
        [
            np.load(file).T
            for LR in ["L", "R"]
            for file in sorted(
                glob.glob(
                    os.path.join(
                        psc_dir,
                        f"{subject}_ses-1_task-PE_{run}_space-{space}_hemi-{LR}_desc-denoised_bold_psc_{base_name}.npy",
                    )
                )
            )
        ]
    )
    return data


subject = sys.argv[1]
subjects = [subject]
# Import settings data from json file

settings = io.load_settings()

# Define paths and data exp parameters
data_dir = os.path.join(settings["general"]["data_dir"], "data")
tasks = settings["design"]["tasks"]
space = settings["mri"]["space"]
PE_runs = settings["design"]["runs_per_task"]
broaden_factor = 1.2

# Loop over all subjects of interest to make the predictions for
for subject in subjects:
    start_time = time.time()
    print(
        f"Starting to predict timecourses for subject: {subject} at {strftime('%Y-%m-%d %H:%M:%S', localtime(start_time))}"
    )

    sub_dir = os.path.join(data_dir, "derivatives", "prf_data", subject, "ses-1")
    prf_fits_dir = os.path.join(sub_dir, "prf_fits")
    prf_params_dir = os.path.join(prf_fits_dir, "prf_params")
    dm_dir = os.path.join(sub_dir, "dms")
    psc_dir = os.path.join(sub_dir, "cut_and_averaged")

    # make output directory to store the predictions in
    output_dir = os.path.join(prf_fits_dir, "prf_predictions")
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all the PE runs to make the predictions for
    for run in PE_runs:
        # Load in the PE run data
        omission_data = load_psc_pe_data(subject, space, run, psc_dir, "omission")
        sparse_data = load_psc_pe_data(subject, space, run, psc_dir, "sparse")
        violation_data = load_psc_pe_data(subject, space, run, psc_dir, "violation")

        # Load in the PE run dms
        sparse_dm = np.load(
            os.path.join(
                dm_dir,
                f"{subject}_ses-1_task-sparse_{run}_dm.npy",
            )
        )

        violation_dm = np.load(
            os.path.join(
                dm_dir,
                f"{subject}_ses-1_task-violation_{run}_dm.npy",
            )
        )

        # Define the objects for the prf fits
        # Predict the timecourses for the pRF task
        broaden_factor_fn = f"{broaden_factor:.1f}".replace(".", "p")
        sparse_fn = f"{subject}_ses-1_task-sparse_{run}_space-{space}_preds_sharpening-{broaden_factor_fn}.npy"

        predict_prf(
            subject,
            space,
            sparse_data,
            sparse_dm,
            prf_params_dir,
            output_dir,
            sparse_fn,
            broaden_factor=broaden_factor,
        )

        end_time = time.time()
        print(
            f"Finished fitting for {subject} {run} at {strftime('%Y-%m-%d %H:%M:%S', localtime(end_time))}"
        )
        print(
            f"Total time fitting for {subject} {run} was {(end_time - start_time)/60/60:.2f} hours"
        )
        print("-----------------------------------")
    print(
        f"Finished fitting for {subject} at {strftime('%Y-%m-%d %H:%M:%S', localtime(end_time))}"
    )
    print("===================================")

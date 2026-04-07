# Import the necessary packages
import numpy as np
import sys
import glob
import os
import time
from time import strftime, localtime
from prf_expect.utils import io
from prf_expect.utils.fit import PRFModel
from nilearn.glm.first_level.hemodynamic_models import spm_hrf

subject = sys.argv[1]
subjects = [subject]
# Import settings data from json file

settings = io.load_settings()

# Define paths and data exp parameters
data_dir = os.path.join(settings["general"]["data_dir"], "data")
tasks = settings["design"]["tasks"]
space = settings["mri"]["space"]
PE_runs = settings["design"]["runs_per_task"]
hrf = spm_hrf(1.6, oversampling=1)


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

def load_pred_pe_data(subject, space, run, pred_dir, base_name):
    data = np.load(
        os.path.join(
            pred_dir,
            f"{subject}_ses-1_task-{base_name}_{run}_space-{space}_preds.npy",
        )
    )
    return data

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
    pred_dir = os.path.join(prf_fits_dir, "prf_predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # load in the model parameters from a pickle file
    params_tsv_name = os.path.join(
        prf_params_dir,
        f"{subject}_ses-1_final-fit_space-{space}_model-norm_stage-iter_desc-prf_params.tsv",
    )


    omission_data_runs = []
    sparse_data_runs = []
    violation_data_runs = []
    omission_pred_runs = []
    sparse_pred_runs = []
    violation_pred_runs = []
    omission_dm_runs = []
    sparse_dm_runs = []
    violation_dm_runs = []


    # Loop over all the PE runs to make the predictions for
    for run in PE_runs:
        # Load in the PE run data
        omission_data_run = load_psc_pe_data(subject, space, run, psc_dir, "omission")
        sparse_data_run = load_psc_pe_data(subject, space, run, psc_dir, "sparse")
        violation_data_run = load_psc_pe_data(subject, space, run, psc_dir, "violation")
        omission_data_runs.append(omission_data_run)
        sparse_data_runs.append(sparse_data_run)
        violation_data_runs.append(violation_data_run)

        # load predictions for this run
        omission_pred_run = load_pred_pe_data(subject, space, run, pred_dir, "omission")
        sparse_pred_run = load_pred_pe_data(subject, space, run, pred_dir, "sparse")
        violation_pred_run = load_pred_pe_data(subject, space, run, pred_dir, "violation")
        omission_pred_runs.append(omission_pred_run)
        sparse_pred_runs.append(sparse_pred_run)
        violation_pred_runs.append(violation_pred_run)
        print(omission_data_run.shape)
        print(omission_pred_run.shape)

        # load dms for this run
        omission_dm_run = np.load(
            os.path.join(
                dm_dir,
                f"{subject}_ses-1_task-omission_{run}_dm.npy"
            )
        )
        sparse_dm_run = np.load(
            os.path.join(
                dm_dir,
                f"{subject}_ses-1_task-sparse_{run}_dm.npy"
            )
        )
        violation_dm_run = np.load(
            os.path.join(
                dm_dir,
                f"{subject}_ses-1_task-violation_{run}_dm.npy"
            )
        )
        print(omission_dm_run.shape)
        omission_dm_runs.append(omission_dm_run)
        sparse_dm_runs.append(sparse_dm_run)
        violation_dm_runs.append(violation_dm_run)

omission_data_runs = np.concatenate(omission_data_runs, axis=1)
sparse_data_runs = np.concatenate(sparse_data_runs, axis=1)
violation_data_runs = np.concatenate(violation_data_runs, axis=1)
omission_pred_runs = np.concatenate(omission_pred_runs, axis=1)
sparse_pred_runs = np.concatenate(sparse_pred_runs, axis=1)
violation_pred_runs = np.concatenate(violation_pred_runs, axis=1)
omission_dm_runs = np.concatenate(omission_dm_runs, axis=2)
sparse_dm_runs = np.concatenate(sparse_dm_runs, axis=2)
violation_dm_runs = np.concatenate(violation_dm_runs, axis=2)

omission_stim_present = np.convolve(np.any(omission_dm_runs, axis=(0, 1)), hrf)[: len(np.any(omission_dm_runs, axis=(0, 1)))]
sparse_stim_present = np.convolve(np.any(sparse_dm_runs, axis=(0, 1)), hrf)[: len(np.any(sparse_dm_runs, axis=(0, 1)))]
violation_stim_present = np.convolve(np.any(violation_dm_runs, axis=(0, 1)), hrf)[: len(np.any(violation_dm_runs, axis=(0, 1)))]

# calculate R^2 for each condition
def calculate_rsq(data, pred):
    ss_res = np.sum((data - pred) ** 2, axis=1)
    ss_tot = np.sum((data - np.mean(data, axis=1, keepdims=True)) ** 2, axis=1)
    rsq = 1 - (ss_res / ss_tot)
    return rsq


def calculate_rsq_during_stim_present(data, pred):
    data_arr = np.asarray(data)
    pred_arr = np.asarray(pred)

    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D (voxel x time), got shape {data_arr.shape}")

    # Bring predictions to 2D and align with data as (voxel x time).
    pred_2d = np.squeeze(pred_arr)
    if pred_2d.ndim != 2:
        raise ValueError(f"pred must be reducible to 2D, got shape {pred_arr.shape}")
    if pred_2d.shape == data_arr.shape:
        pass
    elif pred_2d.T.shape == data_arr.shape:
        pred_2d = pred_2d.T
    else:
        raise ValueError(
            f"pred shape {pred_arr.shape} (squeezed to {pred_2d.shape}) cannot be aligned to data shape {data_arr.shape}"
        )

    # Compute R^2 per voxel using that voxel's own timepoint mask.
    rsq = np.full(data_arr.shape[0], np.nan, dtype=float)
    for v in range(data_arr.shape[0]):
        mask = pred_2d[v] >= 0.01
        if not np.any(mask):
            continue

        data_masked = data_arr[v, mask]
        pred_masked = pred_2d[v, mask]
        ss_res = np.sum((data_masked - pred_masked) ** 2)
        ss_tot = np.sum((data_masked - np.mean(data_masked)) ** 2)
        if ss_tot != 0:
            rsq[v] = 1 - (ss_res / ss_tot)

    return rsq

# def calculate_rsq_during_stim_present(data, pred, stim_present):
#     # Build a 1D timepoint mask and preserve data as 2D (voxel x time).
#     mask = np.asarray(stim_present) >= 0.01
#     print(mask)
#     if mask.ndim != 1:
#         mask = np.squeeze(mask)
#     if mask.ndim != 1 or mask.shape[0] != data.shape[1]:
#         raise ValueError(
#             f"stim_present mask must be 1D with length {data.shape[1]}, got shape {mask.shape}"
#         )
#     if not np.any(mask):
#         return np.full(data.shape[0], np.nan)

#     data_masked = data[:, mask]
#     pred_masked = pred[:, mask]
#     ss_res = np.sum((data_masked - pred_masked) ** 2, axis=1)
#     ss_tot = np.sum((data_masked - np.mean(data_masked, axis=1, keepdims=True)) ** 2, axis=1)
#     rsq = 1 - np.divide(ss_res, ss_tot, out=np.full_like(ss_res, np.nan, dtype=float), where=ss_tot != 0)
#     return rsq
# omission_rsq = calculate_rsq(omission_data_runs, omission_pred_runs)
# sparse_rsq = calculate_rsq(sparse_data_runs, sparse_pred_runs)
# violation_rsq = calculate_rsq(violation_data_runs, violation_pred_runs)
omission_rsq_stim_present = calculate_rsq_during_stim_present(omission_data_runs, omission_pred_runs)
sparse_rsq_stim_present = calculate_rsq_during_stim_present(sparse_data_runs, sparse_pred_runs)
violation_rsq_stim_present = calculate_rsq_during_stim_present(violation_data_runs, violation_pred_runs)

# omission_rsq_stim_present = calculate_rsq_during_stim_present(omission_data_runs, omission_pred_runs, omission_stim_present)
# sparse_rsq_stim_present = calculate_rsq_during_stim_present(sparse_data_runs, sparse_pred_runs, sparse_stim_present)
# violation_rsq_stim_present = calculate_rsq_during_stim_present(violation_data_runs, violation_pred_runs, violation_stim_present)

# save the R^2 values in a numpy file
# np.save(
#     os.path.join(pred_dir, f"{subject}_ses-1_prf_prediction_rsq_omission.npy"),
#     omission_rsq,
# )
# np.save(
#     os.path.join(pred_dir, f"{subject}_ses-1_prf_prediction_rsq_sparse.npy"),
#     sparse_rsq,
# )
# np.save(
#     os.path.join(pred_dir, f"{subject}_ses-1_prf_prediction_rsq_violation.npy"),
#     violation_rsq,
# )
np.save(
    os.path.join(pred_dir, f"{subject}_ses-1_prf_prediction_rsq_omission_stim_present.npy"),
    omission_rsq_stim_present,
)
np.save(
    os.path.join(pred_dir, f"{subject}_ses-1_prf_prediction_rsq_sparse_stim_present.npy"),
    sparse_rsq_stim_present,
)
np.save(
    os.path.join(pred_dir, f"{subject}_ses-1_prf_prediction_rsq_violation_stim_present.npy"),
    violation_rsq_stim_present,
)
"""
Sparse regressor amplitude was first estimated from sparse-condition data, then fixed,
and used in the violation-condition GLM analysis to estimate omission regressor amplitude.
"""

import numpy as np
import json
import glob
import itertools
import os
import time
import sys
import pandas as pd
from prf_expect.utils import io
from prf_expect.utils.mri import con_nan_1darray
from prf_expect.utils.analysis import calc_t_stat


def fit_viol_model(
    viol_data,
    pRF_pred,
    omit_pred,
    spar_pred,
    betas_spar,
):
    betas_omit = con_nan_1darray(omit_pred)
    surprise_ratio = con_nan_1darray(viol_data)
    rsq = con_nan_1darray(viol_data)
    t_contrast = con_nan_1darray(viol_data)
    p_contrast = con_nan_1darray(viol_data)
    y_pred = np.zeros(viol_data.shape)
    residuals0 = viol_data - betas_spar * spar_pred
    nan_nr = 0
    for vertex in range(residuals0.shape[1]):
        if (
            np.isnan(omit_pred[:, vertex]).any()
            or np.isnan(spar_pred[:, vertex]).any()
            or np.isnan(pRF_pred[:, vertex]).any()
        ):
            nan_nr += 1
            continue
        x1 = np.vstack(
            [
                omit_pred[:, vertex],
            ]
        ).T
        # print(f"vertex {vertex} start")
        y = residuals0[:, vertex]
        betas1, _, _, _ = np.linalg.lstsq(x1, y, rcond=None)
        y_pred1 = np.dot(x1, betas1)
        residuals1 = y - y_pred1
        betas_omit[vertex] = betas1[0]
        surprise_ratio[vertex] = (
            (betas_omit[vertex] + betas_spar[vertex])
            and ((betas_spar[vertex] / (betas_omit[vertex] + betas_spar[vertex])) - 0.5)
            or 0
        )

        rsq[vertex] = 1 - np.sum(residuals1**2) / np.sum(
            (viol_data[:, vertex] - np.nanmean(viol_data[:, vertex])) ** 2
        )

        x0 = np.vstack(
            [
                omit_pred[:, vertex],
                spar_pred[:, vertex],
            ]
        ).T

        y_pred[:, vertex] = (
            +np.dot(x1, betas1) + betas_spar[vertex] * spar_pred[:, vertex]
        )
        t, p = calc_t_stat(
            x0,
            viol_data[:, vertex],
            y_pred[:, vertex],
            np.array([betas_omit[vertex], betas_spar[vertex]]),
            [-1, 1],
        )
        t_contrast[vertex] = t
        p_contrast[vertex] = p

    print(f"surprise ratio calculated for {subject}")
    print(f"Output saved to {output_dir}")
    print(f"nan in x for {nan_nr} / {viol_data.shape[1]} vertices")
    return (
        surprise_ratio,
        betas_omit,
        betas_spar,
        rsq,
        t_contrast,
        p_contrast,
    )


def fit_spar_model(spar_data, spar_pred):
    betas_spar = con_nan_1darray(spar_pred)
    rsq = con_nan_1darray(spar_data)
    nan_nr = 0
    for vertex in range(spar_data.shape[1]):
        if np.isnan(spar_pred[:, vertex]).any():
            nan_nr += 1
            continue

        x1 = np.vstack(
            [
                spar_pred[:, vertex],
            ]
        ).T

        y = spar_data[:, vertex]
        betas1, residuals, _, _ = np.linalg.lstsq(x1, y, rcond=None)
        betas_spar[vertex] = betas1[0]

        if len(residuals) == 0:
            rsq[vertex] = np.nan
        else:
            rsq[vertex] = 1 - residuals / np.sum((y - y.mean()) ** 2)

    print(f"surprise ratio calculated for {subject}")
    print(f"Output saved to {output_dir}")
    print(f"nan in x for {nan_nr} / {spar_data.shape[1]} vertices")
    return (
        betas_spar,
        rsq,
    )


subject = sys.argv[1]
start_time = time.time()
print(f"Calculating surprise ratio for {subject}")

# Load settings from json file
settings = io.load_settings()

# Define paths and data exp parameters
data_dir = os.path.join(settings["general"]["data_dir"], "data")
TR = settings["mri"]["TR"]
# subjects = settings["subject_list"]
PE_runs = settings["design"]["runs_per_task"]
bad_PE_runs = settings["QC"]["bad_PE_runs"][subject]
overlapping_runs = []
for run in bad_PE_runs:
    if run in PE_runs:
        overlapping_runs.append(run)
for run in overlapping_runs:
    PE_runs.remove(run)
print("Processing PE runs: ", PE_runs)

# define the directory
output_dir = os.path.join(
    data_dir, "derivatives", "prf_data", subject, "ses-1", "TRMI-type4"
)
os.makedirs(output_dir, exist_ok=True)

params_dir = os.path.join(
    data_dir,
    "derivatives",
    "prf_data",
    subject,
    "ses-1",
    "prf_fits",
    "prf_params",
)

pred_dir = os.path.join(
    data_dir,
    "derivatives",
    "prf_data",
    subject,
    "ses-1",
    "prf_fits",
    "prf_predictions",
)

psc_dir = os.path.join(
    data_dir,
    "derivatives",
    "prf_data",
    subject,
    "ses-1",
    "cut_and_averaged",
)

dm_dir = os.path.join(data_dir, "derivatives", "prf_data", subject, "ses-1", "dms")


# Get HRF parameters
fit_params = pd.read_csv(
    os.path.join(
        params_dir,
        f"{subject}_ses-1_final-fit_space-fsaverage_model-norm_stage-iter_desc-prf_params.tsv",
    ),
    sep="\t",
    header=0,
)

print(fit_params.keys())
hrf_deriv = fit_params["hrf_deriv"].values
hrf_dsip = fit_params["hrf_dsip"].values


# Load the data
viol_data_runs = []
omit_data_runs = []
spar_data_runs = []
omit_pred_runs = []
spar_pred_runs = []
spar_pred_runs = []
spar_dm_runs = []

# load pRF pred data
pRF_pred_run = np.load(
    os.path.join(
        pred_dir,
        f"{subject}_ses-1_task-pRF_final-fit_space-fsaverage_model-norm_stage-iter_desc-prf_pred.npy",
    )
).T

for run in PE_runs:
    # Load the violation data
    viol_data_run = np.concatenate(
        [
            np.load(file).T
            for LR in ["L", "R"]
            for file in sorted(
                glob.glob(
                    os.path.join(
                        psc_dir,
                        f"{subject}_ses-1_task-PE_{run}_space-fsaverage_hemi-{LR}_desc-denoised_bold_psc_violation.npy",
                    )
                )
            )
        ]
    ).T

    # load omission data
    omit_data_run = np.concatenate(
        [
            np.load(file).T
            for LR in ["L", "R"]
            for file in sorted(
                glob.glob(
                    os.path.join(
                        psc_dir,
                        f"{subject}_ses-1_task-PE_{run}_space-fsaverage_hemi-{LR}_desc-denoised_bold_psc_omission.npy",
                    )
                )
            )
        ]
    ).T

    spar_data_run = np.concatenate(
        [
            np.load(file).T
            for LR in ["L", "R"]
            for file in sorted(
                glob.glob(
                    os.path.join(
                        psc_dir,
                        f"{subject}_ses-1_task-PE_{run}_space-fsaverage_hemi-{LR}_desc-denoised_bold_psc_sparse.npy",
                    )
                )
            )
        ]
    ).T

    # load omission pred data
    omit_pred_run = np.load(
        os.path.join(
            pred_dir, f"{subject}_ses-1_task-omission_{run}_space-fsaverage_preds.npy"
        )
    ).T

    # load sparse pred data
    spar_pred_run = np.load(
        os.path.join(
            pred_dir, f"{subject}_ses-1_task-sparse_{run}_space-fsaverage_preds.npy"
        )
    ).T

    # load the sparse dm
    spar_dm_load = np.load(
        os.path.join(dm_dir, f"{subject}_ses-1_task-sparse_{run}_dm.npy")
    )
    spar_dm_sum = np.sum(spar_dm_load, axis=(0, 1))
    spar_dm_run = np.where(spar_dm_sum > 0, 1, 0)

    viol_data_runs.append(viol_data_run)
    omit_data_runs.append(omit_data_run)
    spar_data_runs.append(spar_data_run)
    omit_pred_runs.append(omit_pred_run)
    spar_pred_runs.append(spar_pred_run)
    spar_dm_runs.append(spar_dm_run)

viol_data = np.concatenate(viol_data_runs, axis=0)
omit_data = np.concatenate(omit_data_runs, axis=0)
spar_data = np.concatenate(spar_data_runs, axis=0)
pRF_pred = np.tile(pRF_pred_run, (len(PE_runs), 1))
omit_pred = np.concatenate(omit_pred_runs, axis=0)
spar_pred = np.concatenate(spar_pred_runs, axis=0)
spar_dm = np.concatenate(spar_dm_runs, axis=0)


# For now all the data is in the form of timepoints x vertices
r_betas_spar, r_rsq = fit_spar_model(spar_data, spar_pred)

np.save(
    os.path.join(
        output_dir,
        f"{subject}_ses-1_space-fsaverage_surprise-spar_beta_spardm.npy",
    ),
    r_betas_spar,
)

np.save(
    os.path.join(
        output_dir,
        f"{subject}_ses-1_space-fsaverage_surprise-spar_rsq..npy",
    ),
    r_rsq,
)

print("Elapsed time: ", time.time() - start_time)

"""
loop over vertices for
viol_data = omission_pred * x0 + sparse_pred * x1 + sparse_hrf_convolve * x2 + prfexpect * x3 + residuals
"""

(
    v_surprise_ratio,
    v_betas_omit,
    v_betas_spar,
    v_rsq,
    v_t_contrast,
    v_p_contrast,
) = fit_viol_model(
    viol_data,
    pRF_pred,
    omit_pred,
    spar_pred,
    r_betas_spar,
)

np.save(
    os.path.join(
        output_dir,
        f"{subject}_ses-1_space-fsaverage_surprise-TRMI.npy",
    ),
    v_surprise_ratio,
)

np.save(
    os.path.join(
        output_dir,
        f"{subject}_ses-1_space-fsaverage_surprise-viol_beta_omitdm.npy",
    ),
    v_betas_omit,
)

np.save(
    os.path.join(
        output_dir,
        f"{subject}_ses-1_space-fsaverage_surprise-viol_beta_sparsedm.npy",
    ),
    v_betas_spar,
)

np.save(
    os.path.join(
        output_dir,
        f"{subject}_ses-1_space-fsaverage_surprise-viol_rsq.npy",
    ),
    v_rsq,
)

np.save(
    os.path.join(
        output_dir,
        f"{subject}_ses-1_space-fsaverage_surprise-viol_t_contrast.npy",
    ),
    v_t_contrast,
)

np.save(
    os.path.join(
        output_dir,
        f"{subject}_ses-1_space-fsaverage_surprise-viol_p_contrast.npy",
    ),
    v_p_contrast,
)

print("Elapsed time: ", time.time() - start_time)

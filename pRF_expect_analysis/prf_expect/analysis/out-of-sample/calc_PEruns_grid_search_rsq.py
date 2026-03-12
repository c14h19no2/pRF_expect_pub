# Import the necessary packages
import itertools
import numpy as np
import sys
import glob
import os
import time
from time import strftime, localtime
import pandas as pd
import cortex as cx
from prf_expect.utils import io
from prf_expect.utils.fit import PRFModel


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


# calculate R^2 for each condition
def calculate_rsq(data, pred):
    ss_res = np.sum((data - pred) ** 2, axis=1)
    ss_tot = np.sum((data - np.mean(data, axis=1, keepdims=True)) ** 2, axis=1)
    rsq = 1 - (ss_res / ss_tot)
    return rsq


def build_prf_param_grid(
    prf_params: pd.DataFrame,
    scale_min: float = -0.20,
    scale_max: float = 0.20,
    step: float = 0.01,
):
    """Return parameter columns and a grid that scales one parameter at a time.

    Instead of the full Cartesian product, we vary each parameter independently
    across the scale factors while keeping others at 1.0. A baseline (all ones)
    is included as the first row.
    """

    param_cols = [
        col
        for col in [
            "x",
            "y",
            "prf_size",
            "prf_ampl",
            "bold_bsl",
            "surr_ampl",
            "surr_size",
            "neur_bsl",
            "surr_bsl",
        ]
        if col in prf_params.columns
    ]

    if not param_cols:
        raise ValueError("No parameter columns found to scale")

    scale_factors = np.arange(1 + scale_min, 1 + scale_max + 1e-9, step)
    if 1.0 not in scale_factors:
        scale_factors = np.sort(np.unique(np.append(scale_factors, 1.0)))

    # Build grid: baseline (all ones) plus per-parameter scaling vectors
    grid = [np.ones(len(param_cols))]
    for idx, _ in enumerate(param_cols):
        for sf in scale_factors:
            if sf == 1.0:
                continue
            vec = np.ones(len(param_cols))
            vec[idx] = sf
            grid.append(vec)

    factor_grid = np.array(grid)

    return param_cols, factor_grid


subject = sys.argv[1]
subjects = [subject]
# Import settings data from json file

settings = io.load_settings()

# Define paths and data exp parameters
data_dir = os.path.join(settings["general"]["data_dir"], "data")
tasks = settings["design"]["tasks"]
space = settings["mri"]["space"]
PE_runs = settings["design"]["runs_per_task"]
roi_verts = cx.get_roi_verts('fsaverage', ('V1',))
roi_indices = roi_verts["V1"]
nr_best_voxels = 200

# Loop over all subjects of interest to make the predictions for
for subject in subjects:
    start_time = time.time()
    print(
        f"Starting to predict timecourses for subject: {subject} at {strftime('%Y-%m-%d %H:%M:%S', localtime(start_time))}",
        flush=True,
    )

    sub_dir = os.path.join(data_dir, "derivatives", "prf_data", subject, "ses-1")
    prf_fits_dir = os.path.join(sub_dir, "prf_fits")
    prf_params_dir = os.path.join(prf_fits_dir, "prf_params")
    dm_dir = os.path.join(sub_dir, "dms")
    psc_dir = os.path.join(sub_dir, "cut_and_averaged")

    # make output directory to store the predictions in
    output_dir = os.path.join(prf_fits_dir, "prf_predictions")
    os.makedirs(output_dir, exist_ok=True)

    # load in the model parameters from a pickle file
    params_tsv_name = os.path.join(
        prf_params_dir,
        f"{subject}_ses-1_final-fit_space-{space}_model-norm_stage-iter_desc-prf_params.tsv",
    )
    prf_params = pd.read_csv(params_tsv_name, sep="\t")

    # perform mask to only keep the best voxels in V1
    prf_params = prf_params.iloc[roi_indices]
    prf_params = prf_params.sort_values(by="r2", ascending=False).head(nr_best_voxels)
    # get the index of the best voxels in the original data
    best_voxel_indices = prf_params.index.values
    prf_params = prf_params.reset_index(drop=True)

    # Concatenate all runs per condition before predicting
    omission_data_runs = []
    sparse_data_runs = []
    violation_data_runs = []
    omission_dm_runs = []
    sparse_dm_runs = []
    violation_dm_runs = []

    for run in PE_runs:
        omission_data_runs.append(
            load_psc_pe_data(subject, space, run, psc_dir, "omission")[best_voxel_indices]
        )
        sparse_data_runs.append(
            load_psc_pe_data(subject, space, run, psc_dir, "sparse")[best_voxel_indices]
        )
        violation_data_runs.append(
            load_psc_pe_data(subject, space, run, psc_dir, "violation")[best_voxel_indices]
        )

        omission_dm_runs.append(
            np.load(os.path.join(dm_dir, f"{subject}_ses-1_task-omission_{run}_dm.npy"))
        )
        sparse_dm_runs.append(
            np.load(os.path.join(dm_dir, f"{subject}_ses-1_task-sparse_{run}_dm.npy"))
        )
        violation_dm_runs.append(
            np.load(os.path.join(dm_dir, f"{subject}_ses-1_task-violation_{run}_dm.npy"))
        )

    # Concatenate along time axis (axis=1 for data, axis=-1 for dms)
    omission_data = np.concatenate(omission_data_runs, axis=1)
    sparse_data = np.concatenate(sparse_data_runs, axis=1)
    violation_data = np.concatenate(violation_data_runs, axis=1)

    omission_dm = np.concatenate(omission_dm_runs, axis=-1)
    sparse_dm = np.concatenate(sparse_dm_runs, axis=-1)
    violation_dm = np.concatenate(violation_dm_runs, axis=-1)

    # Predict once on concatenated data
    omission_fn = f"{subject}_ses-1_task-omission_runs-all_space-{space}_gridrsqs.npy"
    sparse_fn = f"{subject}_ses-1_task-sparse_runs-all_space-{space}_gridrsqs.npy"
    violation_fn = f"{subject}_ses-1_task-violation_runs-all_space-{space}_gridrsqs.npy"

    param_cols, factor_grid = build_prf_param_grid(prf_params)
    base_params = prf_params[param_cols].to_numpy()
    r2_vals = prf_params["r2"].values if "r2" in prf_params.columns else None

    n_combos = factor_grid.shape[0]
    n_vox = base_params.shape[0]

    # Initialize PRF models once per condition to avoid repeated setup
    prf_obj_omit = PRFModel()
    prf_obj_omit.get_dm(omission_dm)
    prf_obj_omit.get_data(omission_data)

    prf_obj_spar = PRFModel()
    prf_obj_spar.get_dm(sparse_dm)
    prf_obj_spar.get_data(sparse_data)

    prf_obj_viol = PRFModel()
    prf_obj_viol.get_dm(violation_dm)
    prf_obj_viol.get_data(violation_data)

    rsq_omit_all = np.lib.format.open_memmap(
        os.path.join(output_dir, omission_fn),
        mode="w+",
        dtype="float32",
        shape=(n_combos, n_vox),
    )
    rsq_spar_all = np.lib.format.open_memmap(
        os.path.join(output_dir, sparse_fn),
        mode="w+",
        dtype="float32",
        shape=(n_combos, n_vox),
    )
    rsq_viol_all = np.lib.format.open_memmap(
        os.path.join(output_dir, violation_fn),
        mode="w+",
        dtype="float32",
        shape=(n_combos, n_vox),
    )

    combo_start_time = time.time()

    for combo_idx, factors in enumerate(factor_grid):
        scaled_params = base_params * factors
        prf_params_grid = pd.DataFrame(scaled_params, columns=param_cols)
        if r2_vals is not None:
            prf_params_grid["r2"] = r2_vals

        prf_obj_omit.load_params(prf_params_grid, model="norm", stage="iter")
        prf_preds_omit = prf_obj_omit.make_predictions(model="norm")
        rsq_omit = calculate_rsq(omission_data, prf_preds_omit)

        prf_obj_spar.load_params(prf_params_grid, model="norm", stage="iter")
        prf_preds_spar = prf_obj_spar.make_predictions(model="norm")
        rsq_spar = calculate_rsq(sparse_data, prf_preds_spar)

        prf_obj_viol.load_params(prf_params_grid, model="norm", stage="iter")
        prf_preds_viol = prf_obj_viol.make_predictions(model="norm")
        rsq_viol = calculate_rsq(violation_data, prf_preds_viol)

        rsq_omit_all[combo_idx] = rsq_omit.astype(np.float32)
        rsq_spar_all[combo_idx] = rsq_spar.astype(np.float32)
        rsq_viol_all[combo_idx] = rsq_viol.astype(np.float32)

        if combo_idx == 0 or combo_idx % 5 == 0:
            elapsed = time.time() - combo_start_time
            avg_per_combo = elapsed / (combo_idx + 1)
            remaining = (n_combos - combo_idx - 1) * avg_per_combo
            print(
                f"Processed combo {combo_idx+1}/{n_combos}; grid size {n_combos}, voxels {n_vox}; "
                f"elapsed {elapsed/60:.1f} min; eta {remaining/60:.1f} min",
                flush=True,
            )

    rsq_omit_all.flush()
    rsq_spar_all.flush()
    rsq_viol_all.flush()

    end_time = time.time()
    print(
        f"Finished fitting for {subject} at {strftime('%Y-%m-%d %H:%M:%S', localtime(end_time))}",
        flush=True,
    )
    print(
        f"Total time fitting for {subject} was {(end_time - start_time)/60/60:.2f} hours",
        flush=True,
    )
    print("===================================", flush=True)

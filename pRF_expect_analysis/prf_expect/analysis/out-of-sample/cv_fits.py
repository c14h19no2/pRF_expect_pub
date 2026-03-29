import os
import numpy as np
import glob
import argparse
import re
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Norm_Iso2DGaussianModel
from prfpy.fit import Norm_Iso2DGaussianFitter
from prf_expect.utils import io
from prf_expect.utils.fit import Parameters
import pandas as pd
from pathlib import Path


def _extract_run_subrun(file_path: Path):
    match = re.search(r"run-(\d+).*_(subrun\d+)\.npy$", file_path.name)
    if match is None:
        raise ValueError(f"Could not parse run/subrun from file name: {file_path.name}")
    return match.group(1), match.group(2)


def load_standard_subruns(subject, space, psc_dir):
    left_files = sorted(
        psc_dir.glob(
            f"{subject}_ses-1_task-pRF_run-*_space-{space}_hemi-L_desc-denoised_bold_psc_subrun*.npy"
        )
    )

    if not left_files:
        raise FileNotFoundError(
            f"No left-hemisphere pRF subrun files found in {psc_dir} for {subject}."
        )

    subrun_entries = []
    for l_file in left_files:
        run, subrun = _extract_run_subrun(l_file)
        r_file = Path(str(l_file).replace("_hemi-L_", "_hemi-R_"))
        if not r_file.exists():
            raise FileNotFoundError(f"Missing matching right-hemisphere file: {r_file}")

        # Stored files are typically time x voxels; transpose to voxels x time.
        subrun_data = np.concatenate([np.load(l_file).T, np.load(r_file).T], axis=0)
        subrun_entries.append(
            {
                "run": run,
                "subrun": subrun,
                "name": f"run-{run}_{subrun}",
                "data": subrun_data,
                "left_file": l_file,
                "right_file": r_file,
            }
        )

    if len(subrun_entries) != 9:
        raise ValueError(
            f"Expected 9 pRF subruns for condition='standard', found {len(subrun_entries)}."
        )

    first_shape = subrun_entries[0]["data"].shape
    for entry in subrun_entries[1:]:
        if entry["data"].shape != first_shape:
            raise ValueError(
                "All pRF subruns must have the same shape. "
                f"Got {first_shape} and {entry['data'].shape} (for {entry['name']})."
            )

    return subrun_entries


def _extract_pe_run(file_path: Path):
    match = re.search(r"task-PE_(run-\d+)_", file_path.name)
    if match is None:
        raise ValueError(f"Could not parse PE run from file name: {file_path.name}")
    return match.group(1)


def _run_sort_key(run_label: str):
    match = re.search(r"run-(\d+)", run_label)
    if match is None:
        return run_label
    return int(match.group(1))


def load_condition_runs(subject, space, psc_dir, dm_dir, condition):
    left_files = sorted(
        psc_dir.glob(
            f"{subject}_ses-1_task-PE_run-*_space-{space}_hemi-L_desc-denoised_bold_psc_{condition}.npy"
        ),
        key=lambda p: _run_sort_key(_extract_pe_run(p)),
    )

    if not left_files:
        raise FileNotFoundError(
            f"No left-hemisphere PE files found in {psc_dir} for {subject}, condition={condition}."
        )

    run_entries = []
    for l_file in left_files:
        run = _extract_pe_run(l_file)
        r_file = Path(str(l_file).replace("_hemi-L_", "_hemi-R_"))
        if not r_file.exists():
            raise FileNotFoundError(f"Missing matching right-hemisphere file: {r_file}")

        dm_file = dm_dir / f"{subject}_ses-1_task-{condition}_{run}_dm.npy"
        if not dm_file.exists():
            raise FileNotFoundError(f"Missing design matrix for {condition} {run}: {dm_file}")

        # Stored files are typically time x voxels; transpose to voxels x time.
        run_data = np.concatenate([np.load(l_file).T, np.load(r_file).T], axis=0)
        run_dm = np.load(dm_file)
        if run_data.shape[1] != run_dm.shape[-1]:
            raise ValueError(
                f"Time length mismatch for {condition} {run}: "
                f"data has {run_data.shape[1]} TRs, DM has {run_dm.shape[-1]} TRs."
            )

        run_entries.append(
            {
                "run": run,
                "name": run,
                "data": run_data,
                "dm": run_dm,
                "left_file": l_file,
                "right_file": r_file,
                "dm_file": dm_file,
            }
        )

    if len(run_entries) < 2:
        raise ValueError(
            f"Expected at least 2 runs for condition='{condition}' LOO fitting, found {len(run_entries)}."
        )

    first_n_vox = run_entries[0]["data"].shape[0]
    first_dm_xy = run_entries[0]["dm"].shape[:2]
    for entry in run_entries[1:]:
        if entry["data"].shape[0] != first_n_vox:
            raise ValueError(
                "All PE runs must have the same number of voxels. "
                f"Got {first_n_vox} and {entry['data'].shape[0]} (for {entry['name']})."
            )
        if entry["dm"].shape[:2] != first_dm_xy:
            raise ValueError(
                "All PE run design matrices must have the same spatial shape. "
                f"Got {first_dm_xy} and {entry['dm'].shape[:2]} (for {entry['name']})."
            )

    return run_entries


def build_stimulus(dm, settings):
    return PRFStimulus2D(
        np.array(settings["monitor"]["screen_size_cm"][1]),
        settings["monitor"]["screen_distance_cm"],
        dm,
        settings["mri"]["TR"],
        task_lengths=dm.shape[-1],
    )


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
        mask = np.abs(pred_2d[v]) >= 0.01
        if not np.any(mask):
            continue

        data_masked = data_arr[v, mask]
        pred_masked = pred_2d[v, mask]
        ss_res = np.sum((data_masked - pred_masked) ** 2)
        ss_tot = np.sum((data_masked - np.mean(data_masked)) ** 2)
        if ss_tot != 0:
            rsq[v] = 1 - (ss_res / ss_tot)

    return rsq


def fit_prf_model(starting_params, train_stimulus, train_data, test_data, settings, test_stimulus=None, n_jobs=1):
    verbose = False
    rsq_threshold = 0.1
    ss = train_stimulus.screen_size_degrees
    max_ecc_size = ss / 2.0

    coord_bounds = (-1.5 * max_ecc_size, 1.5 * max_ecc_size)
    prf_size = (0.2, 1.5 * ss)
    norm_bounds = [
        coord_bounds,  # x
        coord_bounds,  # y
        prf_size,  # prf size
        settings["prf"]["prf_ampl"],  # prf amplitude
        settings["prf"]["bold_bsl"],  # bold baseline
        settings["prf"]["norm"]["surround_amplitude_bound"],  # surround amplitude
        (settings["prf"]["eps"], 3 * ss),  # surround size
        settings["prf"]["norm"]["neural_baseline_bound"],  # neural baseline
        settings["prf"]["norm"]["surround_baseline_bound"],
    ]  # surround baseline
    norm_bounds += [
        settings["prf"]["hrf"]["deriv_bound"],
        settings["prf"]["hrf"]["disp_bound"],
    ]
    norm_model = Norm_Iso2DGaussianModel(
        train_stimulus,
        hrf=[1, 4.6, 0],
        filter_predictions=False,
        normalize_RFs=False,
        normalize_hrf=True,
    )
    norm_fitter = Norm_Iso2DGaussianFitter(
        norm_model,
        train_data,
        n_jobs=n_jobs,
    )
    norm_fitter.iterative_fit(
        rsq_threshold,
        verbose=verbose,
        bounds=norm_bounds,
        constraints=[],
        starting_params=starting_params,
    )

    if test_stimulus is not None:
        norm_model.stimulus = test_stimulus
    
    iterative_search_params = norm_fitter.iterative_search_params.copy()
    predictions = []
    for vox in range(iterative_search_params.shape[0]):
        pars = iterative_search_params[vox, ...]
        pred = np.asarray(norm_model.return_prediction(*pars[:-1])).squeeze()
        if pred.ndim != 1:
            raise ValueError(
                f"Unexpected prediction shape for voxel {vox}: {pred.shape}"
            )
        predictions.append(pred)

    # calculate R^2 for the test data
    predictions = np.stack(predictions, axis=0)
    ss_total = np.sum((test_data - np.mean(test_data, axis=1, keepdims=True)) ** 2, axis=1)
    ss_res = np.sum((test_data - predictions) ** 2, axis=1)
    loo_r_squared = 1 - (ss_res / ss_total)
    loo_r_squared[np.isnan(loo_r_squared)] = 0  # Handle cases where ss_total is zero.
    loo_r_squared[loo_r_squared < 0] = 0  # Set negative R^2 to zero.
    loo_r_squared[np.isfinite(loo_r_squared) == False] = 0  # Handle any remaining non-finite values.

    DN_par = Parameters(norm_fitter.iterative_search_params, model="norm")
    DN_par_df = DN_par.to_df()
    DN_par_df["loo_r2"] = loo_r_squared

    loo_r_squared_stim_present = calculate_rsq_during_stim_present(test_data, predictions)
    DN_par_df["loo_r2_stim_present"] = loo_r_squared_stim_present

    return DN_par_df


parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "subject",
    default=None,
    nargs="?",
    help="the subject of the experiment, as a zero-filled integer, such as 001, or 04.",
)

parser.add_argument(
    "condition",
    default=None,
    nargs="?",
    help="the condition of the experiment, standard, violation, omission, sparse",
)
cmd_args = parser.parse_args()
subject, condition = cmd_args.subject, cmd_args.condition

settings = io.load_settings()
data_dir = Path(settings["general"]["data_dir"]) / "data"
sub_dir = data_dir / "derivatives" / "prf_data" / subject / "ses-1"
prf_fits_dir = sub_dir / "prf_fits"
prf_params_dir = prf_fits_dir / "prf_params"
pred_dir = prf_fits_dir / "prf_predictions"
space = settings["mri"]["space"]

params_tsv = (
    prf_params_dir
    / f"{subject}_ses-1_final-fit_space-{space}_model-norm_stage-iter_desc-prf_params.tsv"
)
starting_params = pd.read_csv(params_tsv, sep="\t")

if condition == "standard":
    psc_dir = sub_dir / "cut_and_averaged"
    dm_dir = sub_dir / "dms"
    cv_params_dir = prf_params_dir / "cv"
    cv_params_dir.mkdir(parents=True, exist_ok=True)

    dm_file = dm_dir / "dm_task-pRF_run-01.npy"
    if not dm_file.exists():
        raise FileNotFoundError(f"Could not find pRF design matrix: {dm_file}")

    prf_dm = np.load(dm_file)
    stimulus = build_stimulus(prf_dm, settings)

    # Ensure parameter order matches the fitter's expected array format.
    starting_params_array = Parameters(starting_params, model="norm").to_array()
    subrun_entries = load_standard_subruns(subject, space, psc_dir)

    n_jobs = settings.get("slurm", {}).get("cpus", 1)

    for fold_idx, heldout_entry in enumerate(subrun_entries):
        train_data = np.mean(
            [entry["data"] for i, entry in enumerate(subrun_entries) if i != fold_idx],
            axis=0,
        )
        # Held-out subrun for this LOO fold; used for out-of-sample evaluation.
        test_data = heldout_entry["data"]

        fold_params_df = fit_prf_model(
            starting_params=starting_params_array,
            train_stimulus=stimulus,
            train_data=train_data,
            test_data=test_data,
            settings=settings,
            n_jobs=n_jobs,
        )

        fold_name = (
            f"{subject}_ses-1_standard-fit_space-{space}_loo-{heldout_entry['name']}"
        )
        fold_tsv = (
            cv_params_dir / f"{fold_name}_model-norm_stage-iter_desc-prf_params.tsv"
        )
        
        fold_params_df.to_csv(fold_tsv, sep="\t", index=False)

elif condition in {"violation", "omission", "sparse"}:
    psc_dir = sub_dir / "cut_and_averaged"
    dm_dir = sub_dir / "dms"
    cv_params_dir = prf_params_dir / "cv"
    cv_params_dir.mkdir(parents=True, exist_ok=True)

    # Ensure parameter order matches the fitter's expected array format.
    starting_params_array = Parameters(starting_params, model="norm").to_array()
    run_entries = load_condition_runs(subject, space, psc_dir, dm_dir, condition)

    n_jobs = settings.get("slurm", {}).get("cpus", 1)

    for fold_idx, heldout_entry in enumerate(run_entries):
        train_entries = [entry for i, entry in enumerate(run_entries) if i != fold_idx]

        # PE design matrices differ across runs, so concatenate runs along time.
        train_data = np.concatenate([entry["data"] for entry in train_entries], axis=1)
        train_dm = np.concatenate([entry["dm"] for entry in train_entries], axis=2)

        train_stimulus = build_stimulus(train_dm, settings)
        test_stimulus = build_stimulus(heldout_entry["dm"], settings)
        test_data = heldout_entry["data"]

        fold_params_df = fit_prf_model(
            starting_params=starting_params_array,
            train_stimulus=train_stimulus,
            train_data=train_data,
            test_data=test_data,
            settings=settings,
            test_stimulus=test_stimulus,
            n_jobs=n_jobs,
        )

        fold_name = (
            f"{subject}_ses-1_{condition}-fit_space-{space}_loo-{heldout_entry['name']}"
        )
        fold_tsv = (
            cv_params_dir / f"{fold_name}_model-norm_stage-iter_desc-prf_params.tsv"
        )

        fold_params_df.to_csv(fold_tsv, sep="\t", index=False)

else:
    raise ValueError(
        f"Unsupported condition '{condition}'. Currently implemented: "
        "'standard', 'violation', 'omission', 'sparse'."
    )

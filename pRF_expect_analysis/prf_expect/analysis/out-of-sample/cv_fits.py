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
    stimulus = PRFStimulus2D(
        np.array(settings["monitor"]["screen_size_cm"][1]),
        settings["monitor"]["screen_distance_cm"],
        prf_dm,
        settings["mri"]["TR"],
        task_lengths=prf_dm.shape[-1],
    )

    # Ensure parameter order matches the fitter's expected array format.
    starting_params_array = Parameters(starting_params, model="norm").to_array()
    subrun_entries = load_standard_subruns(subject, space, psc_dir)

    fold_summaries = []
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
            stimulus=stimulus,
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

else:
    raise ValueError(
        f"Unsupported condition '{condition}'. Currently implemented: 'standard'."
    )

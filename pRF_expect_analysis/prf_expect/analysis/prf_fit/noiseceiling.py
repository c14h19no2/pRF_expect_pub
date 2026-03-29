import argparse
import re
from pathlib import Path

import numpy as np
import scipy.stats

from prf_expect.utils import io


def noise_ceiling(data, bias_correction=True, do_zscore=True):
    # https://github.com/gallantlab/voxelwise_tutorials/blob/main/voxelwise_tutorials/utils.py
    """Compute explainable variance for a set of voxels.

    Parameters
    ----------
    data : array of shape (n_repeats, n_times, n_voxels)
        fMRI responses of the repeated test set.
    bias_correction : bool
        Perform bias correction based on the number of repetitions.
    do_zscore : bool
        z-score the data in time. Only set to False if your data time courses
        are already z-scored.

    Returns
    -------
    ev : array of shape (n_voxels,)
        Explainable variance per voxel.
    """
    if data.ndim != 3:
        raise ValueError(
            "Expected data shape (n_repeats, n_times, n_voxels), " f"got {data.shape}."
        )

    if do_zscore:
        data = scipy.stats.zscore(data, axis=1)

    # Variance per time series.
    mean_var = data.var(axis=1, dtype=np.float64, ddof=1).mean(axis=0)
    var_mean = data.mean(axis=0).var(axis=0, dtype=np.float64, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ev = var_mean / mean_var

    if bias_correction:
        n_repeats = data.shape[0]
        if n_repeats < 2:
            raise ValueError("Bias correction requires at least 2 repeats.")
        ev = ev - (1 - ev) / (n_repeats - 1)

    ev = np.nan_to_num(ev, nan=0.0, posinf=0.0, neginf=0.0)
    return ev


def _align_prediction_to_voxel_time(pred, n_voxels, n_times):
    pred_arr = np.asarray(pred)
    pred_2d = np.squeeze(pred_arr)

    if pred_2d.ndim != 2:
        raise ValueError(f"pred must be reducible to 2D, got shape {pred_arr.shape}")

    target_shape = (n_voxels, n_times)
    if pred_2d.shape == target_shape:
        return pred_2d
    if pred_2d.T.shape == target_shape:
        return pred_2d.T

    raise ValueError(
        "pred shape "
        f"{pred_arr.shape} (squeezed to {pred_2d.shape}) cannot be aligned to "
        f"(n_voxels, n_times) = {target_shape}"
    )


def noise_ceiling_during_stim_present(
    data,
    pred,
    bias_correction=True,
    do_zscore=True,
    stim_threshold=0.01,
):
    """Compute explainable variance using only stimulus-present time points.

    Parameters
    ----------
    data : array of shape (n_repeats, n_times, n_voxels)
        fMRI responses of repeated runs.
    pred : array-like reducible to shape (n_voxels, n_times)
        Prediction time courses used only to define stimulus-present masks.
    bias_correction : bool
        Perform bias correction based on the number of repetitions.
    do_zscore : bool
        z-score the data in time. Only set to False if your data time courses
        are already z-scored.
    stim_threshold : float
        Time points with abs(pred) >= stim_threshold are treated as stimulus-present.

    Returns
    -------
    ev : array of shape (n_voxels,)
        Explainable variance per voxel during stimulus-present time points.
    """
    data_arr = np.asarray(data)
    if data_arr.ndim != 3:
        raise ValueError(
            "Expected data shape (n_repeats, n_times, n_voxels), "
            f"got {data_arr.shape}."
        )

    if stim_threshold < 0:
        raise ValueError(f"stim_threshold must be >= 0, got {stim_threshold}")

    n_repeats, n_times, n_voxels = data_arr.shape
    if bias_correction and n_repeats < 2:
        raise ValueError("Bias correction requires at least 2 repeats.")

    pred_2d = _align_prediction_to_voxel_time(pred, n_voxels, n_times)

    if do_zscore:
        data_arr = scipy.stats.zscore(data_arr, axis=1)

    ev = np.full(n_voxels, np.nan, dtype=float)
    for voxel_index in range(n_voxels):
        mask = np.abs(pred_2d[voxel_index]) >= stim_threshold
        if np.count_nonzero(mask) < 2:
            continue

        voxel_data = data_arr[:, mask, voxel_index]

        mean_var = voxel_data.var(axis=1, dtype=np.float64, ddof=1).mean(axis=0)
        var_mean = voxel_data.mean(axis=0).var(axis=0, dtype=np.float64, ddof=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            voxel_ev = var_mean / mean_var

        if bias_correction:
            voxel_ev = voxel_ev - (1 - voxel_ev) / (n_repeats - 1)

        ev[voxel_index] = voxel_ev

    ev = np.nan_to_num(ev, nan=0.0, posinf=0.0, neginf=0.0)
    return ev


def _extract_run_subrun(file_path: Path):
    match = re.search(r"run-(\d+).*_(subrun\d+)\.npy$", file_path.name)
    if match is None:
        raise ValueError(f"Could not parse run/subrun from file name: {file_path.name}")
    return int(match.group(1)), match.group(2)


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

        # Stored files are time x vertices; transpose to vertices x time.
        left_data = np.load(l_file).T
        right_data = np.load(r_file).T
        subrun_data = np.concatenate([left_data, right_data], axis=0)

        subrun_entries.append(
            {
                "run": run,
                "subrun": subrun,
                "name": f"run-{run}_{subrun}",
                "data": subrun_data,
                "n_left_vertices": left_data.shape[0],
                "n_right_vertices": right_data.shape[0],
            }
        )

    subrun_entries = sorted(subrun_entries, key=lambda x: (x["run"], x["subrun"]))

    if len(subrun_entries) != 9:
        raise ValueError(
            f"Expected 9 pRF subruns for condition='standard', found {len(subrun_entries)}."
        )

    first_shape = subrun_entries[0]["data"].shape
    first_left = subrun_entries[0]["n_left_vertices"]
    first_right = subrun_entries[0]["n_right_vertices"]
    for entry in subrun_entries[1:]:
        if entry["data"].shape != first_shape:
            raise ValueError(
                "All pRF subruns must have the same shape. "
                f"Got {first_shape} and {entry['data'].shape} (for {entry['name']})."
            )
        if (
            entry["n_left_vertices"] != first_left
            or entry["n_right_vertices"] != first_right
        ):
            raise ValueError(
                "Left/right vertex counts are not consistent across subruns."
            )

    return subrun_entries


def main():
    parser = argparse.ArgumentParser(
        description="Compute surface noise ceiling from 9 pRF runs."
    )
    parser.add_argument(
        "subject",
        default=None,
        nargs="?",
        help="subject identifier, e.g., 001",
    )
    args = parser.parse_args()

    subject = args.subject

    settings = io.load_settings()
    data_dir = Path(settings["general"]["data_dir"]) / "data"
    sub_dir = data_dir / "derivatives" / "prf_data" / subject / "ses-1"
    psc_dir = sub_dir / "cut_and_averaged"
    space = settings["mri"]["space"]
    pred_dir = sub_dir / "prf_fits" / "prf_predictions"

    output_dir = sub_dir / "cut_and_averaged" / "noise_ceiling"
    output_dir.mkdir(parents=True, exist_ok=True)

    subrun_entries = load_standard_subruns(subject, space, psc_dir)

    # Stack into (n_repeats, n_times, n_vertices) as required by noise_ceiling.
    repeated_data = np.stack([entry["data"].T for entry in subrun_entries], axis=0)

    ev = noise_ceiling(
        repeated_data,
        bias_correction=True,
        do_zscore=False,
    )

    combined_file = (
        output_dir / f"{subject}_ses-1_task-pRF_space-{space}_desc-noiseceiling.npy"
    )

    np.save(combined_file, ev)

    print(f"Computed noise ceiling from {len(subrun_entries)} pRF runs for {subject}.")
    print(f"Saved: {combined_file}")

    # Load the pRF model prediction for this subject/space.
    pred_file = (
        pred_dir
        / f"{subject}_ses-1_task-pRF_final-fit_space-{space}_model-norm_stage-iter_desc-prf_pred.npy"
    )
    if not pred_file.exists():
        raise FileNotFoundError(
            "Could not find pRF prediction file needed for stimulus-present masking: "
            f"{pred_file}"
        )

    prf_pred = np.load(pred_file)

    ev_during_stim_present = noise_ceiling_during_stim_present(
        repeated_data,
        pred=prf_pred,
        bias_correction=True,
        do_zscore=False,
    )

    combined_file_during_stim_present = (
        output_dir / f"{subject}_ses-1_task-pRF_space-{space}_desc-noiseceiling_during_stim_present.npy"
    )

    np.save(combined_file_during_stim_present, ev_during_stim_present)

    print(f"Computed noise ceiling from {len(subrun_entries)} pRF runs for {subject}.")
    print(f"Saved: {combined_file_during_stim_present}")

if __name__ == "__main__":
    main()

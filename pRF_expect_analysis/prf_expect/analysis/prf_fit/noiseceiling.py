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

    subrun_entries = load_standard_subruns(subject, space, psc_dir)

    # Stack into (n_repeats, n_times, n_vertices) as required by noise_ceiling.
    repeated_data = np.stack([entry["data"].T for entry in subrun_entries], axis=0)

    ev = noise_ceiling(
        repeated_data,
        bias_correction=True,
        do_zscore=False,
    )

    output_dir = sub_dir / "cut_and_averaged" / "noise_ceiling"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_file = (
        output_dir / f"{subject}_ses-1_task-pRF_space-{space}_desc-noiseceiling.npy"
    )

    np.save(combined_file, ev)

    print(f"Computed noise ceiling from {len(subrun_entries)} pRF runs for {subject}.")
    print(f"Saved: {combined_file}")


if __name__ == "__main__":
    main()

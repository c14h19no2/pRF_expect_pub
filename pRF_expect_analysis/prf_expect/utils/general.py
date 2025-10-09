import json
import numpy as np
import glob
import pickle
import os
import math
import re


def extract_run_number(s):
    match = re.search(r"_run-(\d+)", s)
    if match:
        return int(match.group(1))  # Convert the matched string to an integer
    else:
        return None  # In case there's no match


def extract_subrun_number(s):
    match = re.search(r"subrun(\d+)", s)
    if match:
        return int(match.group(1))  # Convert the matched string to an integer
    else:
        return None  # In case there's no match


def perm_diffs(diff_array: np.ndarray, n_perms: int = 100000, baseline: float = 0.0):
    """
    Calculate the permutation differences for a given array of differences.
    """

    # Calculate the mean of the differences
    actual_diff_mean = np.nanmean(diff_array) - baseline

    permuted_signs = -1 + (
        2 * np.random.randint(0, 2, size=(n_perms, len(diff_array)), dtype=np.int8)
    )
    permuted_diffs = permuted_signs * diff_array
    perm_diff_means = np.nanmean(permuted_diffs, axis=1)

    perm_diff_ratio = np.nansum(perm_diff_means > actual_diff_mean) / n_perms

    if perm_diff_ratio > 0.5:
        perm_diff_ratio = 1 - perm_diff_ratio

    print(
        f"Actual difference mean: {actual_diff_mean:.5f} \n",
        f"permuted difference mean: {np.nanmean(perm_diff_means):.5f} \n",
        f"perm_diff_ratio: {perm_diff_ratio:.5f} \n",
        f"permutation_sd: {np.nanstd(perm_diff_means):.5f}",
    )
    return (actual_diff_mean + baseline, np.nanstd(perm_diff_means), perm_diff_ratio)


def rsq_weighted_subdata(data, rsq):
    rsq_masked = np.where(np.isnan(data), 0, rsq)
    nr_nonnan_sub = np.count_nonzero(~np.isnan(data), axis=0)

    data_avg = data * rsq_masked / np.sum(rsq_masked, axis=0)
    data_sub_corrected = data_avg * nr_nonnan_sub
    data_avg = np.nansum(data_avg, axis=0)
    return data_avg, data_sub_corrected


def find_bar_dimensions(array):
    """Function to find the dimensions of a bar in a 2D array."""
    if array.ndim != 2:
        raise ValueError("Array should be 2-dimensional")

    if np.all(array == 0):
        return 0, 0  # No bars found

    if np.all(array == 1):
        return array.shape[1], array.shape[0]  # Full bars

    if np.any(np.all(array == 1, axis=1)):
        # Horizontal bar
        row_indices = np.where(np.all(array == 1, axis=1))[0]
        bar_width = len(row_indices)
        return bar_width, array.shape[0]

    if np.any(np.all(array == 1, axis=0)):
        # Vertical bar
        col_indices = np.where(np.all(array == 1, axis=0))[0]
        bar_height = len(col_indices)
        return array.shape[1], bar_height

    return 0, 0  # No bars found


def extract_frame_number(s):
    match = re.search(r"_Screenshots(\d+)", s)
    if match:
        return int(match.group(1))  # Convert the matched string to an integer
    else:
        return None  # In case there's no match


def verbose(msg, verbose, flush=True, **kwargs):
    if verbose:
        print(msg, flush=flush, **kwargs)

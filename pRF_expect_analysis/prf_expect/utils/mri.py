import os
import numpy as np
import math
import shlex
import subprocess
from nilearn.glm.first_level.hemodynamic_models import _gamma_difference_hrf
from nilearn.glm.first_level.hemodynamic_models import (
    spm_hrf,
    spm_time_derivative,
    spm_dispersion_derivative,
)
from nilearn.surface import load_surf_data
from nilearn.signal import clean, standardize_signal
from nilearn.glm.first_level.design_matrix import create_cosine_drift
from prf_expect.utils.io import get_hemi_func_file


def set_hrf(TR, hrf_deriv, hrf_dsip):
    hrf_params = [1, hrf_deriv, hrf_dsip]
    hrf = np.array(
        [
            np.ones_like(hrf_params[1], dtype="float32")
            * hrf_params[0]
            * spm_hrf(tr=TR, oversampling=1, time_length=40),
            hrf_params[1] * spm_time_derivative(tr=TR, oversampling=1, time_length=40),
            hrf_params[2]
            * spm_dispersion_derivative(tr=TR, oversampling=1, time_length=40),
        ],
        dtype="float32",
    ).sum(axis=0)
    hrf /= hrf.max()
    return hrf


def con_nan_1darray(array):
    """
    Convert an array with nans to an array with nans replaced by the mean of the array.
    """
    nanarray = np.zeros(array.shape[1])
    nanarray[:] = np.nan
    return nanarray


def calculate_tsnr(time_series):  # tsnr for bold data
    """
    Calculate the temporal signal-to-noise ratio (tSNR) for a given time series.
    tSNR is the mean of the time series divided by its standard deviation.
    """
    return np.mean(time_series, axis=0) / np.std(time_series, axis=0)


def get_avg_tSNR(data_dir, subject, space):
    """
    Get the hemifield files for a given subject and space, and load them
    """
    runs_hemi_L = get_hemi_func_file(data_dir, subject, "*", "*", space, "L")
    runs_hemi_R = get_hemi_func_file(data_dir, subject, "*", "*", space, "R")
    tSNR_L = []
    tSNR_R = []
    for file in runs_hemi_L:
        data_tmp = load_surf_data(file)
        tSNR_tmp = calculate_tsnr(data_tmp.T)
        tSNR_L.append(tSNR_tmp)

    for file in runs_hemi_R:
        data_tmp = load_surf_data(file)
        tSNR_tmp = calculate_tsnr(data_tmp.T)
        tSNR_R.append(tSNR_tmp)

    # calculate average tSNR for each hemisphere, then concatenate
    return np.concatenate((np.mean(tSNR_L, axis=0), np.mean(tSNR_R, axis=0)))


def get_QC(data_dir, subject, task, run, space):
    """
    Get the hemifield files for a given subject and space, and load them
    """
    runs_hemi_L = get_hemi_func_file(data_dir, subject, task, run, space, "L")
    runs_hemi_R = get_hemi_func_file(data_dir, subject, task, run, space, "R")

    if len(runs_hemi_L) != 1 or len(runs_hemi_R) != 1:
        raise ValueError(
            f"Expected 1 file for each hemisphere, got {len(runs_hemi_L)} for L and {len(runs_hemi_R)} for R hemispheres"
        )

    data_L = load_surf_data(runs_hemi_L[0]).astype(float)
    mean_L = np.mean(data_L, axis=1)
    std_L = np.std(data_L, axis=1)
    tSNR_L = calculate_tsnr(data_L.T)

    data_R = load_surf_data(runs_hemi_R[0]).astype(float)
    mean_R = np.mean(data_R, axis=1)
    std_R = np.std(data_R, axis=1)
    tSNR_R = calculate_tsnr(data_R.T)

    # calculate average tSNR for each hemisphere, then concatenate
    return (
        np.concatenate((mean_L, mean_R)),
        np.concatenate((std_L, std_R)),
        np.concatenate((tSNR_L, tSNR_R)),
    )


def highpass_dct(
    func,
    lb=0.01,
    TR=0.105,
    modes_to_remove=None,
    remove_constant=False,
):
    """highpass_dct

    Discrete cosine transform (DCT) is a basis set of cosine regressors of varying frequencies up to
    a filter cutoff of a specified number of seconds. Many software use 100s or 128s as a default cutoff,
    but we encourage caution that the filter cutoff isn't too short for your specific experimental
    design. Longer trials will require longer filter cutoffs. See this paper for a more technical
    treatment of using the DCT as a high pass filter in fMRI data analysis
    (https://canlab.github.io/_pages/tutorials/html/high_pass_filtering.html).

    Parameters
    ----------
    func: np.ndarray
        <n_voxels, n_timepoints> representing the functional data to be fitered
    lb: float, optional
        cutoff-frequency for low-pass (default = 0.01 Hz)
    TR: float, optional
        Repetition time of functional run, by default 0.105
    modes_to_remove: int, optional
        Remove first X cosines

    Returns
    ----------
    dct_data: np.ndarray
        array of shape(n_voxels, n_timepoints)
    cosine_drift: np.ndarray
        Cosine drifts of shape(n_scans, n_drifts) plus a constant regressor at cosine_drift[:, -1]

    Notes
    ----------
    * *High-pass* filters remove low-frequency (slow) noise and pass high-freqency signals.
    * Low-pass filters remove high-frequency noise and thus smooth the data.
    * Band-pass filters allow only certain frequencies and filter everything else out
    * Notch filters remove certain frequencies
    # copied from Jurjen's linescanning code
    """

    # Create high-pass filter and clean
    n_vol = func.shape[-1]
    st_ref = 0  # offset frametimes by st_ref * tr
    ft = np.linspace(st_ref * TR, (n_vol + st_ref) * TR, n_vol, endpoint=False)
    hp_set = create_cosine_drift(lb, ft)

    # select modes
    if isinstance(modes_to_remove, int):
        hp_set[:, :modes_to_remove]
    else:
        # remove constant column
        if remove_constant:
            hp_set = hp_set[:, :-1]

    dct_data = clean(func.T, detrend=False, standardize=False, confounds=hp_set).T
    return dct_data, hp_set


def set_workbench_resample_command(
    experiment_base_dir,
    workbench_current_sphere,
    workbench_new_sphere,
    workbench_current_area,
    workbench_new_area,
):
    # set up workbench command
    workbench_resample_command = "".format(
        current_sphere=os.path.join(experiment_base_dir, workbench_current_sphere),
        new_sphere=os.path.join(experiment_base_dir, workbench_new_sphere),
        current_area=os.path.join(experiment_base_dir, workbench_current_area),
        new_area=os.path.join(experiment_base_dir, workbench_new_area),
        metric_in="{metric_in}",
        metric_out="{metric_out}",
    )
    return workbench_resample_command


def split_cii(
    fn, workbench_split_command, workbench_resample_command="", resample=True
):
    # https://github.com/tknapen/hcp_movie/blob/f933c599e2a452817a1b62dbeafe65f14b8cc74a/cfhcpy/surf_utils.py#L117
    wbc_c = workbench_split_command.format(cii=fn, cii_n=fn[:-4])
    subprocess.call(wbc_c, shell=True)

    if resample:
        for hemi in ["L", "R"]:
            this_cmd = workbench_resample_command.format(
                metric_in=fn[:-4] + "_" + hemi + ".gii",
                hemi=hemi,
                metric_out=fn[:-4] + "_fsaverage." + hemi + ".gii",
            )
            plist = shlex.split(this_cmd)
            subprocess.Popen(plist)


def find_nearest(array, value, return_nr=1):
    """find_nearest

    Find the index and value in an array given a value. You can either choose to have 1 item
    (the `closest`) returned, or the 5 nearest items (`return_nr=5`), or everything you're
    interested in (`return_nr="all"`)

    Parameters
    ----------
    array: numpy.ndarray
        array to search in
    value: float
        value to search for in `array`
    return_nr: int, str, optional
        number of elements to return after searching for elements in `array` that are close to
        `value`. Can either be an integer or a string *all*

    Returns
    ----------
    int
        integer representing the index of the element in `array` closest to `value`.

    list
        if `return_nr` > 1, a list of indices will be returned

    numpy.ndarray
        value in `array` at the index closest to `value`
    """

    array = np.asarray(array)

    if return_nr == 1:
        idx = np.nanargmin((np.abs(array - value)))
        return idx, array[idx]
    else:

        # check nan indices
        nans = np.isnan(array)

        # initialize output
        idx = np.full_like(array, np.nan)

        # loop through values in array
        for qq, ii in enumerate(array):

            # don't do anything if value is nan
            if not nans[qq]:
                idx[qq] = np.abs(ii - value)

        # sort
        idx = np.argsort(idx)

        # return everything
        if return_nr == "all":
            idc_list = idx.copy()
        else:
            # return closest X values
            idc_list = idx[:return_nr]

        return idc_list, array[idc_list]


def percent_change(ts, ax, nilearn=False, baseline=20, prf=False, dm=None):
    """percent_change

    Function to convert input data to percent signal change. Two options are current supported: the nilearn method (`nilearn=True`), where the mean of the entire timecourse if subtracted from the timecourse, and the baseline method (`nilearn=False`), where the median of `baseline` is subtracted from the timecourse.

    Parameters
    ----------
    ts: numpy.ndarray
        Array representing the data to be converted to percent signal change. Should be of shape (n_voxels, n_timepoints)
    ax: int
        Axis over which to perform the conversion. If shape (n_voxels, n_timepoints), then ax=1. If shape (n_timepoints, n_voxels), then ax=0.
    nilearn: bool, optional
        Use nilearn method, by default False
    baseline: int, list, np.ndarray optional
        Use custom method where only the median of the baseline (instead of the full timecourse) is subtracted, by default 20. Length should be in `volumes`, not `seconds`. Can also be a list or numpy array (1d) of indices which are to be considered as baseline. The list of indices should be corrected for any deleted volumes at the beginning.

    Returns
    ----------
    numpy.ndarray
        Array with the same size as `ts` (voxels,time), but with percent signal change.

    Raises
    ----------
    ValueError
        If `ax` > 2
    """

    if ts.ndim == 1:
        ts = ts[:, np.newaxis]
        ax = 0

    else:
        if nilearn:
            if ax == 0:
                psc = standardize_signal(ts, standardize="psc")
            else:
                psc = standardize_signal(ts.T, standardize="psc").T
        else:

            # first step of PSC; set NaNs to zero if dividing by 0 (in case of crappy timecourses)
            ts_m = ts * np.expand_dims(np.nan_to_num((100 / np.mean(ts, axis=ax))), ax)

            # get median of baseline
            if isinstance(baseline, np.ndarray):
                baseline = list(baseline)

            if ax == 0:
                if isinstance(baseline, list):
                    median_baseline = np.median(ts_m[baseline, :], axis=0)
                else:
                    median_baseline = np.median(ts_m[:baseline, :], axis=0)
            elif ax == 1:
                if isinstance(baseline, list):
                    median_baseline = np.median(ts_m[:, baseline], axis=1)
                else:
                    median_baseline = np.median(ts_m[:, :baseline], axis=1)
            else:
                raise ValueError("ax must be 0 or 1")

            # subtract
            psc = ts_m - np.expand_dims(median_baseline, ax)

        return psc

import json
import os
import glob
from nilearn.surface import load_surf_data
import numpy as np
import nibabel as nb
import yaml


# load json file
def load_jsons(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def load_yamls(yaml_file):
    with open(yaml_file, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def load_settings(settings_file=None):
    if settings_file is None:
        settings_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "settings.yml"
        )
    print(f"Loading settings from {settings_file}")
    return load_yamls(settings_file)


def load_cifti(data_path):
    dat = nb.cifti2.load(data_path)
    tseries_raw = dat.get_fdata()
    return tseries_raw, dat


def surf_data_from_cifti(data, axis, surf_name):
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    for (
        name,
        data_indices,
        model,
    ) in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:  # Just looking for a surface
            data = data.T[
                data_indices
            ]  # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex  # Generally 1-N, except medial wall vertices
            surf_data = np.zeros(
                (vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype
            )
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")


def volume_from_cifti(data, axis, header):
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    data = data.T[axis.volume_mask]  # Assume brainmodels axis is last, move it to front
    volmask = axis.volume_mask  # Which indices on this axis are for voxels?
    vox_indices = tuple(
        axis.voxel[axis.volume_mask].T
    )  # ([x0, x1, ...], [y0, ...], [z0, ...])
    vol_data = np.zeros(
        axis.volume_shape + data.shape[1:],  # Volume + any extra dimensions
        dtype=data.dtype,
    )
    vol_data[vox_indices] = data  # "Fancy indexing"
    return nb.Nifti2Image(vol_data, axis.affine, header=header)


def decompose_cifti(img):
    data = img.get_fdata(dtype=np.float32)
    print(data.shape)
    brain_models = img.header.get_axis(1)  # Assume we know this
    return (
        volume_from_cifti(data, brain_models, header=img.nifti_header),
        surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT"),
        surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT"),
    )


def write_newcifti(filename, old_cifti, data_arr):
    """
    Saves a CIFTI file that has a new size of timepoints
    Parameters
    ----------
    filename : str
        name of output CIFTI file
    old_cifti : CIFTI file
        previous nibabel.cifti2.cifti2.Cifti2Image
    data_arr : array
        data to be stored as vector or matrix (shape n_timepoints x voxels)

    Example:
    datvol = nb.load('/scratch/2021/nprf_ss/derivatives/fmriprep/sub-01/ses-01/func/sub-01_ses-01_task-prf_run-10_space-fsLR_den-170k_bold.dtseries.nii')
    dat = np.asanyarray(datvol.dataobj)
    write_newcifti('/tank/klundert/check2.nii', datvol, dat[10:120])

    This function is contributed by Ron van de Klundert.
    """
    from nibabel import cifti2

    start = old_cifti.header.get_axis(0).start
    step = old_cifti.header.get_axis(0).step
    brain_model = old_cifti.header.get_axis(1)
    size = data_arr.shape[0]
    series = cifti2.SeriesAxis(start, step, size)
    brain_model = old_cifti.header.get_axis(1)

    newheader = cifti2.Cifti2Header.from_axes((series, brain_model))
    img = cifti2.Cifti2Image(data_arr, newheader)

    os.makedirs(os.path.split(filename)[0], exist_ok=True)
    img.to_filename(filename)


def get_hemi_func_file(
    data_dir, subject: str, task: str, run: str, space: str, LR: str
):
    files = sorted(
        glob.glob(
            f"{data_dir}/derivatives/fmriprep/{subject}/ses-1/func/*task-{task}*_run-{run}*space-{space}_hemi-{LR}.func.gii"
        )
    )
    return files

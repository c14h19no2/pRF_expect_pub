# import functions
import numpy as np
import pandas as pd
import json
import glob
import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize, rescale
from prf_expect.utils import io
from prf_expect.utils.general import find_bar_dimensions, extract_frame_number

settings = io.load_settings()

# Define
subjects = settings["general"]["subject_list"]
data_dir = os.path.join(settings["general"]["data_dir"], "data")
runs = settings["design"]["runs"]
pRF_runs = settings["design"]["pRF_runs"]
PE_runs = settings["design"]["PE_runs"]
runs_per_task = settings["design"]["runs_per_task"]

resample_size = settings["dm"]["resample_size"]


# ### **Make the design matrix and save to .npy files**

# Make the dm for the pRF runs, by only taking one of the three subruns (from sub-001, run-1)
# and saving this for all subjects, as the dms for these are all the same
# load the screenshots and take the mean colour
screenshot_list_undorted = sorted(
    glob.glob(f"{data_dir}/sourcedata/sub-001/ses-1/logs/*run-1_Logs/*_Screenshots/*")
)
screenshot_frameindex = [
    extract_frame_number(screenshot) for screenshot in screenshot_list_undorted
]
sorted_frameindex = sorted(
    range(len(screenshot_frameindex)), key=lambda k: screenshot_frameindex[k]
)
screenshot_list_sorted = [screenshot_list_undorted[i] for i in sorted_frameindex]

dm = np.array([mpimg.imread(img) for img in screenshot_list_sorted])
mean_colour_array = np.mean(dm, axis=-1)
print(f"Shape of the original dm is {mean_colour_array.shape}")

# crop the image to remove the overhang
h, w = mean_colour_array[1].shape
overhang = (w - h) // 2
dm_without_overhang = mean_colour_array[:, :, overhang:-overhang]
bool_array = np.where(dm_without_overhang != np.median(dm_without_overhang), 1, 0)

# Take out fixdot
fixdot_mask = np.where(bool_array[0] == 1)
bool_array[:, fixdot_mask[0], fixdot_mask[1]] = 0

print(f"Shape of the dm without overhang is {bool_array.shape}")

resized_dm = resize(
    bool_array.astype(float),
    resample_size,
    anti_aliasing=True,
    order=1,
    mode="reflect",
    preserve_range=True,
)

# Finalizing the dm by only taking the max value of each matrix
# The transform is taken to be able to use the dm in the pRF analysis
DM = np.where(resized_dm > 0.3, 1, 0)
final_pRF_dm = np.transpose(DM, axes=(1, 2, 0))
print(f"Shape of the final dm is {final_pRF_dm.shape}")

# cut the pRF runs and save the dm
# The pRF runs are cut, as the mean over three subruns is taken for these runs
final_pRF_dm = final_pRF_dm[:, :, :125]
final_pRF_dm = np.flipud(final_pRF_dm)
print(f"Shape of the final pRF dm is {final_pRF_dm.shape}")

# now save this dm for all subjects and all (sub-)runs
for subject in subjects:
    output_dir = f"{data_dir}/derivatives/prf_data/{subject}/ses-1/dms/"
    os.makedirs(output_dir, exist_ok=True)
    for run in runs_per_task:
        np.save(f"{output_dir}dm_task-pRF_{run}_uncut.npy", final_pRF_dm)

for subject in subjects:
    output_dir = f"{data_dir}/derivatives/prf_data/{subject}/ses-1/dms/"
    for run in PE_runs:

        # load the screenshots and take the mean colour
        screenshot_list_undorted = sorted(
            glob.glob(
                f"{data_dir}/sourcedata/{subject}/ses-1/logs/*{run}_Logs/*_Screenshots/*"
            )
        )
        screenshot_frameindex = [
            extract_frame_number(screenshot) for screenshot in screenshot_list_undorted
        ]
        sorted_frameindex = sorted(
            range(len(screenshot_frameindex)), key=lambda k: screenshot_frameindex[k]
        )
        screenshot_list_sorted = [
            screenshot_list_undorted[i] for i in sorted_frameindex
        ]
        dm = np.array([mpimg.imread(img) for img in screenshot_list_sorted])
        mean_colour_array = np.mean(dm, axis=-1)
        print(f"Shape of the original dm is {mean_colour_array.shape}")

        # crop the image to remove the overhang
        h, w = mean_colour_array[1].shape
        overhang = (w - h) // 2
        dm_without_overhang = mean_colour_array[:, :, overhang:-overhang]
        bool_array = np.where(
            dm_without_overhang != np.median(dm_without_overhang), 1, 0
        )

        # Take out fixdot
        fixdot_mask = np.where(bool_array[0] == 1)
        bool_array[:, fixdot_mask[0], fixdot_mask[1]] = 0

        print(f"Shape of the dm without overhang is {bool_array.shape}")

        # resize the dm to the desired size
        resized_dm = resize(
            bool_array.astype(float),
            resample_size,
            anti_aliasing=True,
            order=1,
            mode="reflect",
            preserve_range=True,
        )
        DM = np.where(resized_dm > 0.3, 1, 0)
        final_PE_dm_sub = np.transpose(DM, axes=(1, 2, 0))

        # flip the dm so it is in the right orientation
        final_PE_dm_sub = np.flipud(final_PE_dm_sub)
        if run == "run-2":
            np.save(f"{output_dir}dm_task-PE_run-1_uncut.npy", final_PE_dm_sub)
            print(f"now saving PE dm for run 2 for subject {subject}")
        elif run == "run-4":
            np.save(f"{output_dir}dm_task-PE_run-2_uncut.npy", final_PE_dm_sub)
            print(f"now saving PE dm for run 4 for subject {subject}")
        elif run == "run-6":
            np.save(f"{output_dir}dm_task-PE_run-3_uncut.npy", final_PE_dm_sub)
            print(f"now saving PE dm for run 6 for subject {subject}")

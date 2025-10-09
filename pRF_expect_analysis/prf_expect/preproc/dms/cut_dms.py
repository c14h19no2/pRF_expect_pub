
# import functions
import numpy as np
import os
from prf_expect.utils import io

opj = os.path.join

# Load settings
settings = io.load_settings()

# Define
subjects = settings["general"]["subject_list"]
data_dir = os.path.join(settings["general"]["data_dir"], "data")
runs = settings["design"]["runs"]
pRF_runs = settings["design"]["pRF_runs"]
PE_runs = settings["design"]["PE_runs"]
runs_per_task = settings["design"]["runs_per_task"]

resample_size = settings["dm"]["resample_size"]
dm_coil_cutoff = settings["dm"]["coil_cutoff"]

# now save this dm for all subjects and all (sub-)runs
for subject in subjects:
    sub_screen_cut = int(round(dm_coil_cutoff[subject] * resample_size[2]))
    output_dir = f"{data_dir}/derivatives/prf_data/{subject}/ses-1/dms/"
    os.makedirs(output_dir, exist_ok=True)
    for run in runs_per_task:
        final_pRF_dm = np.load(f"{output_dir}dm_task-pRF_{run}_uncut.npy")
        print(f"Cut off for subject {subject} is {sub_screen_cut}")
        final_pRF_dm = final_pRF_dm.copy()
        final_pRF_dm[:sub_screen_cut, :, :] = 0
        np.save(f"{output_dir}dm_task-pRF_{run}.npy", final_pRF_dm)

        final_PE_dm = np.load(f"{output_dir}dm_task-PE_{run}_uncut.npy")
        final_PE_dm_sub = final_PE_dm.copy()
        final_PE_dm_sub[:sub_screen_cut, :, :] = 0

        # # cut the pRF runs and save the dm
        print(f"Shape of the final PE dm is {final_PE_dm_sub.shape}")
        np.save(f"{output_dir}dm_task-PE_{run}.npy", final_PE_dm_sub)
        print(f"now saving PE dm for {run} for subject {subject}")

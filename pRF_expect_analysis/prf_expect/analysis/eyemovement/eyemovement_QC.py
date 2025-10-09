# %%
import numpy as np
import scipy as sp
import seaborn as sn
from scipy.stats import gaussian_kde
import matplotlib.pylab as plt
import os
import hedfpy

from hedfpy.EDFOperator import EDFOperator
from hedfpy.HDFEyeOperator import HDFEyeOperator
from hedfpy.EyeSignalOperator import EyeSignalOperator, detect_saccade_from_data

sn.set(style="ticks")
import warnings

warnings.filterwarnings("ignore")

from PIL import Image

# %%
os.environ["PATH"] += os.pathsep + os.path.expanduser("/tank/shared/software/")


# %%
def gaze_over_time(x, y):
    xy_data = np.array([np.array(x).squeeze(), np.array(y).squeeze()]).T
    vel = np.diff(xy_data, axis=0)
    vel_norm = np.linalg.norm(vel, axis=1)

    saccades = detect_saccade_from_data(
        xy_data=xy_data, l=6, minimum_saccade_duration=0.0025
    )

    main_sequence = np.array(
        [
            [
                sacc["peak_velocity"],
                sacc["expanded_amplitude"],
                sacc["expanded_duration"],
            ]
            for sacc in saccades
        ]
    )

    f = plt.figure(figsize=(14, 5))
    s = f.add_subplot(121)
    plt.plot(np.array(x) - 640)
    plt.plot(np.array(y) - 480)
    s.set_ylim([-640, 640])
    sn.despine(ax=s, offset=10)
    s.set_title("gaze x and y over time")
    for sacc in saccades:
        s.axvline(sacc["expanded_start_time"], lw=0.5, color="k")
        s.axvline(sacc["expanded_end_time"], lw=0.5, color="r")
    s.set_xlabel("time")
    s.set_ylabel("pixels on screen")

    s = f.add_subplot(222)
    plt.scatter(main_sequence[:, 0], main_sequence[:, 1], c="r")
    sn.despine(ax=s, offset=10)
    s.set_title("main sequence")
    s.set_xlabel("peak velocity")
    s.set_ylabel("amplitude")

    s = f.add_subplot(224)
    for sacc in saccades:
        plt.plot(
            vel_norm[sacc["expanded_start_time"] : sacc["expanded_end_time"]],
            "k",
            lw=1,
            alpha=0.5,
        )
    sn.despine(ax=s, offset=10)
    s.set_title("velocity profiles")
    plt.tight_layout()
    s.set_xlabel("velocity")
    s.set_ylabel("time [ms]")


# %%
def con_eye_data(subject, run, alias="prf"):
    edf_file = f"/tank/shared/2023/prfexpect/data/sourcedata/{subject}/ses-1/logs/{subject}_ses-1_run-{run}_Logs/{subject}_ses-1_run-{run}.edf"
    dir_path = os.path.dirname(edf_file)
    # get file name without extension
    file_name = os.path.basename(edf_file)
    file_name = os.path.splitext(file_name)[0] + ".h5"
    file_path = os.path.join(dir_path, file_name)
    print(file_path)

    low_pass_pupil_f, high_pass_pupil_f = 6.0, 0.01

    file_folder = os.path.dirname(edf_file)
    # initialize the hdfeyeoperator
    ho = HDFEyeOperator(os.path.splitext(edf_file)[0] + ".h5")
    # insert the edf file contents only when the h5 is not present.
    if not os.path.isfile(os.path.splitext(edf_file)[0] + ".h5"):
        ho.add_edf_file(edf_file)
        ho.edf_message_data_to_hdf(alias=alias)
        ho.edf_gaze_data_to_hdf(
            alias=alias, pupil_hp=high_pass_pupil_f, pupil_lp=low_pass_pupil_f
        )
    return ho


# %%
alias = "prf"

# %%
# check metadata of asc file
for subject in [
    "sub-001",
    "sub-002",
    "sub-004",
    "sub-005",
    "sub-007",
    "sub-009",
    "sub-012",
]:
    for run in ["1", "2", "3", "4", "5", "6"]:
        file_path = f"/tank/shared/2023/prfexpect/data/sourcedata/{subject}/ses-1/logs/{subject}_ses-1_run-{run}_Logs/{subject}_ses-1_run-{run}.msg"
        print(f"{subject} run {run}")
        with open(file_path, "r") as file:
            for i in range(2):  # Read the first 10 lines
                line = file.readline()
                if "DATE" in line:
                    print(line.strip())
    print("--------------------")

# %%
for subject in [
    "sub-001",
    "sub-002",
    "sub-004",
    "sub-005",
    "sub-007",
    "sub-009",
    "sub-012",
]:
    for run in ["1", "2", "3", "4", "5", "6"]:
        ho = con_eye_data(subject, run)
        time_period = [
            ho.trial_properties(alias)["trial_start_EL_timestamp"].iloc[0],
            ho.trial_properties(alias)["trial_start_EL_timestamp"].iloc[-1],
        ]
        tracked_eye = ho.block_properties(alias)["eye_recorded"][0]
        gaze_data = [
            ho.signal_during_period(
                time_period=time_period,
                alias=alias,
                signal="gaze_x_int",
                requested_eye=tracked_eye,
            ),
            1024
            - ho.signal_during_period(
                time_period=time_period,
                alias=alias,
                signal="gaze_y_int",
                requested_eye=tracked_eye,
            ),
        ]
        plt.figure(figsize=(14, 5))
        # plot the gaze data over time, x and y separately
        if run in ["1", "3", "5"]:
            task = "pRF"
        elif run in ["2", "4", "6"]:
            task = "PE"
        plt.plot(gaze_data[0])
        plt.plot(gaze_data[1])
        plt.title(f"{subject}, run: {run}, task: {task}, tracked eye: {tracked_eye}")
        plt.ylim([0, 1500])
        plt.show()

# %% [markdown]
# ## Now we need to find the relationship between eyemovement and the stimulus/surprise

# %%
# def kde(x,y,image):
#     x_pixels, y_pixels = np.meshgrid(np.arange(0,1280,16),np.arange(0,1024,16))
#     pixel_coordinates = np.array([x_pixels, y_pixels]).reshape((2,-1))
#     gaze = np.array([x.squeeze(),y.squeeze()])
#     gaze_kde = gaussian_kde(gaze)

#     density = gaze_kde.evaluate(pixel_coordinates)

#     # plot these things
#     f = plt.figure(figsize = (14,5))
#     s = f.add_subplot(111)
#     plt.plot(x,y)
#     s.set_xlim([0,1280])
#     s.set_ylim([0,960])
#     sn.despine(ax=s, offset=10)
#     plt.imshow(density.reshape(x_pixels.shape)[::-1,:], alpha=0.6, extent=[0,1280,0,1024], cmap='jet')
#     plt.plot(x, y, 'w', lw=3)

#     s = f.add_subplot(111)
#     plt.imshow(254-image.squeeze()[::-1,::-1], alpha=0.75, clim=[0,512])
#     plt.plot(x, y,'k', lw=3)
#     s.set_xlim([0,1280])
#     s.set_ylim([0,960])
#     sn.despine(ax=s, offset=10)


# for i in range(13):
#     kde(gaze_data[i][0], gaze_data[i][1], images[i])

# %%
# set up black to grey colormap
cmap = plt.cm.gray
# crop the gray colormap to only include the black to grey range
cmap = cmap(np.linspace(0, 0.8, 256))
# create a new colormap with the cropped range
cmap = plt.cm.colors.ListedColormap(cmap)

# %%
ho.trial_properties(alias)["trial_start_EL_timestamp"]

# %%
ho.trial_properties(alias)["trial_start_EL_timestamp"]

# %%
str2int = lambda x: int(x)

# %%
gaze_y_range = [0, 1080]
gaze_x_range = [0, 1920]
for subject in [
    "sub-001",
    "sub-002",
    "sub-004",
    "sub-005",
    "sub-007",
    "sub-009",
    "sub-012",
]:
    os.makedirs(
        f"/tank/shared/2023/prfexpect/data/derivatives/figures/{subject}/ses-1/eyemovement",
        exist_ok=True,
    )
    for run in ["1", "2", "3", "4", "5", "6"]:
        # Determine the task based on the run number
        if run in ["1", "3", "5"]:
            task = "pRF"
            task_name = "pRF"
            task_run = str(int((str2int(run) + 1) / 2))
        elif run in ["2", "4", "6"]:
            task = "PE"
            task_name = "PE"
            task_run = str(int((str2int(run)) / 2))
        dm_dir = (
            "/tank/shared/2023/prfexpect/data/derivatives/prf_data/{subject}/ses-1/dms/"
        )
        dm_fn = os.path.join(
            dm_dir.format(subject=subject), f"dm_task-{task_name}_run-0{task_run}.npy"
        )
        print("Loading ", dm_fn)
        dm_pRF = np.load(dm_fn)
        if task == "pRF":
            # repeat the dm_pRF 3 times to match the number of trials
            dm_pRF = np.tile(dm_pRF, (1, 1, 3))
        ho = con_eye_data(subject, run)

        tracked_eye = ho.block_properties(alias)["eye_recorded"][0]
        figure, axes = plt.subplots(15, 25, figsize=(15, 9), dpi=300)

        for i, ax in enumerate(axes.flatten()):
            if i < 374:
                time_period = [
                    ho.trial_properties(alias)["trial_start_EL_timestamp"].iloc[i],
                    ho.trial_properties(alias)["trial_start_EL_timestamp"].iloc[i + 1],
                ]
            else:
                time_period = [
                    ho.trial_properties(alias)["trial_start_EL_timestamp"].iloc[i],
                    -1,
                ]

            # Get gaze data for the time period
            gaze_data_x = (
                ho.signal_during_period(
                    time_period=time_period,
                    alias=alias,
                    signal="gaze_x_int",
                    requested_eye=tracked_eye,
                ).squeeze()
                - (gaze_x_range[1] - gaze_y_range[1]) / 2
            )
            gaze_data_y = (
                gaze_y_range[1]
                - ho.signal_during_period(
                    time_period=time_period,
                    alias=alias,
                    signal="gaze_y_int",
                    requested_eye=tracked_eye,
                ).squeeze()
            )

            # Plot the dm_pRF image
            # upscale the matrix (dm_pRF[:, :, i]) to match the gaze data (960*960)
            image = dm_pRF[:, :, i]
            image = np.kron(
                image,
                np.ones(
                    (
                        gaze_y_range[1] // image.shape[0],
                        gaze_y_range[1] // image.shape[1],
                    )
                ),
            )
            # flip up-down
            image = np.flipud(image)
            ax.matshow(image, cmap=cmap, alpha=0.75)

            # Plot the gaze data as a red line
            ax.plot(gaze_data_x, gaze_data_y, "r", lw=0.5)

            # add a crosshair to the center of the image
            ax.plot(
                [gaze_y_range[1] / 2, gaze_y_range[1] / 2], gaze_y_range, "b", lw=0.5
            )
            ax.plot(
                gaze_y_range, [gaze_y_range[1] / 2, gaze_y_range[1] / 2], "b", lw=0.5
            )

            # Remove ticks and turn off the axis
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_xlim(gaze_y_range)
            ax.set_ylim(gaze_y_range)
            # ax.axis('off')
            # reduce the axis line width
            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(0.5)

        # plt.suptitle(f"{subject}, run: {run}, task: {task}, tracked eye: {tracked_eye}")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        plt.savefig(
            f"/tank/shared/2023/prfexpect/data/derivatives/figures/{subject}/ses-1/eyemovement/{subject}_ses-1_run-{run}_gaze.pdf"
        )
        plt.show()
        plt.close()

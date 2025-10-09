import os
import numpy as np
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, Norm_Iso2DGaussianModel
from prfpy.fit import Extend_Iso2DGaussianFitter, Norm_Iso2DGaussianFitter
from prf_expect.utils import io, general, mri
import pandas as pd
from time import gmtime, strftime, time

# A bunch of functions are adpated from Jurjen's linescanning pipeline


class PRFModel:
    def __init__(self):

        # Load the settings
        self.settings = io.load_settings()
        self.design_matrix = None
        self.data = None

    def get_dm(self, design_matrix: np.ndarray):
        # Define the stimulus
        self.design_matrix = design_matrix
        # print(self.design_matrix.shape)
        self.stimulus = PRFStimulus2D(
            np.array(self.settings["monitor"]["screen_size_cm"][1]),
            self.settings["monitor"]["screen_distance_cm"],
            self.design_matrix,
            self.settings["mri"]["TR"],
            task_lengths=design_matrix.shape[-1],
        )
        # print(self.stimulus.x_coordinates.shape)

    def get_data(self, data: np.ndarray):
        # Define the data
        self.data = data

    def make_model(self, model):
        if model == "gauss":
            self.gauss_model = Iso2DGaussianModel(
                self.stimulus, self.settings["prf"]["hrf"]
            )
        elif model == "norm":
            self.norm_model = Norm_Iso2DGaussianModel(
                self.stimulus,
                hrf=self.settings["prf"]["hrf"]["pars"],
                filter_predictions=False,
                normalize_RFs=False,
                normalize_hrf=True,
            )

    def fit(
        self,
        rsq_threshold=0.1,
        model="norm",
        n_jobs=1,
        verbose=False,
        output_dir=None,
        output_base=None,
        constraints=["tc", "tc"],
    ):
        if output_dir is None:
            raise ValueError("Output directory (output_dir) must be specified")

        for cst in constraints:
            if cst not in [
                "bgfs",
                "tc",
            ]:
                raise ValueError(
                    f"Constraint '{cst}' not recognized, it must be 'bgfs' or 'tc'"
                )
        self.make_model(model)

        gauss_constr, norm_constr = [
            [] if cst == "tc" else None if cst == "bgfs" else print("emm")
            for cst in constraints
        ]
        grid_nr = self.settings["prf"]["grid_nr"]
        ss = self.stimulus.screen_size_degrees
        max_ecc_size = ss / 2.0
        size_grid = max_ecc_size * np.linspace(0.25, 1, grid_nr) ** 2
        ecc_grid = max_ecc_size * np.linspace(0.1, 1, grid_nr) ** 2
        polar_grid = np.linspace(0, 2 * np.pi, grid_nr)

        coord_bounds = (-1.5 * max_ecc_size, 1.5 * max_ecc_size)
        prf_size = (0.2, 1.5 * ss)

        hrf_1_grid = np.linspace(0, 10, 10)
        hrf_2_grid = np.linspace(0, 0, 1)

        standard_bounds = [
            coord_bounds,  # x
            coord_bounds,  # y
            prf_size,  # prf size
            self.settings["prf"]["prf_ampl"],  # prf amplitude
            self.settings["prf"]["bold_bsl"],  # bold baseline
        ]

        tsv_gauss_name = f"{output_base}_model-gauss_stage-iter_desc-prf_params.tsv"
        tsv_gauss_path = os.path.join(output_dir, tsv_gauss_name)
        tsv_name = f"{output_base}_model-{model}_stage-iter_desc-prf_params.tsv"
        tsv_path = os.path.join(output_dir, tsv_name)

        # Fit the model
        if model == "gauss":
            # gasussian fitter
            self.gauss_fitter = Extend_Iso2DGaussianFitter(
                self.gauss_model, self.data, n_jobs=n_jobs
            )

            general.verbose(
                f"Gauss model grid fit started at {strftime('%Y-%m-%d %H:%M:%S', gmtime())}",
                verbose=True,
            )
            start_time = time()

            self.gauss_fitter.grid_fit(
                ecc_grid=ecc_grid,
                polar_grid=polar_grid,
                size_grid=size_grid,
                verbose=verbose,
                n_batches=n_jobs,
                fixed_grid_baseline=self.settings["prf"]["fixed_grid_baseline"],
                grid_bounds=[tuple(self.settings["prf"]["grid_bounds"])],
                hrf_1_grid=hrf_1_grid,
                hrf_2_grid=hrf_2_grid,
            )

            general.verbose(
                f"Gaussian model grid fit fitted at {strftime('%Y-%m-%d %H:%M:%S', gmtime())}, "
                + f"took {(time() - start_time)/60:.2f} minutes",
                verbose=True,
            )

            start_time = time()

            gauss_bounds = standard_bounds.copy()
            gauss_bounds += [
                self.settings["prf"]["hrf"]["deriv_bound"],
                self.settings["prf"]["hrf"]["disp_bound"],
            ]

            self.gauss_fitter.iterative_fit(
                rsq_threshold,
                verbose=verbose,
                bounds=gauss_bounds,
                constraints=gauss_constr,  # will use method='trust-constr'
            )
            general.verbose(
                f"Gaussian model iterative fitted at {strftime('%Y-%m-%d %H:%M:%S', gmtime())}, "
                + f"took {(time() - start_time)/60:.2f} minutes",
                verbose=True,
            )

            # save the parameters
            gauss_par = Parameters(
                self.gauss_fitter.iterative_search_params, model="gauss"
            )
            tsv_name = f"{output_base}_model-gauss_stage-iter_desc-prf_params.tsv"
            gf_par_df = gauss_par.to_df()
            tsv_path = os.path.join(output_dir, tsv_name)
            gf_par_df.to_csv(tsv_path, sep="\t", index=False)

        if model == "norm":
            # try to load the gaussian parameters, if not possible, fit them
            if not os.path.exists(tsv_gauss_path):
                self.fit(
                    rsq_threshold=rsq_threshold,
                    model="gauss",
                    n_jobs=n_jobs,
                    verbose=verbose,
                    output_dir=output_dir,
                    output_base=output_base,
                )
                self.norm_fitter = Norm_Iso2DGaussianFitter(
                    self.norm_model,
                    self.data,
                    n_jobs=n_jobs,
                    previous_gaussian_fitter=self.gauss_fitter,
                )
            else:
                general.verbose(
                    f"Loading previous gaussian parameters from {tsv_gauss_path} at {strftime('%Y-%m-%d %H:%M:%S', gmtime())}",
                    verbose=True,
                )
                starting_params_grid = self.load_params(
                    tsv_gauss_path, model="gauss", return_pars=True
                )

                self.gauss_model = Iso2DGaussianModel(
                    self.stimulus, self.settings["prf"]["hrf"]
                )
                self.gauss_fitter = Extend_Iso2DGaussianFitter(
                    self.gauss_model, self.data, n_jobs=n_jobs
                )
                self.gauss_fitter.gridsearch_params = starting_params_grid.copy()
                self.gauss_fitter.iterative_search_params = starting_params_grid.copy()
                self.norm_fitter = Norm_Iso2DGaussianFitter(
                    self.norm_model,
                    self.data,
                    n_jobs=n_jobs,
                    previous_gaussian_fitter=self.gauss_fitter,
                )

            # grid fit
            start_time = time()
            general.verbose(
                f"DN model grid fit started at {strftime('%Y-%m-%d %H:%M:%S', gmtime())}",
                verbose=True,
            )
            self.norm_fitter.grid_fit(
                surround_amplitude_grid=self.settings["prf"]["norm"][
                    "surround_amplitude_grid"
                ],
                surround_size_grid=self.settings["prf"]["norm"]["surround_size_grid"],
                neural_baseline_grid=self.settings["prf"]["norm"][
                    "neural_baseline_grid"
                ],
                surround_baseline_grid=self.settings["prf"]["norm"][
                    "surround_baseline_grid"
                ],
                hrf_1_grid=hrf_1_grid,
                hrf_2_grid=hrf_2_grid,
                n_batches=n_jobs,
            )
            general.verbose(
                f"DN model grid fit finished at {strftime('%Y-%m-%d %H:%M:%S', gmtime())}, "
                + f"took {(time() - start_time)/60:.2f} minutes",
                verbose=True,
            )

            # iterative fit
            general.verbose(
                f"DN model iterative fit started at {strftime('%Y-%m-%d %H:%M:%S', gmtime())}",
                verbose=True,
            )
            start_time = time()
            norm_bounds = [
                coord_bounds,  # x
                coord_bounds,  # y
                prf_size,  # prf size
                self.settings["prf"]["prf_ampl"],  # prf amplitude
                self.settings["prf"]["bold_bsl"],  # bold baseline
                self.settings["prf"]["norm"][
                    "surround_amplitude_bound"
                ],  # surround amplitude
                (self.settings["prf"]["eps"], 3 * ss),  # surround size
                self.settings["prf"]["norm"][
                    "neural_baseline_bound"
                ],  # neural baseline
                self.settings["prf"]["norm"]["surround_baseline_bound"],
            ]  # surround baseline
            norm_bounds += [
                self.settings["prf"]["hrf"]["deriv_bound"],
                self.settings["prf"]["hrf"]["disp_bound"],
            ]
            self.norm_fitter.iterative_fit(
                rsq_threshold,
                verbose=verbose,
                bounds=norm_bounds,
                constraints=norm_constr,
            )
            general.verbose(
                f"DN model iterative fit finished at {strftime('%Y-%m-%d %H:%M:%S', gmtime())},"
                + f"took {(time() - start_time)/60:.2f} minutes",
                verbose=True,
            )
            DN_par = Parameters(self.norm_fitter.iterative_search_params, model="norm")
            DN_par_df = DN_par.to_df()
            DN_par_df.to_csv(tsv_path, sep="\t", index=False)

    def load_params(
        self,
        params_file,
        model="gauss",
        stage="iter",
        hemi=None,
        return_pars=False,
    ):
        if isinstance(params_file, str):
            if params_file.endswith("npy"):
                params = np.load(params_file)
            elif params_file.endswith("tsv"):
                params = pd.read_csv(params_file, sep="\t")
                # convert to numpy array
                params = Parameters(params, model=model).to_array()
        elif isinstance(params_file, np.ndarray):
            params = params_file.copy()
        elif isinstance(params_file, list):
            params = np.array(params_file)
        elif isinstance(params_file, pd.DataFrame):
            if hemi:
                # got normalization parameter file
                params = np.array(
                    (
                        params_file["x"][hemi],
                        params_file["y"][hemi],
                        params_file["prf_size"][hemi],
                        params_file["A"][hemi],
                        params_file["bold_bsl"][hemi],
                        params_file["C"][hemi],
                        params_file["surr_size"][hemi],
                        params_file["B"][hemi],
                        params_file["D"][hemi],
                        params_file["r2"][hemi],
                    )
                )
            else:
                params = Parameters(params_file, model=model).to_array()

        else:
            raise ValueError(
                f"Unrecognized input type for '{params_file}' ({type(params_file)})"
            )

        general.verbose(
            f"Inserting parameters from {type(params_file)} as '{model}_{stage}' in {self}",
            True,
        )
        setattr(self, f"{model}_{stage}", params)

        if return_pars:
            return params

    def make_predictions(self, vox_nr=None, model="norm", stage="iter"):
        self.make_model(model)

        try:
            use_model = getattr(self, f"{model}_model")
        except:
            raise ValueError(
                f"{self}-object does not have attribute '{self.model}_model'"
            )

        if hasattr(self, f"{model}_{stage}"):
            params = getattr(self, f"{model}_{stage}")
            if params.ndim == 1:
                params = params[np.newaxis, ...]

            if vox_nr != None:
                if vox_nr == "best":
                    vox, _ = mri.find_nearest(params[..., -1], np.amax(params[..., -1]))
                else:
                    vox = vox_nr

                params = params[vox, ...]
                pred = use_model.return_prediction(*params[:-1]).T
                return pred, params, vox
            else:
                predictions = []
                for vox in range(params.shape[0]):
                    pars = params[vox, ...]
                    predictions.append(use_model.return_prediction(*pars[:-1]).T)

                return np.squeeze(np.array(predictions), axis=-1)

        else:
            raise ValueError(f"Could not find {stage} parameters for {model}")


class Parameters:

    def __init__(self, params, model="gauss"):

        self.params = params
        self.model = model
        self.allow_models = ["gauss", "dog", "css", "norm", "abc", "abd"]

        if isinstance(self.params, str):
            self.params = np.load(self.params)

    def to_df(self):

        if isinstance(self.params, pd.DataFrame):
            return self.params

        if not isinstance(self.params, np.ndarray):
            raise ValueError(f"Input must be np.ndarray, not '{type(self.params)}'")

        if self.params.ndim == 1:
            self.params = self.params[np.newaxis, :]

        # see: https://github.com/VU-Cog-Sci/prfpy_tools/blob/master/utils/postproc_utils.py#L377
        if self.model in self.allow_models:
            if self.model == "gauss":
                params_dict = {
                    "x": self.params[:, 0],
                    "y": self.params[:, 1],
                    "prf_size": self.params[:, 2],
                    "prf_ampl": self.params[:, 3],
                    "bold_bsl": self.params[:, 4],
                    "r2": self.params[:, -1],
                    "ecc": np.sqrt(self.params[:, 0] ** 2 + self.params[:, 1] ** 2),
                    "polar": np.angle(self.params[:, 0] + self.params[:, 1] * 1j),
                }

                if self.params.shape[-1] > 6:
                    params_dict["hrf_deriv"] = self.params[:, -3]
                    params_dict["hrf_disp"] = self.params[:, -2]

            elif self.model in ["norm", "abc", "abd"]:

                params_dict = {
                    "x": self.params[:, 0],
                    "y": self.params[:, 1],
                    "prf_size": self.params[:, 2],
                    "prf_ampl": self.params[:, 3],
                    "bold_bsl": self.params[:, 4],
                    "surr_ampl": self.params[:, 5],
                    "surr_size": self.params[:, 6],
                    "neur_bsl": self.params[:, 7],
                    "surr_bsl": self.params[:, 8],
                    "A": self.params[:, 3],
                    "B": self.params[:, 7],  # /params[:,3],
                    "C": self.params[:, 5],
                    "D": self.params[:, 8],
                    "ratio (B/D)": self.params[:, 7] / self.params[:, 8],
                    "r2": self.params[:, -1],
                    "size ratio": self.params[:, 6] / self.params[:, 2],
                    "suppression index": (self.params[:, 5] * self.params[:, 6] ** 2)
                    / (self.params[:, 3] * self.params[:, 2] ** 2),
                    "ecc": np.sqrt(self.params[:, 0] ** 2 + self.params[:, 1] ** 2),
                    "polar": np.angle(self.params[:, 0] + self.params[:, 1] * 1j),
                }

                if self.params.shape[-1] > 10:
                    params_dict["hrf_deriv"] = self.params[:, -3]
                    params_dict["hrf_dsip"] = self.params[:, -2]

            elif self.model == "dog":
                params_dict = {
                    "x": self.params[:, 0],
                    "y": self.params[:, 1],
                    "prf_size": self.params[:, 2],
                    "prf_ampl": self.params[:, 3],
                    "bold_bsl": self.params[:, 4],
                    "surr_ampl": self.params[:, 5],
                    "surr_size": self.params[:, 6],
                    "r2": self.params[:, -1],
                    "size ratio": self.params[:, 6] / self.params[:, 2],
                    "suppression index": (self.params[:, 5] * self.params[:, 6] ** 2)
                    / (self.params[:, 3] * self.params[:, 2] ** 2),
                    "ecc": np.sqrt(self.params[:, 0] ** 2 + self.params[:, 1] ** 2),
                    "polar": np.angle(self.params[:, 0] + self.params[:, 1] * 1j),
                }

                if self.params.shape[-1] > 8:
                    params_dict["hrf_deriv"] = self.params[:, -3]
                    params_dict["hrf_dsip"] = self.params[:, -2]

            elif self.model == "css":
                params_dict = {
                    "x": self.params[:, 0],
                    "y": self.params[:, 1],
                    "prf_size": self.params[:, 2],
                    "prf_ampl": self.params[:, 3],
                    "bold_bsl": self.params[:, 4],
                    "css_exp": self.params[:, 5],
                    "r2": self.params[:, -1],
                    "ecc": np.sqrt(self.params[:, 0] ** 2 + self.params[:, 1] ** 2),
                    "polar": np.angle(self.params[:, 0] + self.params[:, 1] * 1j),
                }

                if self.params.shape[-1] > 7:
                    params_dict["hrf_deriv"] = self.params[:, -3]
                    params_dict["hrf_dsip"] = self.params[:, -2]

        else:
            raise ValueError(
                f"Model must be one of {self.allow_models}. Not '{self.model}'"
            )

        return pd.DataFrame(params_dict)

    def to_array(self):

        if not isinstance(self.params, pd.DataFrame):
            raise ValueError(f"Input must be pd.DataFrame, not '{type(self.params)}'")

        if self.model == "gauss":
            item_list = [
                "x",
                "y",
                "prf_size",
                "prf_ampl",
                "bold_bsl",
                "hrf_deriv",
                "hrf_disp",
                "r2",
            ]
        elif self.model in ["norm", "abc", "abd"]:
            item_list = [
                "x",
                "y",
                "prf_size",
                "prf_ampl",
                "bold_bsl",
                "surr_ampl",
                "surr_size",
                "neur_bsl",
                "surr_bsl",
                "hrf_deriv",
                "hrf_disp",
                "r2",
            ]
        elif self.model == "dog":
            item_list = [
                "x",
                "y",
                "prf_size",
                "prf_ampl",
                "bold_bsl",
                "surr_ampl",
                "surr_size",
                "hrf_deriv",
                "hrf_disp",
                "r2",
            ]
        elif self.model == "css":
            item_list = [
                "x",
                "y",
                "prf_size",
                "prf_ampl",
                "bold_bsl",
                "css_exp",
                "hrf_deriv",
                "hrf_disp",
                "r2",
            ]
        else:
            raise ValueError(
                f"Model must be one of 'gauss','norm','dog','css'; not '{self.model}'"
            )

        self.parr = []

        # parallel counter in case HRF-parameters are not present
        ct = 0
        for ii in item_list:
            if ii in list(self.params.columns):
                pars = self.params[ii].values
                if not np.isnan(pars).all():
                    self.parr.append(pars[..., np.newaxis])
                    ct += 1

        return np.concatenate(self.parr, axis=1)


def fwhmax_fwatmin(model, params, normalize_RFs=False, return_profiles=False):
    model = model.lower()
    x = np.linspace(-50, 50, 1000).astype("float32")

    prf = params[..., 3] * np.exp(-0.5 * x[..., np.newaxis] ** 2 / params[..., 2] ** 2)
    vol_prf = 2 * np.pi * params[..., 2] ** 2

    if "dog" in model or "norm" in model:
        srf = params[..., 5] * np.exp(
            -0.5 * x[..., np.newaxis] ** 2 / params[..., 6] ** 2
        )
        vol_srf = 2 * np.pi * params[..., 6] ** 2

    if normalize_RFs == True:

        if model == "gauss":
            profile = prf / vol_prf
        elif model == "css":
            # amplitude is outside exponent in CSS
            profile = (prf / vol_prf) ** params[..., 5] * params[..., 3] ** (
                1 - params[..., 5]
            )
        elif model == "dog":
            profile = prf / vol_prf - srf / vol_srf
        elif "norm" in model:
            profile = (prf / vol_prf + params[..., 7]) / (
                srf / vol_srf + params[..., 8]
            ) - params[..., 7] / params[..., 8]
    else:
        if model == "gauss":
            profile = prf
        elif model == "css":
            # amplitude is outside exponent in CSS
            profile = prf ** params[..., 5] * params[..., 3] ** (1 - params[..., 5])
        elif model == "dog":
            profile = prf - srf
        elif "norm" in model:
            profile = (prf + params[..., 7]) / (srf + params[..., 8]) - params[
                ..., 7
            ] / params[..., 8]

    half_max = np.max(profile, axis=0) / 2
    fwhmax = np.abs(2 * x[np.argmin(np.abs(profile - half_max), axis=0)])

    if "dog" in model or "norm" in model:

        min_profile = np.min(profile, axis=0)
        fwatmin = np.abs(2 * x[np.argmin(np.abs(profile - min_profile), axis=0)])

        result = fwhmax, fwatmin
    else:
        result = fwhmax

    if return_profiles:
        return result, profile.T
    else:
        return result

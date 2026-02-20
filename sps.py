import numpy as np
import logging
import os
from astropy.io import fits
from scipy import interpolate

try:
    from prospect.sources import FastStepBasis, CSPSpecBasis
except ImportError:
    FastStepBasis, CSPSpecBasis = None, None

logger = logging.getLogger(__name__)


class ProspectorSPSBuilder:
    """
    Manages the initialization of the SPS (Stellar Population Synthesis) object.
    Applies instrumental smoothing (LSF) by reading data directly from the
    configured file, forcing the use of the MILES library.
    """

    def __init__(self, config, data_handler, model):
        self.config = config
        self.data_handler = data_handler
        self.model = model

    def build_sps(self):
        """Builds and returns the configured SPS object."""
        if FastStepBasis is None:
            logger.error("Missing prospect.models.sedmodel libraries.")
            raise ImportError("Prospector is not installed correctly.")

        zcont = getattr(self.config, "z_continuous", 1)

        if "agebins" in self.model.model_params:
            sps = FastStepBasis(zcontinuous=zcont)
            logger.info("Built SPS: FastStepBasis (Non-parametric SFH).")

        elif "tau" in self.model.model_params:
            sps = CSPSpecBasis(zcontinuous=zcont)
            logger.info("Built SPS: CSPSpecBasis (Parametric SFH).")

        else:
            logger.error("Prospector model lacks both 'agebins' and 'tau'.")
            raise ValueError("Model parameters must include either 'agebins' or 'tau'.")

        if self.config.add_sigmav and ("agebins" in self.model.model_params):
            if self.config.dispersion_file is not None:
                self._apply_smoothing(sps)
            else:
                logger.info(
                    "Smoothing (add_sigmav) requested, but dispersion_file is None. Skipping LSF."
                )

        return sps

    def _read_jwst_dispersion(self, path):
        """Reads the JWST FITS file and returns an interpolation function."""
        with fits.open(path) as hdul:
            nirspec_wave = hdul[1].data["WAVELENGTH"]  # Usually in microns
            nirspec_R = hdul[1].data["R"]

            func_nirspec = interpolate.interp1d(
                nirspec_wave,
                2.998e5 / (nirspec_R * 2.355),
                bounds_error=False,
                fill_value="extrapolate",
            )
            return func_nirspec
    
    def _read_dispersion(self, path):
        
        return

    def _apply_smoothing(self, sps):
        """Reads the dispersion file and injects the LSF into the SPS object."""
        disp_file = self.config.dispersion_file

        if not os.path.exists(disp_file):
            logger.error(f"Dispersion file does not exist: {disp_file}")
            raise FileNotFoundError(f"Dispersion file not found: {disp_file}")

        logger.debug(f"Reading LSF dispersion file: {disp_file}")
        try:
            wave_obs = self.data_handler.spectroscopy["wavelength"]
            interp_func = self._read_jwst_dispersion(disp_file)

            # Divide by 10000 to convert Angstroms to Microns
            # to match JWST FITS wavelength units.
            sigma_v = interp_func(wave_obs / 10000.0)

        except Exception as e:
            logger.error(f"Error during LSF creation or evaluation: {e}")
            raise ValueError(
                f"Cannot apply dispersion file {disp_file}. Details: {e}"
            ) from e

        logger.debug("Computing LSF forcing the MILES library.")
        wave_lsf, delta_v = self._get_lsf(wave_obs, sigma_v)

        sps.ssp.params["smooth_lsf"] = True
        sps.ssp.set_lsf(wave_lsf, delta_v)

        logger.info(
            f"Instrumental smoothing (LSF) successfully applied from {os.path.basename(disp_file)}."
        )

    def _get_lsf(self, wave_obs, sigma_v):
        """
        Computes the quadrature difference between instrumental dispersion and the MILES library.
        Returns rest-frame wavelength and delta_v in km/s.
        """
        zred = self.config.redshift if self.config.redshift is not None else 0.0
        lightspeed = 2.998e5  # km/s

        # Filter out zero or negative instrumental dispersion values
        valid_dispersion = sigma_v > 0
        wave_obs_clean = wave_obs[valid_dispersion]
        sigma_v_clean = sigma_v[valid_dispersion]

        wave_rest = wave_obs_clean / (1.0 + zred)

        # Get MILES theoretical library resolution (rest-frame)
        miles_fwhm_aa = 2.54
        # formula: c * FWHM / (2.355 * lambda)
        sigma_v_lib = lightspeed * miles_fwhm_aa / (2.355 * wave_rest)

        # MILES validity domain
        valid_rest_wave = (wave_rest > 3525.0) & (wave_rest < 7500.0)

        # Clip at 0 prevents sqrt of negative numbers if library resolution is worse than instrument
        dsv = np.sqrt(np.clip(sigma_v_clean**2 - sigma_v_lib**2, 0, np.inf))

        return wave_rest[valid_rest_wave], dsv[valid_rest_wave]

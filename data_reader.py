import h5py
import os
import logging
from typing import Optional, Dict, Any
import numpy as np

try:
    from sedpy import observate
except ImportError:
    observate = None

logger = logging.getLogger(__name__)


class GalaxyDataManager:
    def __init__(self, config):
        self.config = config

        if not self.config.file:
            raise ValueError("Error: 'file' is not defined in the configuration.")

        self.filepath = self.config.file
        self.version = getattr(self.config, "version", None) or "V1"

        self.photometry: Optional[Dict[str, Any]] = None
        self.spectroscopy: Optional[Dict[str, Any]] = None
        self.metadata: Optional[Dict[str, Any]] = None

    def load_data(self):
        """Load data and start validation."""
        if not os.path.exists(self.filepath):
            logger.error(f"File not found: {self.filepath}")
            raise FileNotFoundError(f"Data file does not exist: {self.filepath}")

        logger.info(f"Opening HDF5 file: {self.filepath}")

        with h5py.File(self.filepath, "r") as h5_file:
            if self.version not in h5_file:
                logger.error(f"Version '{self.version}' missing in the file.")
                raise KeyError(
                    f"Version '{self.version}' not found in {self.filepath}."
                )

            logger.debug(f"Accessing version group: {self.version}")
            version_group = h5_file[self.version]

            # --- Photometry ---
            if self.config.use_photometry:
                if "Photometry" in version_group:
                    logger.info("Extracting photometric data...")
                    self.photometry = self._extract_to_memory(
                        version_group["Photometry"]
                    )
                    self._validate_photometry()
                    self._build_photometric_filters()
                else:
                    logger.warning(
                        "Photometry requested by config, but not found in the HDF5 file."
                    )

            # --- Spectroscopy ---
            if self.config.use_spectroscopy:
                if "Spectroscopy" in version_group:
                    logger.info("Extracting spectroscopic data...")
                    self.spectroscopy = self._extract_to_memory(
                        version_group["Spectroscopy"]
                    )
                    self._validate_spectroscopy()
                else:
                    logger.warning(
                        "Spectroscopy requested by config, but not found in the HDF5 file."
                    )

            # --- Metadata ---
            if "Metadata" in version_group:
                logger.info("Extracting metadata...")
                self.metadata = self._extract_to_memory(version_group["Metadata"])
                logger.debug(f"Metadata found: {list(self.metadata.keys())}")

                self._update_config_from_metadata()
            else:
                logger.debug("No 'Metadata' group found in the file.")

        logger.info("Data loading and validation completed successfully.")

    def _extract_to_memory(self, h5_object) -> Dict[str, Any]:
        """Recursively convert HDF5 datasets into a dictionary of numpy arrays in RAM."""
        data_dict = {}
        if isinstance(h5_object, h5py.Group):
            for key, item in h5_object.items():
                if isinstance(item, h5py.Dataset):
                    data_dict[key] = item[()]
                else:
                    data_dict[key] = self._extract_to_memory(item)
        elif isinstance(h5_object, h5py.Dataset):
            data_dict["data"] = h5_object[()]
        return data_dict

    def _validate_photometry(self):
        """Check photometry consistency and apply optional filters."""
        if not self.photometry:
            return

        logger.debug("Starting photometry validation and mask generation...")
        req_keys = ["flux", "flux_err", "filters"]
        for k in req_keys:
            if k not in self.photometry:
                logger.error(f"Corrupted photometry: missing key '{k}'.")
                raise ValueError(f"Corrupted photometric data: missing key '{k}'.")

        flux = self.photometry["flux"]
        flux_err = self.photometry["flux_err"]
        n_elements = len(flux)

        if self.config.use_mask and "mask" in self.photometry:
            logger.debug("Using photometric mask from the HDF5 file.")
            mask = np.array(self.photometry["mask"], dtype=bool)
            if len(mask) != n_elements:
                raise ValueError(
                    f"Photometric mask length ({len(mask)}) differs from data length ({n_elements})."
                )
        else:
            logger.debug("Creating default photometric mask.")
            mask = np.ones(n_elements, dtype=bool)

        if self.config.filter_photo:
            logger.debug("Applying NaN and Inf filters to photometry.")
            mask &= np.isfinite(flux)
            mask &= np.isfinite(flux_err)
            mask &= flux_err > 0

        self.photometry["mask"] = mask
        logger.debug(
            f"Photometric mask created: {mask.sum()}/{n_elements} valid pixels."
        )

    def _validate_spectroscopy(self):
        """Check spectroscopy consistency and apply optional filters."""
        if not self.spectroscopy:
            return

        logger.debug("Starting spectroscopy validation and mask generation...")
        req_keys = ["wavelength", "flux", "flux_err"]
        for k in req_keys:
            if k not in self.spectroscopy:
                logger.error(f"Corrupted spectroscopy: missing key '{k}'.")
                raise ValueError(f"Corrupted spectroscopic data: missing key '{k}'.")

        wave = self.spectroscopy["wavelength"]
        flux = self.spectroscopy["flux"]
        flux_err = self.spectroscopy["flux_err"]
        n_elements = len(flux)

        if self.config.use_mask and "mask" in self.spectroscopy:
            logger.debug("Using spectroscopic mask from the HDF5 file.")
            mask = np.array(self.spectroscopy["mask"], dtype=bool)
            if len(mask) != n_elements:
                raise ValueError(
                    f"Spectroscopic mask length ({len(mask)}) differs from data length ({n_elements})."
                )
        else:
            logger.debug("Creating default spectroscopic mask.")
            mask = np.ones(n_elements, dtype=bool)

        if self.config.filter_spec:
            logger.debug("Applying NaN and Inf filters to spectroscopy.")
            mask &= np.isfinite(flux)
            mask &= np.isfinite(flux_err)
            mask &= flux_err > 0
            mask &= np.isfinite(wave)
            mask &= wave > 0

        self.spectroscopy["mask"] = mask
        logger.debug(
            f"Spectroscopic mask created: {mask.sum()}/{n_elements} valid pixels."
        )

    def _build_photometric_filters(self):
        """Convert filter names from the H5 file into sedpy.observate.Filter objects."""
        if not self.photometry:
            return

        if observate is None:
            logger.error("The 'sedpy' library is not installed but is required.")
            raise ImportError(
                "The 'sedpy' library is not installed, but is required for photometry."
            )

        raw_filters = self.photometry["filters"]
        logger.debug(f"Building sedpy objects for {len(raw_filters)} filters...")

        filter_names = []
        for f in raw_filters:
            if isinstance(f, bytes):
                filter_names.append(f.decode("utf-8"))
            else:
                filter_names.append(str(f))

        try:
            filters_list = [observate.Filter(f) for f in filter_names]
        except Exception as e:
            logger.error(f"sedpy error while loading filters: {e}")
            raise ValueError(
                f"Error in sedpy: unable to find one or more filters. Details: {e}"
            )

        filters_wave = np.array([flt.wave_effective for flt in filters_list])

        self.photometry["sedpy_filters"] = filters_list
        self.photometry["wave_effective"] = filters_wave
        logger.debug("sedpy filters built successfully.")

    def _update_config_from_metadata(self):
        """
        Update the FitConfig object using metadata read from the HDF5 file,
        respecting CLI priorities.
        """
        if not self.metadata:
            return

        if "redshift" in self.metadata:
            z_file = self.metadata["redshift"]
            if isinstance(z_file, dict) and "data" in z_file:
                z_file = z_file["data"]

            try:
                z_file = float(z_file)
            except (ValueError, TypeError):
                logger.warning(f"Invalid redshift value in file: {z_file}")
                return

            if self.config.redshift is None:
                self.config.redshift = z_file
                logger.info(
                    f"Redshift automatically updated from metadata: z = {self.config.redshift:.4f}"
                )
            else:
                logger.info(
                    f"CLI redshift (z={self.config.redshift}) retained. Ignoring file z={z_file:.4f}."
                )

    def to_dict(self):
        observation = {
            "wavelength": self.spectroscopy["wavelength"],
            "spectrum": self.spectroscopy["flux"],
            "unc": self.spectroscopy["flux_err"],
            "mask": self.spectroscopy["mask"],
            "filters": self.photometry["sedpy_filters"],
            "maggies": self.photometry["flux"],
            "maggies_unc": self.photometry["flux_err"],
            "phot_mask": self.photometry["mask"],
            "phot_wave": self.photometry["wave_effective"],
        }
        return observation

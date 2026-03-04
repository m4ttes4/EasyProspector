import h5py
import os
import logging
from typing import Optional, Dict, Any
import numpy as np
import rich.box
from rich.table import Table
from rich.console import Console


try:
    from sedpy import observate
except ImportError:
    observate = None

logger = logging.getLogger(__name__)


def show_data_summary(obs: dict):
    """
    Stampa una tabella riassuntiva dei dati osservativi (Fotometria e Spettroscopia)
    che verranno passati a Prospector.
    """
    table = Table(
        title="Observation Data Summary",
        header_style="bold magenta",
        box=rich.box.ROUNDED,
        show_lines=True,
    )

    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Total\nPoints", justify="right", style="white")
    table.add_column("Valid", justify="right", style="green")
    table.add_column("Masked", justify="right", style="red")
    table.add_column("Wave Range (Å)", justify="center", style="yellow")
    table.add_column("Median Flux\n(Maggies)", justify="right", style="white")
    table.add_column("Median SNR", justify="right", style="blue")

    # --- FOTOMETRIA ---
    if obs.get("maggies") is not None:
        phot_flux = obs["maggies"]
        phot_unc = obs.get("maggies_unc", np.zeros_like(phot_flux))
        phot_mask = obs.get("phot_mask", np.ones_like(phot_flux, dtype=bool))
        phot_wave = obs.get("phot_wave", [])
        
        tot_phot = len(phot_flux)
        valid_phot = np.sum(phot_mask)
        masked_phot = tot_phot - valid_phot
        
        if valid_phot > 0:
            valid_flux = phot_flux[phot_mask]
            valid_unc = phot_unc[phot_mask]
            
            med_flux_phot = f"{np.nanmedian(valid_flux):.2e}"
            # Evitiamo divisioni per zero
            snr_phot = np.divide(valid_flux, valid_unc, out=np.zeros_like(valid_flux), where=valid_unc!=0)
            med_snr_phot = f"{np.nanmedian(snr_phot):.1f}"
            wave_range_phot = f"{np.nanmin(phot_wave):.0f} - {np.nanmax(phot_wave):.0f}"
        else:
            med_flux_phot = "N/A"
            med_snr_phot = "N/A"
            wave_range_phot = "N/A"
            
        table.add_row(
            "Photometry",
            str(tot_phot),
            str(valid_phot),
            str(masked_phot),
            wave_range_phot,
            med_flux_phot,
            med_snr_phot
        )

    # --- SPETTROSCOPIA ---
    if obs.get("spectrum") is not None:
        spec_flux = obs["spectrum"]
        spec_unc = obs.get("unc", np.zeros_like(spec_flux))
        spec_mask = obs.get("mask", np.ones_like(spec_flux, dtype=bool))
        spec_wave = obs.get("wavelength", [])
        
        tot_spec = len(spec_flux)
        valid_spec = np.sum(spec_mask)
        masked_spec = tot_spec - valid_spec
        
        if valid_spec > 0:
            valid_flux_spec = spec_flux[spec_mask]
            valid_unc_spec = spec_unc[spec_mask]
            
            med_flux_spec = f"{np.nanmedian(valid_flux_spec):.2e}"
            # Evitiamo divisioni per zero
            snr_spec = np.divide(valid_flux_spec, valid_unc_spec, out=np.zeros_like(valid_flux_spec), where=valid_unc_spec!=0)
            med_snr_spec = f"{np.nanmedian(snr_spec):.1f}"
            wave_range_spec = f"{np.nanmin(spec_wave):.0f} - {np.nanmax(spec_wave):.0f}"
        else:
            med_flux_spec = "N/A"
            med_snr_spec = "N/A"
            wave_range_spec = "N/A"

        table.add_row(
            "Spectroscopy",
            str(tot_spec),
            str(valid_spec),
            str(masked_spec),
            wave_range_spec,
            med_flux_spec,
            med_snr_spec
        )

    # Rendering sicuro per i log (senza codici ANSI su file)
    console = Console(width=100, color_system=None)
    with console.capture() as capture:
        console.print(table)
        
    logger.info("\n" + capture.get())


def show_photometry_details(obs: dict):
    """
    Stampa una tabella dettagliata per ogni filtro fotometrico fornito a Prospector.
    """
    if obs.get("maggies") is None or obs.get("filters") is None:
        return

    table = Table(
        title="Photometric Filters Details",
        header_style="bold cyan",
        box=rich.box.ROUNDED,
        show_lines=True,
    )

    table.add_column("Filter Name", style="magenta")
    table.add_column("Eff. Wave (Å)", justify="right", style="yellow")
    table.add_column("Flux (Maggies)", justify="right", style="white")
    table.add_column("Error", justify="right", style="white")
    table.add_column("SNR", justify="right", style="blue")
    table.add_column("Status", justify="center")

    filters = obs["filters"]
    fluxes = obs["maggies"]
    uncs = obs.get("maggies_unc", np.zeros_like(fluxes))
    mask = obs.get("phot_mask", np.ones_like(fluxes, dtype=bool))
    waves = obs.get("phot_wave", [getattr(f, 'wave_effective', 0.0) for f in filters])

    for i, f in enumerate(filters):
        name = getattr(f, "name", f"Filter_{i}")
        wave = waves[i]
        flux = fluxes[i]
        unc = uncs[i]
        is_valid = mask[i]
        
        # Formattazione
        flux_str = f"{flux:.3e}" if np.isfinite(flux) else "NaN"
        unc_str = f"{unc:.3e}" if np.isfinite(unc) else "NaN"
        
        if unc > 0 and np.isfinite(flux) and np.isfinite(unc):
            snr = flux / unc
            snr_str = f"{snr:.1f}"
        else:
            snr_str = "N/A"
            
        status_str = "[bold green]Valid[/]" if is_valid else "[bold red]Masked[/]"

        table.add_row(
            name,
            f"{wave:.1f}",
            flux_str,
            unc_str,
            snr_str,
            status_str
        )

    console = Console(width=100, color_system=None)
    with console.capture() as capture:
        console.print(table)
        
    logger.info("\n" + capture.get())


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
            mask &= flux > 0
            # phot_mask = [
            #     True
            #     if flux[i] > 0 or ~np.isfinite(flux[i])
            #     else False
            #     for i in range(len(flux))
            # ]

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

        # self.photometry["sedpy_filters"] = filters_list
        # self.photometry["wave_effective"] = filters_wave
        # --- LOGICA DI SORTING E LOGGING ---
        sort_idx = np.argsort(filters_wave)
        original_idx = np.arange(len(filters_wave))

        # Controlla se l'array ordinato è diverso da quello originale
        if not np.array_equal(sort_idx, original_idx):
            old_order = [filter_names[i] for i in original_idx]
            new_order = [filter_names[i] for i in sort_idx]
            logger.info("Photometric filters were not sorted by wavelength. Reordering them now.")
            logger.debug(f"Old order: {old_order}")
            logger.debug(f"New order: {new_order}")

        # Applica il sorting agli oggetti sedpy e alle lunghezze d'onda
        self.photometry["sedpy_filters"] = [filters_list[i] for i in sort_idx]
        self.photometry["wave_effective"] = filters_wave[sort_idx]

        # --- CONVERSIONE IN MAGGIES E SORTING DEI DATI ---
        raw_flux = self.photometry["flux"]
        raw_err = self.photometry["flux_err"]
        
        # Converte da uJy a Maggies
        flux_mag, err_mag = raw_flux, raw_err#self._uJy_to_maggies(raw_flux, raw_err)
        
        # Riapplica il sorting ai flussi
        self.photometry["flux"] = flux_mag[sort_idx]
        self.photometry["flux_err"] = err_mag[sort_idx]
        
        # Applica il sorting anche alla maschera (se esiste)
        if "mask" in self.photometry:
            self.photometry["mask"] = self.photometry["mask"][sort_idx]

        logger.debug("sedpy filters built and data sorted successfully.")

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


    def show(self):
        tmp = self.to_dict()
        show_data_summary(tmp)
        # show_photometry_details(tmp)

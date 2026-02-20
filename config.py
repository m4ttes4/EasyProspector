import sys
import os
import argparse
from typing import Dict, Optional, Any
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class FitConfig:
    # --- 1. Identifiers and Paths (I/O) ---

    file: Optional[str] = None  # Full path to the input .h5 file
    out: Optional[str] = None  # Output name
    out_folder: Optional[str] = "results/out"  # Folder where results will be saved
    file_list: Optional[str] = None

    name: Optional[str] = "result"
    logging_to_file: bool = False  # Default: logging to terminal/stdout
    log_folder: str = "results/log"

    version: Optional[str] = "V1"  # E.g.: "F160W_selected"
    use_mask: bool = True  # Use the mask inside the .h5 file
    dispersion_file: Optional[str] = None
    param_file: str = field(default_factory=lambda: sys.argv[0])

    # --- 2. Data Selection ---
    use_photometry: bool = True
    use_spectroscopy: bool = True

    filter_photo: bool = True
    filter_spec: bool = True
    fit_outliers_photo: bool = False
    fit_outliers_spec: bool = False

    # --- 3. Physical Parameters and Model ---
    model_type: str = "ContinuitySFH"
    redshift: Optional[float] = None
    fixed_z: bool = False

    nbins: int = 8
    z_continuous: int = 1

    add_nebular: bool = True
    add_duste: bool = True
    add_dust1: bool = True
    add_agn: bool = False
    add_sigmav: bool = True

    # --- 4. Fitting Configuration (Engine) ---
    optimize: bool = False
    emcee: bool = False
    dynesty: bool = True

    dynesty_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "nested_nlive_init": 300,
            "nested_sample": "rwalk",
            "nested_target_n_effective": 300,
            "nested_dlogz_init": 0.01,
        }
    )

    # --- 5. Debug, Plotting, and Interactivity ---
    verbose: bool = True
    interactive: bool = False
    lines: Dict[str, tuple] = field(default_factory=dict)

    def __post_init__(self):
        """Validations and automatic setup after initialization."""
        self.targets = []

        # Check that the value exists before expanding the tilde (~)
        if self.file:
            self.file = os.path.expanduser(self.file)

        if self.out_folder:
            self.out_folder = os.path.expanduser(self.out_folder)

        if self.dispersion_file:
            self.dispersion_file = os.path.expanduser(self.dispersion_file)

        if self.file_list:
            self.file_list = os.path.expanduser(self.file_list)

        if self.file_list and os.path.exists(self.file_list):
            # If a list is provided, read the file and ignore empty lines
            with open(self.file_list, "r") as f:
                self.targets = [line.strip() for line in f if line.strip()]

        elif self.file:
            self.targets = [self.file]

    def to_dict(self) -> Dict[str, Any]:
        """Converts the configuration into a dictionary (e.g., for Prospector)."""
        data = asdict(self)
        data.update(self.dynesty_kwargs)
        return data

    def update_from_cli(self):
        """
        Updates values by reading from the command line.
        Uses argparse.SUPPRESS to avoid overwriting defaults with 'None'.
        """
        parser = argparse.ArgumentParser(description="Prospector Run Configuration")

        # Helper function to avoid writing default=argparse.SUPPRESS everywhere
        def add_arg(*args, **kwargs):
            kwargs["default"] = argparse.SUPPRESS
            parser.add_argument(*args, **kwargs)

        # Helper function to handle booleans (explicit True/False)
        def add_bool(name, dest):
            # E.g.: --interactive sets the flag to True, --no-interactive sets it to False
            parser.add_argument(
                f"--{name}", dest=dest, action="store_true", default=argparse.SUPPRESS
            )
            parser.add_argument(
                f"--no-{name}",
                dest=dest,
                action="store_false",
                default=argparse.SUPPRESS,
            )

        # 1. Strings and Numbers
        add_arg("--galaxy_name", type=str)
        add_arg("--file", type=str)
        add_arg("--file_list", type=str)
        add_arg("--out", type=str)
        add_arg("--out_folder", type=str)
        add_arg("--version", type=str)
        add_arg("--dispersion_file", type=str)
        add_arg("--redshift", type=float)
        add_arg("--model_type", type=str)
        add_arg("--nbins", type=int)
        add_arg("--log_folder", type=str)
        add_arg("--z_continuous", type=int)

        # New Boolean arguments
        add_bool("use_mask", "use_mask")
        add_bool("filter_photo", "filter_photo")
        add_bool("filter_spec", "filter_spec")
        add_bool("fit_outliers_photo", "fit_outliers_photo")
        add_bool("fit_outliers_spec", "fit_outliers_spec")

        # 2. Booleans (Tristate Management)
        add_bool("interactive", "interactive")
        add_bool("verbose", "verbose")
        add_bool("photometry", "use_photometry")
        add_bool("spectroscopy", "use_spectroscopy")
        add_bool("sigmav", "add_sigmav")
        add_bool("nebular", "add_nebular")
        add_bool("duste", "add_duste")
        add_bool("dust1", "add_dust1")
        add_bool("agn", "add_agn")
        add_bool("fixed_z", "fixed_z")
        add_bool("optimize", "optimize")
        add_bool("dynesty", "dynesty")
        add_bool("emcee", "emcee")
        add_bool("logging_file", "logging_to_file")


        args, unknown = parser.parse_known_args()

        # Actual instance update
        if unknown:
            logger.warning(f"Ignored or unknown CLI arguments: {unknown}")

        updated_keys = []
        for key, value in vars(args).items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                updated_keys.append(f"{key}: {old_value} -> {value}")
            else:
                # Warnings can be printed immediately or added to an error list
                print(f"Warning: CLI argument '{key}' does not exist.")

        self.__post_init__()

        return updated_keys

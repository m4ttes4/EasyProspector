from typing import Dict, Any
import numpy as np
from prospect.models import priors, transforms
from prospect.models.templates import TemplateLibrary, adjust_continuity_agebins

import logging
import rich.box
from rich.table import Table
from rich.console import Console

logger = logging.getLogger(__name__)


def format_dict(d: Dict) -> str:
    """Helper to format dictionaries into a readable string."""
    lines = []
    for key, value in d.items():
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        lines.append(f"{key} --> {formatted_value}")
    return "\n".join(lines)


def format_long_list(value: Any, max_items_per_line: int = 3) -> str:
    """Formats long lists or arrays across multiple lines."""
    if isinstance(value, (list, np.ndarray)) and len(value) > max_items_per_line:
        if isinstance(value, np.ndarray):
            val_list = value.tolist()
        else:
            val_list = value

        lines = [
            ", ".join(map(str, val_list[i : i + max_items_per_line]))
            for i in range(0, len(val_list), max_items_per_line)
        ]
        return "\n".join(lines)
    return str(value)


def show_model(model_params: Dict[str, Any]):
    """
    Tabulates Prospector model parameters using Rich.
    Prints the table cleanly and safely using the global logging module.
    """
    table = Table(
        title="Prospector Model Parameters",
        header_style="bold cyan",
        box=rich.box.ROUNDED,
        show_lines=True,
    )

    table.add_column("Parameter", style="yellow", no_wrap=True)
    table.add_column("Initial Value", style="white", width=45)
    table.add_column("Is Free?", justify="center", style="yellow", width=15)
    table.add_column("Prior", style="green", width=25)
    table.add_column("Depends on", style="yellow", width=25)
    table.add_column("Shape", justify="right", style="blue", width=15)

    for key, param_info in model_params.items():
        is_free = param_info.get("isfree", False)
        prior = param_info.get("prior", "N/A")
        init_value = param_info.get("init", "N/A")
        dep = param_info.get("depends_on", None)

        dep_str = dep.__name__ if dep else "N/A"

        # Rich parses these tags if the console has colors enabled,
        # otherwise it strips them automatically for file logging.
        is_free_display = "[bold green]True[/]" if is_free else "[bold red]False[/]"

        if not isinstance(prior, (str, type(None))):
            # Safe check to prevent crashes if the prior object lacks .params
            if hasattr(prior, "params"):
                param_str = format_dict(prior.params) + "\n"
            else:
                param_str = ""
            prior = prior.__class__.__name__ + "\n" + param_str

        shape = "N/A"
        if init_value is not None:
            calculated_shape = np.shape(init_value)
            shape = "scalar" if calculated_shape == () else repr(calculated_shape)

        init_value_display = format_long_list(init_value)

        table.add_row(
            key, init_value_display, is_free_display, str(prior), dep_str, shape
        )

    # Create a "virtual" console.
    # color_system=None is CRUCIAL when logging to a file to prevent ANSI escape codes.
    console = Console(width=120, color_system=None)

    with console.capture() as capture:
        console.print(table)

    table_str = capture.get()

    # Prepend a newline so the table starts cleanly below the log timestamp
    logger.info("\n" + table_str)


class ProspectorModelBuilder:
    """
    Base class to build Prospector model parameters.
    """

    def __init__(self):
        self.model_params = {}

    def get_params(self) -> Dict[str, Any]:
        return self.model_params


class BaseModel(ProspectorModelBuilder):
    """
    Builds a model with a non-parametric Star Formation History (Continuity SFH).
    Handles complex configurations like nebular emission, dust, smoothing, and outliers.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self._setup_sfh()
        self._setup_physical_params()
        self._setup_dust()

        if self.config.add_nebular:
            self._setup_nebular()

        if self.config.use_spectroscopy:
            self._setup_spectroscopy()

        self._setup_outliers()

        if getattr(self.config, "margin_elines", False):
            self._setup_line_marginalization()

    def _setup_sfh(self):
        """Sets up parameters for the Continuity SFH (mass and time bins)."""
        self.model_params.update(TemplateLibrary["continuity_sfh"])

        z = self.config.redshift if self.config.redshift is not None else 0.0

        # Adjust time bins based on the universe age at z
        # Note: tuniv=13.7 is a broad approximation
        self.model_params.update(
            adjust_continuity_agebins(
                self.model_params, tuniv=13.7, nbins=self.config.nbins
            )
        )

        # Total Mass
        self.model_params["logmass"] = {
            "N": 1,
            "isfree": True,
            "init": 10.5,
            "units": "Solar masses formed",
            "prior": priors.TopHat(mini=6.0, maxi=13.0),
        }

        # Bin Masses (Calculated from logmass + ratios)
        self.model_params["mass"] = {
            "N": self.config.nbins,
            "isfree": False,
            "init": 1e6,
            "units": "Solar masses formed",
            "depends_on": transforms.logsfr_ratios_to_masses,
        }

        # Redshift Fitting
        if not self.config.fixed_z:
            self.model_params["zred"] = {
                "N": 1,
                "isfree": True,
                "init": z,
                "units": "redshift",
                "prior": priors.ClippedNormal(
                    mean=z, sigma=0.05, mini=z - 0.5, maxi=z + 0.5
                ),
            }
        else:
            self.model_params["zred"] = {
                "N": 1,
                "isfree": False,
                "init": z,
                "units": "redshift",
            }

    def _setup_physical_params(self):
        """Stellar Metallicity and IMF."""
        self.model_params["logzsol"] = {
            "N": 1,
            "isfree": True,
            "init": -0.3,
            "units": r"$\log (Z/Z_\odot)$",
            "prior": priors.TopHat(mini=-2.0, maxi=0.5),
        }

        # IMF (1 = Chabrier)
        self.model_params["imf_type"] = {
            "N": 1,
            "isfree": False,
            "init": 1,
            "units": "FSPS index",
        }

    def _setup_dust(self):
        """Dust attenuation and emission configuration."""
        # Continuum Attenuation (Dust2 / Calzetti+)
        self.model_params["dust_type"] = {
            "N": 1,
            "isfree": False,
            "init": 4,
            "units": "FSPS index",
        }

        self.model_params["dust2"] = {
            "N": 1,
            "isfree": True,
            "init": 0.5,
            "units": "optical depth at 5500AA",
            "prior": priors.TopHat(mini=0.0, maxi=4.0),
        }

        self.model_params["dust_index"] = {
            "N": 1,
            "isfree": True,
            "init": 0.0,
            "prior": priors.ClippedNormal(mini=-1.5, maxi=0.4, mean=0.0, sigma=0.3),
        }

        # Birth Cloud Dust (Dust1)
        if self.config.add_dust1:
            self.model_params["dust1"] = {
                "N": 1,
                "isfree": False,
                "depends_on": transforms.dustratio_to_dust1,
                "init": 0.0,
            }
            self.model_params["dust1_fraction"] = {
                "N": 1,
                "isfree": True,
                "init": 1.0,
                "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3),
            }

        # Dust Emission (IR)
        if self.config.add_duste:
            self.model_params.update(TemplateLibrary["dust_emission"])
            self.model_params["duste_gamma"]["isfree"] = True
            self.model_params["duste_gamma"]["prior"] = priors.TopHat(
                mini=0.0, maxi=1.0
            )

            self.model_params["duste_qpah"]["isfree"] = True
            self.model_params["duste_qpah"]["prior"] = priors.TopHat(
                mini=0.5, maxi=10.0
            )

            self.model_params["duste_umin"]["isfree"] = True
            self.model_params["duste_umin"]["prior"] = priors.TopHat(
                mini=0.1, maxi=25.0
            )

    def _setup_nebular(self):
        """Nebular Emissions (HII regions)."""
        self.model_params.update(TemplateLibrary["nebular"])

        # Default: do not include lines directly in stellar spectrum
        self.model_params["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}

        # Gas physics
        self.model_params["gas_logz"] = {
            "N": 1,
            "isfree": True,
            "init": 0.0,
            "prior": priors.TopHat(mini=-2.0, maxi=0.5),
        }
        self.model_params["gas_logu"] = {
            "N": 1,
            "isfree": True,
            "init": -2.0,
            "prior": priors.TopHat(mini=-4.0, maxi=-1.0),
        }
        # Intrinsic line velocity dispersion
        self.model_params["eline_sigma"] = {
            "N": 1,
            "isfree": True,
            "init": 150.0,
            "units": "km/s",
            "prior": priors.TopHat(mini=50, maxi=500),
        }

    def _setup_spectroscopy(self):
        """Instrumental parameters: Smoothing and Calibration polynomials."""

        # 1. Spectral Smoothing (Sigma Smooth)
        self.model_params.update(TemplateLibrary["spectral_smoothing"])
        self.model_params["sigma_smooth"] = {
            "N": 1,
            "isfree": True,
            "init": 1000.0,
            "units": "km/s",
            "prior": priors.TopHat(mini=200.0, maxi=2000.0),
        }

        # 2. Continuum Optimization (Polynomials)
        self.model_params.update(TemplateLibrary["optimize_speccal"])

        # Spectrum Normalization (Scaling factor)
        self.model_params["spec_norm"] = {
            "N": 1,
            "isfree": True,
            "init": 1.0,
            "prior": priors.Normal(mean=1.0, sigma=0.2),
        }

        # Jitter (Noise floor underestimation)
        self.model_params["spec_jitter"] = {
            "N": 1,
            "isfree": True,
            "init": 1.0,
            "prior": priors.TopHat(mini=0.0, maxi=5.0),
        }

        # Polynomial Order
        self.model_params["polyorder"] = {
            "N": 1,
            "isfree": False,
            "init": 10,
        }

    def _setup_outliers(self):
        """Mixture models to handle 'bad' data (outliers)."""

        if self.config.fit_outliers_spec and self.config.use_spectroscopy:
            self.model_params["f_outlier_spec"] = {
                "N": 1,
                "isfree": True,
                "init": 0.01,
                "prior": priors.TopHat(mini=1e-5, maxi=0.2),
            }
            self.model_params["nsigma_outlier_spec"] = {
                "N": 1,
                "isfree": False,
                "init": 50.0,
            }

        if self.config.fit_outliers_photo and self.config.use_photometry:
            self.model_params["f_outlier_phot"] = {
                "N": 1,
                "isfree": True,
                "init": 0.00,
                "prior": priors.TopHat(mini=0, maxi=0.1),
            }
            self.model_params["nsigma_outlier_phot"] = {
                "N": 1,
                "isfree": False,
                "init": 50.0,
            }

    def _setup_line_marginalization(self):
        """
        Complex logic to marginalize emission lines.
        Requires 'lines' in config and valid observed data.
        """
        if not self.config.add_nebular:
            raise ValueError("Marginalization requires add_nebular=True")

        # Retrieve the lines to fit from config
        lines_dict = getattr(self.config, "lines", {})
        if not lines_dict:
            logger.warning("margin_elines=True but no lines found in config.")
            return

        to_fit_names = list(lines_dict.keys())
        valid_lines = to_fit_names

        if not valid_lines:
            return

        # Load marginalization template
        self.model_params.update(TemplateLibrary["nebular_marginalization"])

        self.model_params["elines_to_fit"] = {
            "N": 1,
            "isfree": False,
            "init": np.array(valid_lines),
        }

        # Amplitude prior (velocity sigma for these lines)
        self.model_params["eline_sigma"] = {
            "N": 1,
            "isfree": True,
            "init": 300.0,
            "prior": priors.TopHat(mini=50.0, maxi=1000.0),
        }

        # Specific Redshift fit for lines (Offset relative to stars)
        if getattr(self.config, "fit_eline_redshift", False):
            n_lines = len(valid_lines)
            self.model_params["fit_eline_redshift"] = {
                "N": 1,
                "isfree": False,
                "init": True,
            }

            self.model_params["eline_delta_zred"] = {
                "N": n_lines,
                "isfree": True,
                "init": np.zeros(n_lines),
                "prior": priors.TopHat(mini=-0.01, maxi=0.01),
            }

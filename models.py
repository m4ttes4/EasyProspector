from typing import Dict, Any
import numpy as np
from prospect.models import priors, transforms
from prospect.models.templates import TemplateLibrary, adjust_continuity_agebins
from astropy.cosmology import Planck18

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
        
        
        if self.config.add_agn:
            self._setup_agn()

        if self.config.add_nebular:
            self._setup_nebular()

        if self.config.use_spectroscopy:
            self._setup_spectroscopy()

        self._setup_outliers()

        if getattr(self.config, "margin_elines", False):
            self._setup_line_marginalization()
    
    def _setup_agn(self):
        self.model_params.update(TemplateLibrary["agn"])
        # Rendi i parametri dell'AGN liberi in modo che Dynesty li fitti
        self.model_params["fagn"]["isfree"] = True
        self.model_params["agn_tau"]["isfree"] = True

        # 3. (Opzionale) Aggiungi le righe di emissione dell'AGN
        self.model_params.update(TemplateLibrary["agn_eline"])

        # Rendi i parametri delle righe liberi
        self.model_params["agn_elum"]["isfree"] = True
        self.model_params["agn_eline_sigma"] = {
            "N": 1,
            "isfree": True,
            "init": 5000,
            "prior": priors.TopHat(mini=3000, maxi=9000),
        }
        # self.modelparams["agn_eline_sigma"]["init"] = 5000
        # self.model_params["agn_eline_sigma"]["isfree"] = True

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
            "prior": priors.TopHat(mini=8.0, maxi=13.0),
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
            self.model_params["duste_gamma"]["isfree"] = False
            self.model_params["duste_gamma"]["init"] = 0.01
            self.model_params["duste_gamma"]["prior"] = priors.TopHat(
                mini=0.0, maxi=1.0
            )

            self.model_params["duste_qpah"]["isfree"] = True
            self.model_params["duste_qpah"]["init"] = 3.5
            self.model_params["duste_qpah"]["prior"] = priors.TopHat(
                mini=0.5, maxi=10.0
            )

            self.model_params["duste_umin"]["isfree"] = False
            self.model_params["duste_umin"]["init"] = 1.0
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
            "prior": priors.TopHat(mini=-2.0, maxi=0.5),
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
            "prior": priors.TopHat(mini=500.0, maxi=5000.0),
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

        is_free_display = "[bold green]True[/]" if is_free else "[bold red]False[/]"

        if not isinstance(prior, (str, type(None))):
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

    console = Console(width=120, color_system=None)

    with console.capture() as capture:
        console.print(table)

    table_str = capture.get()
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

        if self.config.add_agn:
            self._setup_agn()

        if self.config.add_nebular:
            self._setup_nebular()

        if self.config.use_spectroscopy:
            self._setup_spectroscopy()

        self._setup_outliers()

        if getattr(self.config, "margin_elines", False):
            self._setup_line_marginalization()

    def _setup_agn(self):
        self.model_params.update(TemplateLibrary["agn"])
        self.model_params["fagn"]["isfree"] = True
        self.model_params["agn_tau"]["isfree"] = True

        self.model_params.update(TemplateLibrary["agn_eline"])

        self.model_params["agn_elum"]["isfree"] = True
        self.model_params["agn_eline_sigma"] = {
            "N": 1,
            "isfree": True,
            "init": 5000,
            "prior": priors.TopHat(mini=3000, maxi=9000),
        }

    def _setup_sfh(self):
        """Sets up parameters for the Continuity SFH (mass and time bins)."""
        self.model_params.update(TemplateLibrary["continuity_sfh"])

        z = self.config.redshift if self.config.redshift is not None else 0.0

        self.model_params.update(
            adjust_continuity_agebins(
                self.model_params, tuniv=13.7, nbins=self.config.nbins
            )
        )

        self.model_params["logmass"] = {
            "N": 1,
            "isfree": True,
            "init": 10.5,
            "units": "Solar masses formed",
            "prior": priors.TopHat(mini=8.0, maxi=13.0),
        }

        self.model_params["mass"] = {
            "N": self.config.nbins,
            "isfree": False,
            "init": 1e6,
            "units": "Solar masses formed",
            "depends_on": transforms.logsfr_ratios_to_masses,
        }

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

        self.model_params["imf_type"] = {
            "N": 1,
            "isfree": False,
            "init": 1,
            "units": "FSPS index",
        }

    def _setup_dust(self):
        """Dust attenuation and emission configuration."""
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

        if self.config.add_duste:
            self.model_params.update(TemplateLibrary["dust_emission"])
            self.model_params["duste_gamma"]["isfree"] = False
            self.model_params["duste_gamma"]["init"] = 0.01
            self.model_params["duste_gamma"]["prior"] = priors.TopHat(
                mini=0.0, maxi=1.0
            )

            self.model_params["duste_qpah"]["isfree"] = True
            self.model_params["duste_qpah"]["init"] = 3.5
            self.model_params["duste_qpah"]["prior"] = priors.TopHat(
                mini=0.5, maxi=10.0
            )

            self.model_params["duste_umin"]["isfree"] = False
            self.model_params["duste_umin"]["init"] = 1.0
            self.model_params["duste_umin"]["prior"] = priors.TopHat(
                mini=0.1, maxi=25.0
            )

    def _setup_nebular(self):
        """Nebular Emissions (HII regions)."""
        self.model_params.update(TemplateLibrary["nebular"])

        self.model_params["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}

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
            "prior": priors.TopHat(mini=-2.0, maxi=0.5),
        }
        self.model_params["eline_sigma"] = {
            "N": 1,
            "isfree": True,
            "init": 150.0,
            "units": "km/s",
            "prior": priors.TopHat(mini=50, maxi=500),
        }

    def _setup_spectroscopy(self):
        """Instrumental parameters: Smoothing and Calibration polynomials."""
        self.model_params.update(TemplateLibrary["spectral_smoothing"])
        self.model_params["sigma_smooth"] = {
            "N": 1,
            "isfree": True,
            "init": 1000.0,
            "units": "km/s",
            "prior": priors.TopHat(mini=500.0, maxi=5000.0),
        }

        self.model_params.update(TemplateLibrary["optimize_speccal"])

        self.model_params["spec_norm"] = {
            "N": 1,
            "isfree": True,
            "init": 1.0,
            "prior": priors.Normal(mean=1.0, sigma=0.2),
        }

        self.model_params["spec_jitter"] = {
            "N": 1,
            "isfree": True,
            "init": 1.0,
            "prior": priors.TopHat(mini=0.0, maxi=5.0),
        }

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

        lines_dict = getattr(self.config, "lines", {})
        if not lines_dict:
            logger.warning("margin_elines=True but no lines found in config.")
            return

        to_fit_names = list(lines_dict.keys())
        valid_lines = to_fit_names

        if not valid_lines:
            return

        self.model_params.update(TemplateLibrary["nebular_marginalization"])

        self.model_params["elines_to_fit"] = {
            "N": 1,
            "isfree": False,
            "init": np.array(valid_lines),
        }

        self.model_params["eline_sigma"] = {
            "N": 1,
            "isfree": True,
            "init": 300.0,
            "prior": priors.TopHat(mini=50.0, maxi=1000.0),
        }

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


class ContinuitySFH(ProspectorModelBuilder):
    """
    Builds a Continuity SFH model reading all configuration from a FitConfig object.
    Drop-in replacement for the old dict-based ContinuitySFH, with identical parameters.
    """

    def __init__(self, config):
        super().__init__()

        # --- Unpack FitConfig into local variables ---
        # This mirrors the old dict-based interface exactly, making diffs easy to audit.
        fit_spec = config.use_spectroscopy
        add_duste = config.add_duste
        add_nebular = config.add_nebular
        nbins = config.nbins
        fixed_z = config.fixed_z
        fit_outliers_spec = config.fit_outliers_spec
        fit_outliers_photo = config.fit_outliers_photo
        add_dust1 = config.add_dust1
        add_agn = config.add_agn
        margin_elines = getattr(config, "margin_elines", False)
        fit_eline_redshift = getattr(config, "fit_eline_redshift", False)

        # Redshift: None means it was not provided (free param from broad prior)
        z = config.redshift
        has_z = z is not None

        # ------------------------------------------------------------------
        # 1. SPECTROSCOPY
        # ------------------------------------------------------------------
        if fit_spec:
            self.model_params.update(TemplateLibrary["spectral_smoothing"])
            self.model_params["sigma_smooth"] = {
                "N": 1,
                "init": 1000.0,
                "isfree": True,
                "units": "Km/s",
                "prior": priors.TopHat(mini=200.0, maxi=2000.0),
            }

            self.model_params.update(TemplateLibrary["optimize_speccal"])
            self.model_params["polyorder"] = {"N": 1, "isfree": False, "init": 10}
            self.model_params["spec_norm"] = {
                "N": 1,
                "isfree": True,
                "init": 1,
                "units": "f_true/f_obs",
                "prior": priors.Normal(mean=1.0, sigma=0.1),
            }
            self.model_params["spec_jitter"] = {
                "N": 1,
                "isfree": True,
                "init": 1,
                "prior": priors.TopHat(mini=-0.5, maxi=5),
            }

        # ------------------------------------------------------------------
        # 2. OUTLIERS
        # ------------------------------------------------------------------
        if fit_outliers_spec:
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

        if fit_outliers_photo:
            self.model_params["f_outlier_phot"] = {
                "N": 1,
                "isfree": True,
                "init": 0.00,
                "prior": priors.TopHat(mini=0, maxi=0.5),
            }
            self.model_params["nsigma_outlier_phot"] = {
                "N": 1,
                "isfree": False,
                "init": 50.0,
            }

        # ------------------------------------------------------------------
        # 3. DUST EMISSION (IR)
        # ------------------------------------------------------------------
        if add_duste:
            self.model_params.update(TemplateLibrary["dust_emission"])
            self.model_params["duste_gamma"]["isfree"] = True
            self.model_params["duste_gamma"]["init"] = 0.01
            self.model_params["duste_gamma"]["prior"] = priors.TopHat(
                mini=0.0, maxi=1.0
            )
            self.model_params["duste_qpah"]["isfree"] = True
            self.model_params["duste_qpah"]["init"] = 3.5
            self.model_params["duste_qpah"]["prior"] = priors.TopHat(
                mini=0.5, maxi=10.0
            )
            self.model_params["duste_umin"]["isfree"] = True
            self.model_params["duste_umin"]["init"] = 1.0
            self.model_params["duste_umin"]["prior"] = priors.TopHat(
                mini=0.1, maxi=25.0
            )

        # ------------------------------------------------------------------
        # 4. NEBULAR EMISSION
        # ------------------------------------------------------------------
        if add_nebular:
            self.model_params.update(TemplateLibrary["nebular"])
            self.model_params["nebemlineinspec"] = {
                "N": 1,
                "isfree": False,
                "init": False,
            }
            self.model_params["eline_sigma"] = {
                "N": 1,
                "isfree": True,
                "init": 150.0,
                "units": "km/s",
                "prior": priors.TopHat(mini=50, maxi=3000),
            }
            self.model_params["gas_logz"] = {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "units": "log Z/Zsun",
                "prior": priors.TopHat(mini=-2, maxi=0.5),
            }
            self.model_params["gas_logu"] = {
                "N": 1,
                "isfree": True,
                "init": -2.0,
                "units": "Q_H/N_H",
                "prior": priors.TopHat(mini=-4, maxi=-1),
            }

        # ------------------------------------------------------------------
        # 5. CONTINUITY SFH  (must come after nebular to avoid key conflicts)
        # ------------------------------------------------------------------
        self.model_params.update(TemplateLibrary["continuity_sfh"])
        tuniv = 13.7
        self.model_params.update(
            adjust_continuity_agebins(self.model_params, tuniv=tuniv, nbins=nbins)
        )

        self.model_params["mass"] = {
            "N": nbins,
            "isfree": False,
            "init": 1e6,
            "units": "Solar masses formed",
            "depends_on": transforms.logsfr_ratios_to_masses,
        }

        # ------------------------------------------------------------------
        # 6. PHYSICAL PARAMETERS
        # ------------------------------------------------------------------
        self.model_params["logzsol"] = {
            "N": 1,
            "isfree": True,
            "init": -0.3,
            "units": r"$\log (Z/Z_\odot)$",
            "prior": priors.TopHat(mini=-2, maxi=0.50),
        }
        if fixed_z:
            self.model_params["logzsol"]["isfree"] = False

        self.model_params["imf_type"] = {
            "N": 1,
            "isfree": False,
            "init": 1,  # 1 = Chabrier
            "units": "FSPS index",
            "prior": None,
        }

        self.model_params["logmass"] = {
            "N": 1,
            "isfree": True,
            "init": 10.5,
            "units": "Solar masses formed",
            "prior": priors.TopHat(mini=8.0, maxi=13.0),
        }

        # ------------------------------------------------------------------
        # 7. DUST ABSORPTION
        # ------------------------------------------------------------------
        self.model_params["dust_type"] = {
            "N": 1,
            "isfree": False,
            "init": 4,
            "units": "FSPS index",
            "prior": None,
        }
        self.model_params["dust2"] = {
            "N": 1,
            "isfree": True,
            "init": 0.5,
            "units": "optical depth at 5500AA",
            "prior": priors.TopHat(mini=0.0, maxi=4.0 / 1.086),
        }
        self.model_params["dust_index"] = {
            "N": 1,
            "isfree": True,
            "init": 0.0,
            "units": "power-law multiplication of Calzetti",
            "prior": priors.ClippedNormal(mini=-1.5, maxi=0.4, mean=0.0, sigma=0.3),
        }

        if add_dust1:
            self.model_params["dust1"] = {
                "N": 1,
                "isfree": False,
                "depends_on": transforms.dustratio_to_dust1,
                "init": 0.0,
                "units": "optical depth towards young stars",
                "prior": None,
            }
            self.model_params["dust1_fraction"] = {
                "N": 1,
                "isfree": True,
                "init": 1.0,
                "units": "ratio Dust2 to Dust1",
                "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3),
            }

        # ------------------------------------------------------------------
        # 8. REDSHIFT
        # ------------------------------------------------------------------
        if has_z:
            self.model_params["zred"] = {
                "N": 1,
                "isfree": True,
                "init": z,
                "units": "redshift",
                "prior": priors.ClippedNormal(
                    mini=z - 0.5,
                    maxi=z + 0.5,
                    mean=z,
                    sigma=0.05,
                ),
            }
        else:
            # No redshift in config: fit freely over a broad prior
            self.model_params["zred"] = {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "units": "redshift",
                "prior": priors.TopHat(mini=0.0, maxi=12),
            }

        # ------------------------------------------------------------------
        # 9. EMISSION LINE MARGINALIZATION (optional)
        # ------------------------------------------------------------------
        if add_agn:
            self.model_params.update(TemplateLibrary["agn"])
            self.model_params["fagn"]["isfree"] = True
            self.model_params["agn_tau"]["isfree"] = True

            self.model_params.update(TemplateLibrary["agn_eline"])

            self.model_params["agn_elum"]["isfree"] = True
            self.model_params["agn_eline_sigma"] = {
                "N": 1,
                "isfree": True,
                "init": 5000,
                "prior": priors.TopHat(mini=3000, maxi=9000),
            }
        if margin_elines:
            if not add_nebular:
                raise ValueError("Cannot marginalize without nebular added")

            lines_dict = getattr(config, "lines", {})
            if not lines_dict:
                raise KeyError("margin_elines=True but no lines found in config.")

            to_fit = list(lines_dict.keys())
            to_fit_exist = np.array(to_fit.copy())

            self.model_params.update(TemplateLibrary["nebular_marginalization"])

            self.model_params["elines_to_fit"]["init"] = to_fit_exist
            self.model_params["eline_sigma"] = {
                "N": 1,
                "isfree": True,
                "init": 1e3,
                "units": "eline sigma",
                "prior": priors.TopHat(mini=200.0, maxi=2000.0),
            }

            if fit_eline_redshift:
                self.model_params["fit_eline_redshift"] = {
                    "N": 1,
                    "isfree": False,
                    "init": True,
                }
                self.model_params["eline_delta_zred"] = {
                    "N": len(to_fit),
                    "isfree": True,
                    "init": np.array([100 for _ in range(len(to_fit))]),
                    "units": "eline sigma",
                    "prior": priors.TopHat(
                        mini=np.array([0 for _ in range(len(to_fit))]),
                        maxi=np.array([500 for _ in range(len(to_fit))]),
                    ),
                }


class AmirModel(ProspectorModelBuilder):
    """
    Prospector model matching the externally provided Amir specification.
    """

    ADD_EM_NEB = True
    ADD_EM_DUST = True
    MASS_INIT = 10.2
    MASS_STD = 2
    ZDELTA = 0.5

    def __init__(self, config):
        super().__init__()
        self.config = config

        zred = getattr(config, "redshift", None)
        if zred is None:
            raise ValueError("redshift must be specified for AmirModel")

        add_duste = getattr(config, "add_duste", self.ADD_EM_DUST)
        add_nebular = getattr(config, "add_nebular", self.ADD_EM_NEB)
        add_agn = getattr(config, "add_agn", False)
        z_is_free = not getattr(config, "fixed_z", False)
        fit_spec = config.use_spectroscopy
        
        
        self.model_params["zred"] = {
            "N": 1,
            "isfree": z_is_free,
            "init": zred,
            "units": "redshift",
            "prior": priors.TopHat(
                mini=zred - self.ZDELTA,
                maxi=zred + self.ZDELTA,
            ),
        }

        self.model_params["logzsol"] = {
            "N": 1,
            "isfree": True,
            "init": -0.5,
            "units": r"$\log (Z/Z_\odot)$",
            "prior": priors.TopHat(mini=-2, maxi=0.50),
        }

        self.model_params["logt_wmb_hot"] = {
            "N": 1,
            "isfree": False,
            "init": 8.0,
            "units": r"$\log (Z/Z_\odot)$",
            "prior": priors.TopHat(mini=6, maxi=10),
        }

        self.model_params["f_outlier_phot"] = {
            "N": 1,
            "isfree": True,
            "init": 0.00,
            "prior": priors.TopHat(mini=0.0, maxi=0.3),
        }

        self.model_params["nsigma_outlier_phot"] = {
            "N": 1,
            "isfree": False,
            "init": 50.0,
        }

        tuniv = Planck18.age(zred).value
        agelims_myr = np.append(
            np.logspace(np.log10(30.0), np.log10(0.8 * tuniv * 1000), 12),
            [0.9 * tuniv * 1000, tuniv * 1000],
        )
        agelims = np.concatenate(([0.0], np.log10(agelims_myr * 1e6)))
        agebins = np.array([agelims[:-1], agelims[1:]]).T
        nbins = len(agelims) - 1

        self.model_params["logmass"] = {
            "N": 1,
            "isfree": True,
            "init": self.MASS_INIT,
            "units": "Solar masses formed",
            "prior": priors.TopHat(
                mini=self.MASS_INIT - self.MASS_STD,
                maxi=self.MASS_INIT + self.MASS_STD,
            ),
        }

        self.model_params["mass"] = {
            "N": nbins,
            "isfree": False,
            "init": (10 ** 10.5) / nbins,
            "units": "Solar masses formed",
            "depends_on": transforms.logsfr_ratios_to_masses,
        }

        self.model_params["agebins"] = {
            "N": nbins,
            "isfree": False,
            "init": agebins,
            "units": "log(yr)",
        }

        self.model_params["logsfr_ratios"] = {
            "N": nbins - 1,
            "isfree": True,
            "init": np.full(nbins - 1, 0.0),
            "units": "",
            "prior": priors.StudentT(
                mean=np.full(nbins - 1, 0.0),
                scale=np.full(nbins - 1, 0.3),
                df=np.full(nbins - 1, 2),
            ),
        }

        self.model_params["imf_type"] = {
            "N": 1,
            "isfree": False,
            "init": 1,
            "units": "FSPS index",
            "prior": None,
        }

        self.model_params["dust_type"] = {
            "N": 1,
            "isfree": False,
            "init": 4,
            "units": "FSPS index",
        }

        self.model_params["dust2"] = {
            "N": 1,
            "isfree": True,
            "init": 1.0,
            "units": "optical depth at 5500AA",
            "prior": priors.TopHat(mini=0.0, maxi=4.0 / 1.086),
        }

        self.model_params["dust_index"] = {
            "N": 1,
            "isfree": True,
            "init": 0.0,
            "units": "power-law multiplication of Calzetti",
            "prior": priors.ClippedNormal(
                mini=-1.5,
                maxi=0.4,
                mean=0.0,
                sigma=0.1,
            ),
        }

        self.model_params["dust1"] = {
            "N": 1,
            "isfree": False,
            "depends_on": transforms.dustratio_to_dust1,
            "init": 0,
            "units": "optical depth towards young stars",
            "prior": None,
        }

        self.model_params["dust1_fraction"] = {
            "N": 1,
            "isfree": True,
            "init": 1.0,
            "prior": priors.ClippedNormal(
                mini=0.0,
                maxi=2.0,
                mean=1.0,
                sigma=0.1,
            ),
        }

        if add_duste:
            self.model_params.update(TemplateLibrary["dust_emission"])
            self.model_params["duste_gamma"]["isfree"] = True
            self.model_params["duste_gamma"]["init"] = 0.1
            self.model_params["duste_gamma"]["prior"] = priors.TopHat(
                mini=0.0,
                maxi=1.0,
            )

            self.model_params["duste_qpah"]["isfree"] = True
            self.model_params["duste_qpah"]["init"] = 3.0
            self.model_params["duste_qpah"]["prior"] = priors.TopHat(
                mini=0.5,
                maxi=10.0,
            )

            self.model_params["duste_umin"]["isfree"] = True
            self.model_params["duste_umin"]["init"] = 1.0
            self.model_params["duste_umin"]["prior"] = priors.TopHat(
                mini=0.1,
                maxi=25.0,
            )

        if add_agn:
            self.model_params.update(TemplateLibrary["agn"])
            self.model_params["fagn"]["isfree"] = True
            self.model_params["fagn"]["prior"] = priors.LogUniform(
                mini=1e-5,
                maxi=3.0,
            )
            self.model_params["agn_tau"]["isfree"] = True
            self.model_params["agn_tau"]["prior"] = priors.LogUniform(
                mini=5.0,
                maxi=150.0,
            )

        # if add_nebular:
        #     self.model_params.update(TemplateLibrary["nebular"])
        #     self.model_params["gas_logu"]["isfree"] = True
        #     self.model_params["gas_logz"]["isfree"] = True
        #     self.model_params["nebemlineinspec"] = {
        #         "N": 1,
        #         "isfree": False,
        #         "init": True,
        #     }
        #     self.model_params["gas_logz"].pop("depends_on", None)
        
        if fit_spec:
            self.model_params.update(TemplateLibrary["spectral_smoothing"])
            self.model_params["sigma_smooth"] = {
                "N": 1,
                "init": 1000.0,
                "isfree": True,
                "units": "Km/s",
                "prior": priors.TopHat(mini=200.0, maxi=2000.0),
            }

            self.model_params.update(TemplateLibrary["optimize_speccal"])
            self.model_params["polyorder"] = {"N": 1, "isfree": False, "init": 10}
            self.model_params["spec_norm"] = {
                "N": 1,
                "isfree": True,
                "init": 1,
                "units": "f_true/f_obs",
                "prior": priors.Normal(mean=1.0, sigma=0.1),
            }
            self.model_params["spec_jitter"] = {
                "N": 1,
                "isfree": True,
                "init": 1,
                "prior": priors.TopHat(mini=-0.5, maxi=5),
            }
        
        if add_nebular:
            self.model_params.update(TemplateLibrary["nebular"])
            self.model_params["nebemlineinspec"] = {
                "N": 1,
                "isfree": False,
                "init": False,
            }
            self.model_params["eline_sigma"] = {
                "N": 1,
                "isfree": True,
                "init": 150.0,
                "units": "km/s",
                "prior": priors.TopHat(mini=50, maxi=3000),
            }
            self.model_params["gas_logz"] = {
                "N": 1,
                "isfree": True,
                "init": 0.0,
                "units": "log Z/Zsun",
                "prior": priors.TopHat(mini=-2, maxi=0.5),
            }
            self.model_params["gas_logu"] = {
                "N": 1,
                "isfree": True,
                "init": -2.0,
                "units": "Q_H/N_H",
                "prior": priors.TopHat(mini=-4, maxi=-1),
            }
            

            
# class ContinuitySFH(ProspectorModelBuilder):
#     def __init__(self, run_params, obs=None):
#         super().__init__()

#         fit_spec = True#run_params.get("use_spectroscopy", False)
#         add_duste = True#run_params.get("add_duste", False)
#         add_nebular = True#run_params.get("add_nebular", False)
#         has_z = True#"redshift" in run_params
#         nbins = 8#run_params.get("nbins", 12)
#         fixed_z = False#run_params.get("fixed_z", False)

#         fit_outliers_spec = False#run_params.get("fit_outliers_spec", False)
#         fit_ouliers_photo = False#run_params.get("fit_ouliers_photo", False)
#         add_dust1 = True#run_params.get("add_dust1", False)

#         smooth_type = "vel"#run_params.get("smooth_type", "vel")
#         margin_elines = False#run_params.get("margin_elines", False)
#         fit_eline_redshift = False#run_params.get("fit_eline_redshift", False)

#         # cosmo = Planck18

#         if fit_spec:
#             # --- Spectral Smoothing ---
#             self.model_params.update(TemplateLibrary["spectral_smoothing"])
#             self.model_params["sigma_smooth"] = {
#                 "N": 1,
#                 "init": 1000.0,
#                 "isfree": True,
#                 "units": "Km/s",
#                 "prior": priors.TopHat(mini=200.0, maxi=2000.0),
#             }
#             # R prisma ~ 100
#             # --- Continuum Optimization ---
#             self.model_params.update(TemplateLibrary["optimize_speccal"])
#             self.model_params["polyorder"] = {"N": 1, "isfree": False, "init": 10}
#             self.model_params["spec_norm"] = {
#                 "N": 1,
#                 "isfree": True,
#                 "init": 1,
#                 "units": "f_true/f_obs",
#                 "prior": priors.Normal(mean=1.0, sigma=0.1),
#             }
#             self.model_params["spec_jitter"] = {
#                 "N": 1,
#                 "isfree": True,
#                 "init": 1,
#                 "prior": priors.TopHat(mini=-0.5, maxi=5),
#             }

#             # This is a pixel outlier model. It helps to marginalize over
#             # poorly modeled noise, such as residual sky lines or
#             # even missing absorption lines
#             # self.model_params["smooth_type"]["init"] = smooth_type

#         if fit_outliers_spec:
#             self.model_params["f_outlier_spec"] = {
#                 "N": 1,
#                 "isfree": True,
#                 "init": 0.01,
#                 "prior": priors.TopHat(mini=1e-5, maxi=0.2),
#             }

#             self.model_params["nsigma_outlier_spec"] = {
#                 "N": 1,
#                 "isfree": False,
#                 "init": 50.0,
#             }

#         if fit_ouliers_photo:
#             self.model_params["f_outlier_phot"] = {
#                 "N": 1,
#                 "isfree": True,
#                 "init": 0.00,
#                 "prior": priors.TopHat(mini=0, maxi=0.5),
#             }

#             self.model_params["nsigma_outlier_phot"] = {
#                 "N": 1,
#                 "isfree": False,
#                 "init": 50.0,
#             }

#         # -----------------------
#         # --- ADDITIONALS HERE ---
#         # -----------------------
#         # because by def they can override manual instertions
#         if add_duste:
#             # --- Dust Emission ---
#             self.model_params.update(TemplateLibrary["dust_emission"])
#             self.model_params["duste_gamma"]["isfree"] = True
#             self.model_params["duste_gamma"]["init"] = 0.01
#             self.model_params["duste_gamma"]["prior"] = priors.TopHat(
#                 mini=0.0, maxi=1.0
#             )

#             self.model_params["duste_qpah"]["isfree"] = True
#             self.model_params["duste_qpah"]["init"] = 3.5
#             self.model_params["duste_qpah"]["prior"] = priors.TopHat(
#                 mini=0.5, maxi=10.0
#             )

#             self.model_params["duste_umin"]["isfree"] = True
#             self.model_params["duste_umin"]["init"] = 1.0
#             self.model_params["duste_umin"]["prior"] = priors.TopHat(
#                 mini=0.1, maxi=25.0
#             )

#         if add_nebular:
#             # add_nebular

#             self.model_params.update(TemplateLibrary["nebular"])
#             self.model_params["nebemlineinspec"] = {
#                 "N": 1,
#                 "isfree": False,
#                 "init": False,
#             }
#             self.model_params["eline_sigma"] = {
#                 "N": 1,
#                 "isfree": True,
#                 "init": 150.0,
#                 "units": "km/s",
#                 "prior": priors.TopHat(mini=50, maxi=3000),
#             }
#             self.model_params["gas_logz"] = {
#                 "N": 1,
#                 "isfree": True,
#                 "init": 0.0,
#                 "units": "log Z/Zsun",
#                 "prior": priors.TopHat(mini=-2, maxi=0.5),
#             }
#             self.model_params["gas_logu"] = {
#                 "N": 1,
#                 "isfree": True,
#                 "init": -2.0,
#                 "units": "Q_H/N_H",
#                 "prior": priors.TopHat(mini=-4, maxi=-1),
#             }

#         self.model_params.update(TemplateLibrary["continuity_sfh"])
#         tuniv = 13.7  # cosmo.age(redshift).value
#         self.model_params.update(
#             adjust_continuity_agebins(self.model_params, tuniv=tuniv, nbins=nbins)
#         )
#         # nbins = 12
#         # agelims = [0.0, 7.0] + np.linspace(
#         #     start=7,
#         #     stop=np.log10((cosmo.age(z=redshift) - cosmo.age(z=50)).to("yr").value),
#         #     num=nbins,
#         # )[1:].tolist()
#         # agebins = np.array([agelims[:-1], agelims[1:]]).T

#         # self.model_params["agebins"] = {
#         #     "N": nbins,
#         #     "isfree": False,
#         #     "init": agebins,
#         #     "units": "log(yr)",
#         # }

#         # self.model_params["logsfr_ratios"] = {
#         #     "N": nbins - 1,
#         #     "isfree": True,
#         #     "init": np.full(nbins - 1, 0.0),
#         #     "prior": priors.StudentT(
#         #         mean=np.full(nbins - 1, 0.0),
#         #         scale=np.full(nbins - 1, 0.3),
#         #         df=np.full(nbins - 1, 2),
#         #     ),
#         # }

#         self.model_params["mass"] = {
#             "N": nbins,
#             "isfree": False,
#             "init": 1e6,  # np.array([10**6.5 for _ in range(nbins)]),
#             "units": "Solar masses formed",
#             "depends_on": transforms.logsfr_ratios_to_masses,
#         }

#         # --- Metallicity ---
#         self.model_params["logzsol"] = {
#             "N": 1,
#             "isfree": True,
#             "init": -0.3,
#             "units": r"$\log (Z/Z_\odot)$",
#             "prior": priors.TopHat(mini=-2, maxi=0.50),
#         }
#         if fixed_z:
#             self.model_params["logzsol"]["isfree"] = False

#         # --- IMF ---
#         self.model_params["imf_type"] = {
#             "N": 1,
#             "isfree": False,
#             "init": 1,  # 1 = Chabrier
#             "units": "FSPS index",
#             "prior": None,
#         }

#         # --- Continuity SFH ---
#         self.model_params["logmass"] = {
#             "N": 1,
#             "isfree": True,
#             "init": 10.5,
#             "units": "Solar masses formed",
#             "prior": priors.TopHat(mini=8.0, maxi=13.0),
#         }

#         # --- Dust Absorption / Emission ---
#         self.model_params["dust_type"] = {
#             "N": 1,
#             "isfree": False,
#             "init": 4,
#             "units": "FSPS index",
#             "prior": None,
#         }
#         self.model_params["dust2"] = {
#             "N": 1,
#             "isfree": True,
#             "init": 0.5,
#             "units": "optical depth at 5500AA",
#             "prior": priors.TopHat(mini=0.0, maxi=4.0 / 1.086),
#         }

#         self.model_params["dust_index"] = {
#             "N": 1,
#             "isfree": True,
#             "init": 0.0,
#             "units": "power-law multiplication of Calzetti",
#             "prior": priors.ClippedNormal(mini=-1.5, maxi=0.4, mean=0.0, sigma=0.3),
#         }
#         if add_dust1:
#             self.model_params["dust1"] = {
#                 "N": 1,
#                 "isfree": False,
#                 "depends_on": transforms.dustratio_to_dust1,
#                 "init": 0.0,
#                 "units": "optical depth towards young stars",
#                 "prior": None,
#             }
#             self.model_params["dust1_fraction"] = {
#                 "N": 1,
#                 "isfree": True,
#                 "init": 1.0,
#                 "units": "ratio Dust2 to Dust1",
#                 "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3),
#             }

#         # --- Redshift ---
#         if has_z:
#             redshift = run_params.redshift#run_params["redshift"]
#             # print(F"\nINIT MODEL AT Z = {redshift}\n")
#             self.model_params["zred"] = {
#                 "N": 1,
#                 "isfree": True,
#                 "init": redshift,
#                 "units": "redshift",
#                 "prior": priors.ClippedNormal(
#                     mini=redshift - 0.5,
#                     maxi=redshift + 0.5,
#                     mean=redshift,
#                     sigma=0.05,
#                 ),
#             }
#         elif has_z is False:
#             redshift = 0.0
#             # print("\nINIT MODEL WITH Z AS FREE PARAM\n")
#             self.model_params["zred"] = {
#                 "N": 1,
#                 "isfree": True,
#                 "init": redshift,
#                 "units": "redshift",
#                 "prior": priors.TopHat(mini=0.0, maxi=12),
#             }

#         if margin_elines:
#             if not add_nebular:
#                 raise ValueError("Cannot marginalize without nebular added")

#             if "elines_to_fit" not in run_params:
#                 raise KeyError("eline_to_fit dict is not present in run_params")

#             # self.model_params['gas_logu']['isfree'] = True
#             # self.model_params['gas_logz']['isfree'] = True

#             # self.model_params['nebemlineinspec'] = {'N': 1,
#             #                                 'isfree': False,
#             #                                 'init': False}

#             # self.model_params["gas_logz"].pop("depends_on")

#             self.model_params.update(TemplateLibrary["nebular_marginalization"])

#             # # no NOT use cloudy line prior
#             # self.model_params['use_eline_prior'] = {'N': 1,
#             #                                         'is_free': False,
#             #                                         'init': False}

#             # self.model_params['eline_sigma'] = {"N": 1,
#             #                             "isfree": True,
#             #                             "init": 1000.0,
#             #                             "prior": priors.TopHat(mini=300, maxi=4000)}

#             to_fit = list(run_params["elines_to_fit"].keys())
#             to_fit_exist = np.array(to_fit.copy())
#             # ['[S III] 3722', '[O II] 3726', '[O II] 3729', 'Ba-7 3835', 'Ba-6 3889', \
#             #         'Ba-5 3970','Ba-delta 4101.76A', 'Ba-gamma 4341', '[O III] 4363', 'He I 4471.49A', \
#             #             '[C  I] 4621', '[Ne IV] 4720', '[O III] 4959', '[O III] 5007', \
#             #             '[N II] 6548', '[N II] 6584']

#             # emi_wavelengths = [line for line in list(run_params["elines_to_fit"].values())]

#             # obs_wave_mask = obs['wavelength'][obs['mask']]
#             # obs_spec_mask = obs['spectrum'][obs['mask']]
#             # obs_unc_mask = obs['unc'][obs['mask']]

#             # for i, w_emi in enumerate(emi_wavelengths):
#             #     wave_emi = obs_wave_mask[abs((obs_wave_mask/(1+redshift) - w_emi)/w_emi) < 10_000/3e5]
#             #     spec_emi = obs_spec_mask[abs((obs_wave_mask/(1+redshift) - w_emi)/w_emi) < 10_000/3e5]
#             #     unc_emi  = obs_unc_mask[abs((obs_wave_mask/(1+redshift) - w_emi)/w_emi) < 10_000/3e5]

#             #     print(wave_emi, spec_emi, unc_emi)
#             #     if len(wave_emi) < 5:
#             #         print('remove', str(to_fit[i]), 'due to no pixels, pixel num=', str(len(wave_emi)))
#             #         to_fit_exist.remove(to_fit[i])
#             #         # keep in mind that the length of "to_fit_exist" changes with the for loop

#             #     elif len(spec_emi[np.isnan(spec_emi)]) > 0 or len(unc_emi[np.isnan(unc_emi)]) > 0 or len(unc_emi[unc_emi == 0]) > 0 \
#             #             or len(spec_emi[spec_emi < 0]) > 0 or len(unc_emi[unc_emi <=0]) > 0:
#             #         print('remove', str(to_fit[i]), 'due to BAD pixels')
#             #         to_fit_exist.remove(to_fit[i])

#             self.model_params["elines_to_fit"]["init"] = to_fit_exist
#             self.model_params["eline_sigma"] = {
#                 "N": 1,
#                 "isfree": True,
#                 "init": 1e3,  # np.array([10**6.5 for _ in range(nbins)]),
#                 "units": "eline sigma",
#                 "prior": priors.TopHat(mini=200.0, maxi=2000.0),
#             }
#             # eline_prior_width

#             # One per Line
#             if fit_eline_redshift:
#                 self.model_params["fit_eline_redshift"] = {
#                     "N": 1,
#                     "isfree": False,
#                     "init": True,
#                 }
#                 self.model_params["eline_delta_zred"] = {
#                     "N": len(to_fit),
#                     "isfree": True,
#                     "init": np.array(
#                         [100 for _ in range(len(to_fit))]
#                     ),  # np.array([10**6.5 for _ in range(nbins)]),
#                     "units": "eline sigma",
#                     "prior": priors.TopHat(
#                         mini=np.array([0 for _ in range(len(to_fit))]),
#                         maxi=np.array([500 for _ in range(len(to_fit))]),
#                     ),
#                 }
#             pass

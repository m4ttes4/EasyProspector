import sys
import os
import argparse
from typing import Dict, Optional, Any
from dataclasses import dataclass, field, asdict
import logging
import math
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)

DEFAULT_EMISSION_LINES = {
        "Lyα": (1215.67,),
        "[O II]": (3727.00,),
        "Hβ": (4861.33,),
        "[O III] 4959": (4958.91,),
        "[O III] 5007": (5006.84,),
        "Hα": (6562.82,),
        "[N II] 6584": (6583.45,),
        "[S II] 6716": (6716.44,),
        "[S II] 6731": (6730.82,),
        "Na": (5895,)
}

@dataclass
class FitConfig:
    # --- 1. Identifiers and Paths (I/O) ---

    file: Optional[str] = None  # Full path to the input .h5 file
    out: Optional[str] = None  # Output name
    out_folder: Optional[str] = "results/out"  # Folder where results will be saved
    file_list: Optional[str] = None
    ext: Optional[str] = None #extesion for the name

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
    model_type: str = "AmirModel"
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
            "nested_nlive_init": 600,
            "nested_sample": "rwalk",
            "nested_target_n_effective": 1000,
            "nested_dlogz_init": 0.01,
        }
    )

    # --- 5. Debug, Plotting, and Interactivity ---
    verbose: bool = True
    debug: bool = False
    interactive: bool = False

    lines: Dict[str, tuple] = field(default_factory=lambda: dict(DEFAULT_EMISSION_LINES))

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

        if self.log_folder:
            self.log_folder = os.path.expanduser(self.log_folder)

        self.validate(require_inputs=False)

        if self.file and self.file_list:
            raise ValueError("Use either --file or --file-list, not both.")

        if self.file_list and not os.path.exists(self.file_list):
            raise FileNotFoundError(f"File list does not exist: {self.file_list}")

        if self.file_list:
            # If a list is provided, read the file and ignore empty lines
            with open(self.file_list, "r") as f:
                self.targets = [
                    os.path.expanduser(line.strip()) for line in f if line.strip()
                ]

        elif self.file:
            self.targets = [self.file]

    def validate(self, require_inputs: bool = False):
        """Validate configuration values that do not require opening data files."""
        if require_inputs and not (self.file or self.file_list):
            raise ValueError("Provide either --file or --file-list.")

        if not self.use_photometry and not self.use_spectroscopy:
            raise ValueError("At least one data component must be enabled.")

        if self.interactive and not self.use_spectroscopy:
            raise ValueError("--interactive requires spectroscopy to be enabled.")

        if not isinstance(self.nbins, int) or self.nbins < 2:
            raise ValueError("nbins must be an integer >= 2.")

        if not isinstance(self.z_continuous, int) or self.z_continuous not in (0, 1, 2):
            raise ValueError("z_continuous must be one of 0, 1, or 2.")

        if self.redshift is not None:
            if not math.isfinite(self.redshift) or self.redshift < 0:
                raise ValueError("redshift must be a finite value >= 0.")

        if self.out_folder is not None and not str(self.out_folder).strip():
            raise ValueError("out_folder cannot be empty.")

        if self.logging_to_file and not str(self.log_folder).strip():
            raise ValueError("log_folder cannot be empty when file logging is enabled.")

        if not str(self.model_type).strip():
            raise ValueError("model_type cannot be empty.")

        if self.emcee:
            raise ValueError("emcee is parsed for compatibility but is not wired in run.py.")

        if not self.dynesty:
            raise ValueError("--no-dynesty is not supported because result writing expects sampling output.")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the configuration into a dictionary (e.g., for Prospector)."""
        data = asdict(self)
        data.update(self.dynesty_kwargs)
        return data

    def show_help(self):
        """Render a concise Rich guide for command-line usage."""
        console = Console()

        title = Text("EasyProspector", style="bold cyan")
        subtitle = Text(
            "Prospector SED fitting from validated HDF5 photometry/spectroscopy.",
            style="white",
        )
        header = Panel(
            Group(title, subtitle),
            border_style="cyan",
            box=box.DOUBLE,
            padding=(1, 2),
        )

        quick_start = Table(
            title="Start Here",
            box=box.ROUNDED,
            header_style="bold cyan",
            show_lines=False,
        )
        quick_start.add_column("Task", style="bold white", no_wrap=True)
        quick_start.add_column("Command", style="green")
        quick_start.add_row(
            "Single target",
            "python run.py --file data/galaxy.h5 --redshift 1.23",
        )
        quick_start.add_row(
            "Photometry only",
            "python run.py --file data/galaxy.h5 --no-spectroscopy --redshift 1.23",
        )
        quick_start.add_row(
            "Batch list",
            "python run.py --file-list targets.txt --out-folder results/out",
        )
        quick_start.add_row(
            "Choose model",
            "python run.py --file data/galaxy.h5 --model ContinuitySFH",
        )

        inputs = Table(
            title="Inputs and Outputs",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
        )
        inputs.add_column("Option", style="bold yellow", no_wrap=True)
        inputs.add_column("Default", style="magenta", no_wrap=True)
        inputs.add_column("Meaning", style="white")
        inputs.add_row("--file PATH", "None", "Fit one HDF5 target.")
        inputs.add_row("--file-list PATH", "None", "Fit one HDF5 path per line.")
        inputs.add_row("--out-folder DIR", self.out_folder, "Directory for result HDF5 files.")
        inputs.add_row("--out NAME", "input basename", "Output basename for a single target.")
        inputs.add_row("--version NAME", self.version, "HDF5 group to read, for example V1.")

        data = Table(title="Data Selection", box=box.SIMPLE_HEAVY, header_style="bold cyan")
        data.add_column("Option", style="bold yellow", no_wrap=True)
        data.add_column("Default", style="magenta", no_wrap=True)
        data.add_column("Meaning", style="white")
        data.add_row("--photometry / --no-photometry", str(self.use_photometry), "Use photometric data.")
        data.add_row("--spectroscopy / --no-spectroscopy", str(self.use_spectroscopy), "Use spectroscopic data.")
        data.add_row("--use-mask / --no-use-mask", str(self.use_mask), "Use HDF5 mask datasets.")
        data.add_row("--filter-photo / --no-filter-photo", str(self.filter_photo), "Mask invalid photometric points.")
        data.add_row("--filter-spec / --no-filter-spec", str(self.filter_spec), "Mask invalid spectral pixels.")

        model = Table(title="Model", box=box.SIMPLE_HEAVY, header_style="bold cyan")
        model.add_column("Option", style="bold yellow", no_wrap=True)
        model.add_column("Default", style="magenta", no_wrap=True)
        model.add_column("Meaning", style="white")
        model.add_row("--model NAME", self.model_type, "Registered model: AmirModel, ContinuitySFH, BaseModel.")
        model.add_row("--redshift Z", "metadata or None", "Overrides V1/Metadata/redshift.")
        model.add_row("--fixed-z / --no-fixed-z", str(self.fixed_z), "Fix or fit zred.")
        model.add_row("--nbins N", str(self.nbins), "Continuity SFH age bins.")
        model.add_row("--sigmav / --no-sigmav", str(self.add_sigmav), "Apply LSF smoothing when possible.")
        model.add_row("--dispersion-file PATH", "None", "JWST dispersion FITS file for LSF.")

        fitting = Table(title="Fitting and Logs", box=box.SIMPLE_HEAVY, header_style="bold cyan")
        fitting.add_column("Option", style="bold yellow", no_wrap=True)
        fitting.add_column("Default", style="magenta", no_wrap=True)
        fitting.add_column("Meaning", style="white")
        fitting.add_row("--dynesty", str(self.dynesty), "Run Dynesty nested sampling.")
        fitting.add_row("--optimize / --no-optimize", str(self.optimize), "Run Prospector optimization.")
        fitting.add_row("--verbose / --no-verbose", str(self.verbose), "Show rich data/model summary tables.")
        fitting.add_row("--debug / --no-debug", str(self.debug), "Enable debug-level logs.")
        fitting.add_row("--log-to-file", str(self.logging_to_file), "Write per-target logs to --log-folder.")
        fitting.add_row("--interactive", str(self.interactive), "Open spectral masking GUI before fitting.")

        notes = Table.grid(padding=(0, 1))
        notes.add_column(style="bold cyan", no_wrap=True)
        notes.add_column(style="white")
        notes.add_row("Boolean flags", "Every displayed flag also accepts --no-<flag> when disabling is meaningful.")
        notes.add_row("Legacy aliases", "Underscore spellings such as --file_list and --out_folder still work.")
        notes.add_row("Default model", "AmirModel requires --redshift or V1/Metadata/redshift.")
        notes.add_row("Validation", "Requested HDF5 components must exist and have matching 1D array lengths.")

        schema = Text()
        schema.append("galaxy.h5\n", style="bold white")
        schema.append("└── V1/                         ", style="cyan")
        schema.append("selected by --version\n", style="dim")
        schema.append("    ├── Photometry/\n", style="cyan")
        schema.append("    │   ├── flux       ", style="green")
        schema.append("1D float, maggies\n", style="white")
        schema.append("    │   ├── flux_err   ", style="green")
        schema.append("1D float, positive uncertainties\n", style="white")
        schema.append("    │   ├── filters    ", style="green")
        schema.append("1D sedpy filter names, same length as flux\n", style="white")
        schema.append("    │   └── mask       ", style="yellow")
        schema.append("optional 1D bool or 0/1, True = valid\n", style="white")
        schema.append("    ├── Spectroscopy/\n", style="cyan")
        schema.append("    │   ├── wavelength ", style="green")
        schema.append("1D float, observed Angstrom\n", style="white")
        schema.append("    │   ├── flux       ", style="green")
        schema.append("1D float\n", style="white")
        schema.append("    │   ├── flux_err   ", style="green")
        schema.append("1D float, positive uncertainties\n", style="white")
        schema.append("    │   └── mask       ", style="yellow")
        schema.append("optional 1D bool or 0/1, True = valid\n", style="white")
        schema.append("    └── Metadata/\n", style="cyan")
        schema.append("        └── redshift   ", style="yellow")
        schema.append("optional scalar float; required by AmirModel if --redshift is absent", style="white")

        console.print(header)
        console.print(quick_start)
        console.print(
            Panel(
                schema,
                title="Expected HDF5 Data Schema",
                border_style="green",
                box=box.ROUNDED,
            )
        )
        console.print(inputs)
        console.print(data)
        console.print(model)
        console.print(fitting)
        console.print(
            Panel(
                notes,
                title="Rules Worth Remembering",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )

    def update_from_cli(self, argv=None):
        """
        Updates values by reading from the command line.
        Uses argparse.SUPPRESS to avoid overwriting defaults with 'None'.
        """
        cli_args = sys.argv[1:] if argv is None else list(argv)
        if any(arg in ("-h", "--help") for arg in cli_args):
            self.show_help()
            raise SystemExit(0)

        parser = argparse.ArgumentParser(
            description="Prospector SED fitting configuration",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        def help_with_default(dest, help_text):
            if help_text is argparse.SUPPRESS:
                return argparse.SUPPRESS
            value = getattr(self, dest)
            return f"{help_text} (default: {value})"

        def infer_dest(args, kwargs):
            if "dest" in kwargs:
                return kwargs["dest"]
            for arg in args:
                if arg.startswith("--"):
                    return arg[2:].replace("-", "_")
            return None

        # Helper function to avoid writing default=argparse.SUPPRESS everywhere
        def add_arg(group, *args, **kwargs):
            if "help" in kwargs:
                dest = infer_dest(args, kwargs)
                if dest and hasattr(self, dest):
                    kwargs["help"] = help_with_default(dest, kwargs["help"])
            kwargs["default"] = argparse.SUPPRESS
            group.add_argument(*args, **kwargs)

        # Helper function to handle booleans (explicit True/False)
        def add_bool(group, name, dest, help_text):
            # E.g.: --interactive sets the flag to True, --no-interactive sets it to False
            group.add_argument(
                f"--{name}",
                dest=dest,
                action="store_true",
                default=argparse.SUPPRESS,
                help=help_with_default(dest, help_text),
            )
            group.add_argument(
                f"--no-{name}",
                dest=dest,
                action="store_false",
                default=argparse.SUPPRESS,
                help=argparse.SUPPRESS,
            )

        io_group = parser.add_argument_group("Input and output")
        add_arg(io_group, "--file", type=str, help="Path to one HDF5 input file.")
        add_arg(io_group, "--file-list", "--file_list", dest="file_list", type=str, help="Text file with one HDF5 path per line.")
        add_arg(io_group, "--out", type=str, help="Output basename for a single target.")
        add_arg(io_group, "--out-folder", "--out_folder", dest="out_folder", type=str, help="Directory for Prospector result files.")
        add_arg(io_group, "--version", type=str, help="HDF5 version group to read.")
        add_arg(io_group, "--ext", type=str, help="Suffix appended to the output basename.")
        add_arg(io_group, "--galaxy-name", "--galaxy_name", "--name", dest="name", type=str, help="Display/output name for a single target.")

        data_group = parser.add_argument_group("Data selection and validation")
        add_bool(data_group, "photometry", "use_photometry", "Use photometric data.")
        add_bool(data_group, "spectroscopy", "use_spectroscopy", "Use spectroscopic data.")
        add_bool(data_group, "use-mask", "use_mask", "Apply masks stored in the HDF5 file.")
        data_group.add_argument("--use_mask", dest="use_mask", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        data_group.add_argument("--no-use_mask", dest="use_mask", action="store_false", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        add_bool(data_group, "filter-photo", "filter_photo", "Auto-mask invalid photometric points.")
        data_group.add_argument("--filter_photo", dest="filter_photo", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        data_group.add_argument("--no-filter_photo", dest="filter_photo", action="store_false", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        add_bool(data_group, "filter-spec", "filter_spec", "Auto-mask invalid spectral pixels.")
        data_group.add_argument("--filter_spec", dest="filter_spec", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        data_group.add_argument("--no-filter_spec", dest="filter_spec", action="store_false", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

        model_group = parser.add_argument_group("Model")
        add_arg(model_group, "--model", "--model-type", "--model_type", dest="model_type", type=str, help="Registered model name.")
        add_arg(model_group, "--redshift", type=float, help="Galaxy redshift; overrides HDF5 metadata.")
        add_arg(model_group, "--nbins", type=int, help="Number of Continuity SFH age bins.")
        add_arg(model_group, "--z-continuous", "--z_continuous", dest="z_continuous", type=int, help="FSPS zcontinuous setting.")
        add_bool(model_group, "fixed-z", "fixed_z", "Fix the redshift parameter.")
        model_group.add_argument("--fixed_z", dest="fixed_z", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        model_group.add_argument("--no-fixed_z", dest="fixed_z", action="store_false", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        add_bool(model_group, "sigmav", "add_sigmav", "Apply instrumental LSF smoothing when a dispersion file is available.")
        add_arg(model_group, "--dispersion-file", "--dispersion_file", dest="dispersion_file", type=str, help="JWST dispersion FITS file for LSF smoothing.")
        add_bool(model_group, "nebular", "add_nebular", "Include nebular emission.")
        add_bool(model_group, "duste", "add_duste", "Include dust emission.")
        add_bool(model_group, "dust1", "add_dust1", "Include birth-cloud dust.")
        add_bool(model_group, "agn", "add_agn", "Include AGN component.")

        engine_group = parser.add_argument_group("Fitting engine")
        add_bool(engine_group, "fit-outliers-photo", "fit_outliers_photo", "Fit photometric outlier fraction.")
        engine_group.add_argument("--fit_outliers_photo", dest="fit_outliers_photo", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        engine_group.add_argument("--no-fit_outliers_photo", dest="fit_outliers_photo", action="store_false", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        add_bool(engine_group, "fit-outliers-spec", "fit_outliers_spec", "Fit spectral outlier fraction.")
        engine_group.add_argument("--fit_outliers_spec", dest="fit_outliers_spec", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        engine_group.add_argument("--no-fit_outliers_spec", dest="fit_outliers_spec", action="store_false", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        add_bool(engine_group, "optimize", "optimize", "Run Prospector optimization.")
        add_bool(engine_group, "dynesty", "dynesty", "Run Dynesty nested sampling.")
        engine_group.add_argument("--emcee", dest="emcee", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        engine_group.add_argument("--no-emcee", dest="emcee", action="store_false", default=argparse.SUPPRESS, help=argparse.SUPPRESS)

        log_group = parser.add_argument_group("Logging and interactivity")
        add_bool(log_group, "interactive", "interactive", "Open the interactive spectral masking GUI.")
        add_bool(log_group, "verbose", "verbose", "Show rich data/model summary tables.")
        add_bool(log_group, "debug", "debug", "Enable debug-level logs.")
        add_bool(log_group, "log-to-file", "logging_to_file", "Write logs to per-target files instead of the terminal.")
        log_group.add_argument("--logging_file", dest="logging_to_file", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        log_group.add_argument("--no-logging_file", dest="logging_to_file", action="store_false", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        add_arg(log_group, "--log-folder", "--log_folder", dest="log_folder", type=str, help="Directory for log files.")

        args = parser.parse_args(cli_args)

        # Actual instance update
        updated_keys = []
        for key, value in vars(args).items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                updated_keys.append(f"{key}: {old_value} -> {value}")
            else:
                raise ValueError(f"CLI argument '{key}' does not exist.")

        self.__post_init__()
        self.validate(require_inputs=True)

        return updated_keys

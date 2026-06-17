import logging
import os
import sys
from copy import deepcopy

from config import FitConfig
from data_reader import GalaxyDataManager
from models import build_model, resolve_model_name, show_model
from sps import ProspectorSPSBuilder
from prospect.utils.obsutils import fix_obs
from prospect.models import PolySpecModel
from prospect.likelihood import NoiseModel
from prospect.likelihood.kernels import Uncorrelated
from prospect.fitting import fit_model, lnprobfn
from prospect.io import write_results as writer
from rich.logging import RichHandler
from utils import interactive_masking, plot_spectrum


# TODO non forzare V1 nel file h5 

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
)


try:
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    HAS_MPI = False

# logger = logging.getLogger("rich")
logger = logging.getLogger(__name__)

def setup_logging(config: FitConfig, rank: int, galaxy_name: str = "main"):
    """
    Configures logging. If called multiple times, resets previous logs
    so each galaxy gets a clean, independent log file.
    """
    # MODIFICA QUI: ottieni il VERO root logger omettendo il nome
    root_logger = logging.getLogger()

    # 1. CLEAR PREVIOUS HANDLERS
    # Ora questo rimuoverà correttamente il RichHandler del terminale
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(logging.DEBUG if getattr(config, "debug", False) else logging.INFO)

    # 2. IF LOGGING TO FILE
    if getattr(config, "logging_to_file", False):
        os.makedirs(config.log_folder, exist_ok=True)
        log_file_path = os.path.join(config.log_folder, f"{galaxy_name}.log")

        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setFormatter(
            logging.Formatter(f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s")
        )
        root_logger.addHandler(file_handler)

    # 3. OTHERWISE PRINT TO SCREEN
    else:
        # Se non loggi su file, reinserisci il RichHandler che hai appena cancellato
        root_logger.addHandler(
            RichHandler(rich_tracebacks=True, show_path=False, markup=True)
        )


def run_fitting_pipeline(config, rank=0, galaxy_name="test"):
    """
    Encapsulates the entire fitting process for a single galaxy.
    """
    try:
        config.validate(require_inputs=True)
        resolve_model_name(config.model_type)
        os.makedirs(config.out_folder, exist_ok=True)

        output_name = config.out or galaxy_name
        if config.ext is not None:
            output_name += config.ext

        output_path = os.path.join(config.out_folder, f"{output_name}.h5")

        logger.info("Run | target=%s", galaxy_name)
        logger.info("Run | input=%s", config.file)
        logger.info("Run | output=%s", os.path.abspath(output_path))

        # 1. Data Setup
        data = GalaxyDataManager(config)
        data.load_data()
        config.validate(require_inputs=True)

        # 2. Model Setup
        model = build_model(config)

        # Prevent 50 nodes from printing the same table simultaneously
        if getattr(config, "verbose", False): #and rank == 0:
            data.show()
            show_model(model.model_params)


        # 3. SPS Setup
        source = ProspectorSPSBuilder(config, data, model)
        sps = source.build_sps()

        # 4. Fit Configuration
        raw_obs = data.to_dict()
        if config.interactive:
            new_mask, lines = interactive_masking(config, raw_obs)
            raw_obs["mask"] = raw_obs["mask"] & new_mask
            logger.info(
                "Mask | spectroscopy %s/%s valid", raw_obs["mask"].sum(), len(new_mask)
            )
            # logger.info(raw_obs["maggies"])
            # logger.info(raw_obs["maggies_unc"])
            # logger.info(raw_obs["phot_mask"])
            ## TMP save new mask in data
            # f = h5py.File(f"tmp/{galaxy_name}.h5", "w")
            # with h5py.File(f"tmp/{galaxy_name}.h5", "w") as f:
            #     f.create_dataset("mask", data=raw_obs["mask"] & new_mask)
        
        if config.interactive:
            phot_mask = raw_obs["phot_mask"]
            if config.use_photometry and phot_mask is not None:
                phot_wavelengths = raw_obs["phot_wave"][phot_mask]
                phot_flux = raw_obs["maggies"][phot_mask]
                phot_flux_error = raw_obs["maggies_unc"][phot_mask]
            else:
                phot_wavelengths = None
                phot_flux = None
                phot_flux_error = None

            plot_spectrum(
                raw_obs["wavelength"],
                raw_obs["spectrum"],
                raw_obs["unc"],
                mask=raw_obs["mask"],
                phot_wavelengths=phot_wavelengths,
                phot_flux=phot_flux,
                phot_flux_error=phot_flux_error,
            )
            # wavelengths,
            # flux,
            # flux_error,
            # redshift=0,
            # mask=None,
        
        

        obs = fix_obs(raw_obs)
        mod = PolySpecModel(model.model_params)
        # mod = AGNSpecModel(model.model_params)
        # if getattr(config, "verbose", False):  # and rank == 0:
        #     plot_unicode_spectrum(raw_obs)

        if config.use_spectroscopy:
            jitter = Uncorrelated(parnames=["spec_jitter"])
            noise = NoiseModel(kernels=[jitter], metric_name="unc", weight_by=["unc"])
        else:
            noise = None

        # 5. Execute Fit
        output = fit_model(
            obs=obs,
            model=mod,
            sps=sps,
            lnprobfn=lnprobfn,
            optimize=config.optimize,
            noise = (noise, None),
            dynesty=config.dynesty,
            **config.dynesty_kwargs,
        )

        # 6. Save Results
        writer.write_hdf5(
            output_path,
            config.to_dict(),
            mod,
            obs,
            output["sampling"][0],
            None,
            sps=sps,
            tsample=output["sampling"][1],
            toptimize=0.0,
            model_params=model.model_params,
        )

        logger.info("Run | completed %s", config.name)
        return output

    except Exception as e:
        logger.error("Run | failed %s: %s", config.name, e, exc_info=True)
        raise


if __name__ == "__main__":
    # --- 1. GLOBAL MPI SETUP ---
    if HAS_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    # --- 2. BASE CONFIGURATION ---
    # Read the command line ONLY ONCE for all nodes
    base_config = FitConfig()
    try:
        modifiche_cli = base_config.update_from_cli()
    except Exception as exc:
        if rank == 0:
            logger.error("Config | %s", exc)
        sys.exit(2)

    # Create a temporary screen logger for boot messages
    setup_logging(base_config, rank, galaxy_name="boot")

    total_targets = len(base_config.targets)

    if total_targets == 0:
        if rank == 0:
            logger.error("No targets found. Provide --file or --file-list.")
        sys.exit(1)

    elif total_targets == 1 and size > 1:
        # CASE A: Single file, parallel engine
        if rank == 0:
            logger.info("Mode | single target across MPI ranks")

        # Extract the name of the single file unless the user supplied one.
        default_name = os.path.splitext(os.path.basename(base_config.targets[0]))[0]
        gal_name = base_config.name if base_config.name != "result" else default_name
        base_config.name = gal_name

        setup_logging(base_config, rank, galaxy_name=gal_name)
        run_fitting_pipeline(base_config, rank, galaxy_name=gal_name)

    else:
        # CASE B: Batch Mode
        # Distribution handled: if size > total_targets, excess nodes get an empty list.
        local_targets = base_config.targets[rank::size]

        if not local_targets:
            logger.info("Mode | no targets assigned to this rank")
            sys.exit(0)

        logger.info("Mode | assigned=%s total=%s", len(local_targets), total_targets)

        for target_path in local_targets:
            local_config = deepcopy(base_config)

            # Extract galaxy name from path before post_init.
            default_name = os.path.splitext(os.path.basename(target_path))[0]
            if total_targets == 1 and base_config.name != "result":
                galaxy_name = base_config.name
            else:
                galaxy_name = default_name

            # Update config fields
            local_config.file = target_path
            local_config.file_list = None
            local_config.name = galaxy_name
            local_config.__post_init__()

            # Configure the logger exclusively for this galaxy
            setup_logging(local_config, rank, galaxy_name=galaxy_name)

            run_fitting_pipeline(local_config, rank, galaxy_name=galaxy_name)

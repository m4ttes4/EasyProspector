import logging
import os
import sys
from copy import deepcopy

from config import FitConfig
from data_reader import GalaxyDataManager
from models import BaseModel, show_model
from sps import ProspectorSPSBuilder
from prospect.utils.obsutils import fix_obs
from prospect.models import PolySpecModel
from prospect.fitting import fit_model, lnprobfn
from prospect.io import write_results as writer
from rich.logging import RichHandler


# TODO non forzare V1 nel file h5 

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
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

    log_format = logging.Formatter(
        f"%(asctime)s - [Rank {rank}] - %(levelname)s - %(message)s"
    )
    root_logger.setLevel(logging.DEBUG if config.verbose else logging.INFO)

    # 2. IF LOGGING TO FILE
    if getattr(config, "logging_to_file", False):
        os.makedirs(config.log_folder, exist_ok=True)
        log_file_path = os.path.join(config.log_folder, f"{galaxy_name}.log")

        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)

    # 3. OTHERWISE PRINT TO SCREEN
    else:
        # Se non loggi su file, reinserisci il RichHandler che hai appena cancellato
        root_logger.addHandler(RichHandler(rich_tracebacks=True))


def run_fitting_pipeline(config, rank=0, galaxy_name="test"):
    """
    Encapsulates the entire fitting process for a single galaxy.
    """
    try:
        # Ensure output folder exists before trying to save
        os.makedirs(config.out_folder, exist_ok=True)
        output_path = os.path.join(config.out_folder, f"{galaxy_name}.h5")

        logger.info(f"Starting processing for file: {config.file}")
        logger.info(f"Save file path set to {os.path.abspath(output_path)}")

        # 1. Data Setup
        data = GalaxyDataManager(config)
        data.load_data()

        # 2. Model Setup
        model = BaseModel(config)

        # Prevent 50 nodes from printing the same table simultaneously
        if getattr(config, "verbose", False) and rank == 0:
            show_model(model.model_params)

        # 3. SPS Setup
        source = ProspectorSPSBuilder(config, data, model)
        sps = source.build_sps()

        # 4. Fit Configuration
        obs = fix_obs(data.to_dict())
        mod = PolySpecModel(model.model_params)

        # 5. Execute Fit
        output = fit_model(
            obs=obs,
            model=mod,
            sps=sps,
            lnprobfn=lnprobfn,
            optimize=config.optimize,
            dynesty=True,
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

        logger.info(f"Fit successfully completed for: {config.name}")
        return output

    except Exception as e:
        logger.error(f"Critical error on {config.name}: {e}", exc_info=True)
        return None


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
    modifiche_cli = base_config.update_from_cli()

    # Create a temporary screen logger for boot messages
    setup_logging(base_config, rank, galaxy_name="boot")

    total_targets = len(base_config.targets)

    if total_targets == 0:
        if rank == 0:
            logger.error("No targets found! Please provide --file or --file_list.")
        sys.exit(1)

    elif total_targets == 1 and size > 1:
        # CASE A: Single file, parallel engine
        if rank == 0:
            logger.info("Single Parallel Fit mode detected.")

        # Extract the name of the single file
        gal_name = os.path.splitext(os.path.basename(base_config.targets[0]))[0]
        base_config.name = gal_name

        setup_logging(base_config, rank, galaxy_name=gal_name)
        run_fitting_pipeline(base_config, rank, galaxy_name=gal_name)

    else:
        # CASE B: Batch Mode
        # Distribution handled: if size > total_targets, excess nodes get an empty list.
        local_targets = base_config.targets[rank::size]

        if not local_targets:
            logger.info(
                "No targets assigned to this node (N_MPI > N_Galaxies). Standing by."
            )
            sys.exit(0)

        logger.info(
            f"Assigned {len(local_targets)} galaxies out of {total_targets} total."
        )

        for target_path in local_targets:
            local_config = deepcopy(base_config)

            # Extract galaxy name from path before post_init
            galaxy_name = os.path.splitext(os.path.basename(target_path))[0]

            # Update config fields
            local_config.file = target_path
            local_config.name = galaxy_name
            local_config.__post_init__()

            # Configure the logger exclusively for this galaxy
            setup_logging(local_config, rank, galaxy_name=galaxy_name)

            run_fitting_pipeline(local_config, rank, galaxy_name=galaxy_name)



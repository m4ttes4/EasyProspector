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

try:
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    HAS_MPI = False

logger = logging.getLogger(__name__)


def setup_logging(config: FitConfig, rank: int, galaxy_name: str = "main"):
    """
    Configura il logging. Se chiamato più volte, resetta i log precedenti.
    In questo modo ogni galassia ha il suo file pulito e indipendente.
    """
    root_logger = logging.getLogger()

    # 1. SVUOTA GLI HANDLER PRECEDENTI!
    # Se il nodo 0 passa dalla galassia A alla galassia B, dobbiamo chiudere
    # il file della galassia A e aprire quello della galassia B.
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Formattiamo il log includendo il Rank, così a schermo capiamo chi parla
    log_format = logging.Formatter(
        f"%(asctime)s - [Rank {rank}] - %(levelname)s - %(message)s"
    )
    root_logger.setLevel(logging.DEBUG if config.verbose else logging.INFO)

    # 2. SE LOGGING SU FILE
    if getattr(config, "logging_to_file", False):
        os.makedirs(config.log_folder, exist_ok=True)
        # Usa il nome effettivo della galassia passato alla funzione
        log_file_path = os.path.join(config.log_folder, f"{galaxy_name}.log")

        # 'w' sovrascrive il file se esiste già (utile se rilanci lo script)
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)
    # 3. ALTRIMENTI SOLO A SCHERMO
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        root_logger.addHandler(console_handler)


def run_fitting_pipeline(config, rank=0):
    """
    Incapsula l'intero processo di fitting per una singola galassia.
    """
    try:
        logger.info(f"Inizio elaborazione file: {config.file}")

        data = GalaxyDataManager(config)
        data.load_data()

        model = BaseModel(config)

        if getattr(config, "verbose", False):
            show_model(model.model_params)

        source = ProspectorSPSBuilder(config, data, model)
        sps = source.build_sps()

        obs = fix_obs(data.to_dict())
        mod = PolySpecModel(model.model_params)

        output = fit_model(
            obs=obs,
            model=mod,
            sps=sps,
            lnprobfn=lnprobfn,
            optimize=config.optimize,
            dynesty=True,
            **config.dynesty_kwargs,
        )

        logger.info(f"Fit completato con successo per: {config.name}")
        return output

    except Exception as e:
        logger.error(f"Errore critico su {config.name}: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    if HAS_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    base_config = FitConfig()
    modifiche_cli = base_config.update_from_cli()

    # Creiamo un log temporaneo a schermo per i messaggi di boot
    setup_logging(base_config, rank, galaxy_name="boot")

    total_targets = len(base_config.targets)

    if total_targets == 0:
        if rank == 0:
            logger.error("Nessun target trovato! Fornisci --file o --file_list.")
        sys.exit(1)

    elif total_targets == 1 and size > 1:
        # CASO A: Singolo file, engine parallelo
        if rank == 0:
            logger.info("Modalità 'Singolo Fit Parallelo' rilevata.")

        # Estraiamo il nome del singolo file
        gal_name = os.path.splitext(os.path.basename(base_config.targets[0]))[0]
        base_config.name = gal_name

        setup_logging(base_config, rank, galaxy_name=gal_name)
        run_fitting_pipeline(base_config, rank)

    else:
        # CASO B: Batch Mode
        # TODO 1 RISOLTO: Distribuzione gestita. Se size > total_targets, i nodi in eccesso hanno lista vuota
        local_targets = base_config.targets[rank::size]

        if not local_targets:
            logger.info(
                "Nessun target assegnato a questo nodo (N_MPI > N_Galassie). Mi metto a riposo."
            )
            sys.exit(0)

        logger.info(
            f"Assegnate {len(local_targets)} galassie su {total_targets} totali."
        )

        for target_path in local_targets:
            local_config = deepcopy(base_config)

            # TODO 2 RISOLTO: Estraiamo il nome della galassia dal path prima di fare post_init
            galaxy_name = os.path.splitext(os.path.basename(target_path))[0]

            # Aggiorniamo i campi del config
            local_config.file = target_path
            local_config.name = galaxy_name
            local_config.__post_init__()

            # TODO 3 RISOLTO: Configuriamo il logger in modo esclusivo per questa galassia
            setup_logging(local_config, rank, galaxy_name=galaxy_name)

            run_fitting_pipeline(local_config, rank)

"""

eg
python run.py   --file = .h5 single file (no MPI)
                --param_file = model file?
                --file_list = a list .h5 files to run with MPI
                --root = root folder for full path for file or file list
                --interactive = if display interactive screen
                --version = root inside .h5 file that specify the version to use
                --out_dir = out directory where to save data
                --dispersion_file = LSF file
                --log_folder = folder where to put log files
                
                
file = a .h5 file con già i dati pronti
param_file =
"""
import sys
import os
import argparse
from typing import Dict, Optional, Any
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)
@dataclass
class FitConfig:
    # --- 1. Identificativi e Percorsi (I/O) ---
    

    file: Optional[str] = None  # Path completo al file .h5 di input
    out: Optional[str] = None #out name
    out_folder: Optional[str] = "results/out"  # Cartella dove salvare i risultati
    file_list: Optional[str] = None

    name: Optional[str] = "result"
    logging_to_file: bool = False  # Default: logging su terminale
    log_folder: str = "results/log"

    version: Optional[str] = "V1"  # Es: "F160W_selected"
    use_mask: bool = True #use the mask inside h5 file
    dispersion_file: Optional[str] = None
    param_file: str = field(default_factory=lambda: sys.argv[0])

    # --- 2. Selezione Dati ---
    use_photometry: bool = True
    use_spectroscopy: bool = True

    filter_photo: bool = True
    filter_spec: bool = True
    fit_outliers_photo: bool = False
    fit_outliers_spec: bool = False

    # --- 3. Parametri Fisici e Modello ---
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

    # --- 4. Configurazione del Fitting (Engine) ---
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

    # --- 5. Debug, Plotting e Interattività ---
    verbose: bool = True
    interactive: bool = False
    lines: Dict[str, tuple] = field(default_factory=dict)


    def __post_init__(self):
        """Validazioni e setup automatico dopo l'inizializzazione."""
        # Controlliamo che il valore esista prima di espandere la tilde
        self.targets = []
        
        if self.file:
            self.file = os.path.expanduser(self.file)
        
        if self.out_folder:
            self.out_folder = os.path.expanduser(self.out_folder)
        
        if self.dispersion_file:
            self.dispersion_file = os.path.expanduser(self.dispersion_file)
        
        if self.file_list:
            self.file_list = os.path.expanduser(self.file_list)
        
        if self.file_list and os.path.exists(self.file_list):
            # Se viene fornita una lista, legge il file e ignora le righe vuote
            with open(self.file_list, "r") as f:
                self.targets = [line.strip() for line in f if line.strip()]
                
        elif self.file:
            self.targets = [self.file]


    def to_dict(self) -> Dict[str, Any]:
        """Converte la configurazione in un dizionario (es. per Prospector)."""
        data = asdict(self)
        data.update(self.dynesty_kwargs)
        return data

    def update_from_cli(self):
        """
        Aggiorna i valori leggendo da linea di comando.
        Usa argparse.SUPPRESS per evitare di sovrascrivere i default con 'None'.
        """
        parser = argparse.ArgumentParser(description="Prospector Run Configuration")

        # Funzione helper per evitare di scrivere default=argparse.SUPPRESS ovunque
        def add_arg(*args, **kwargs):
            kwargs["default"] = argparse.SUPPRESS
            parser.add_argument(*args, **kwargs)

        # Funzione helper per gestire i booleani (True/False espliciti)
        def add_bool(name, dest):
            # Es: --interactive attiva il flag, --no-interactive lo disattiva
            parser.add_argument(
                f"--{name}", dest=dest, action="store_true", default=argparse.SUPPRESS
            )
            parser.add_argument(
                f"--no-{name}",
                dest=dest,
                action="store_false",
                default=argparse.SUPPRESS,
            )

        # 1. Stringhe e Numeri
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

        # Nuovi argomenti Booleani
        add_bool("use_mask", "use_mask")
        add_bool("filter_photo", "filter_photo")
        add_bool("filter_spec", "filter_spec")
        add_bool("fit_outliers_photo", "fit_outliers_photo")
        add_bool("fit_outliers_spec", "fit_outliers_spec")

        # 2. Booleani (Gestione Tristate)
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

        # Parsing: ignora i flag non registrati senza crashare
        args, unknown = parser.parse_known_args()

        # Aggiornamento effettivo dell'istanza
        if unknown:
            logger.warning(f"Argomenti CLI ignorati o sconosciuti: {unknown}")

        updated_keys = []
        for key, value in vars(args).items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                updated_keys.append(f"{key}: {old_value} -> {value}")
            else:
                # I warning possiamo stamparli subito, o aggiungerli a una lista di errori
                print(f"Warning: L'argomento CLI '{key}' non esiste.")

        self.__post_init__()

        # Restituiamo la lista di cosa è cambiato per loggarlo dopo!
        return updated_keys
            
            
if __name__ == "__main__":
    # 1. Inizializza la configurazione con i valori di default
    config = FitConfig()

    # 2. (Opzionale) Puoi forzare dei valori "hardcoded" specifici per questo script
    # prima di leggere da linea di comando, se necessario.
    # config.galaxy_name = "M31"

    # 3. Leggi e applica gli argomenti passati da terminale
    config.update_from_cli()

    # 4. Validazione minima prima di lanciare il codice pesante
    if not config.file:
        print("ERRORE: Devi fornire un file di input! Usa --file <path>")
        sys.exit(1)

    
    
    
    
"""
Documentazione: FitConfig

La classe FitConfig gestisce tutti i parametri necessari per eseguire il fitting (es. tramite Prospector). Supporta l'inizializzazione con valori di default robusti e l'override dinamico tramite riga di comando (CLI).
🚀 Come passare i parametri via CLI

Per i valori testuali e numerici, usa la sintassi standard:
--nome_parametro valore (es. --nbins 10)

Per i valori booleani (True/False), il sistema utilizza flag interruttori (tristate):

    Per forzare a True: usa --nome_flag (es. --interactive)

    Per forzare a False: usa --no-nome_flag (es. --no-interactive)

    Se omesso: viene mantenuto il valore di default definito nella classe.

📂 1. Identificativi e Percorsi (I/O)

Gestione dei file di input, output e logging.
Parametro Interno	Tipo	Default	Flag CLI	Descrizione
galaxy_name	str	None	--galaxy_name	Nome identificativo della galassia.
input_file	str	None	--input_file	Percorso completo al file .h5 di input (supporta ~).
output_folder	str	None	--output_folder	Cartella di destinazione per i risultati (supporta ~).
logging_to_file	bool	False	--logging_file / --no-logging_file	Se True, salva i log in un file oltre che a schermo.
log_folder	str	"results/log"	--log_folder	Cartella in cui salvare i file di log.
catalog_name	str	None	--catalog_name	Nome del catalogo fotometrico nell'HDF5 (es. F160W_selected).
use_file_mask	bool	False	--use_file_mask / --no-use_file_mask	Se True, utilizza la maschera salvata nel file H5.
dispersion_file	str	None	--dispersion_file	Percorso opzionale al file di dispersione (LSF).
param_file	str	sys.argv[0]	(Non esposto in CLI)	Traccia lo script in esecuzione per riproducibilità.

Parametri generati automaticamente (non modificabili direttamente):

    hfile_name: Generato come {galaxy_name}.h5.

    output_path: Percorso finale risultante dall'unione di output_folder e hfile_name.

🔭 2. Selezione Dati

Controllo su quali dati includere nel fit e come filtrarli.
Parametro Interno	Tipo	Default	Flag CLI	Descrizione
use_photometry	bool	True	--photometry / --no-photometry	Abilita/disabilita l'uso della fotometria nel fit.
use_spectroscopy	bool	True	--spectroscopy / --no-spectroscopy	Abilita/disabilita l'uso della spettroscopia.
filter_photo	bool	True	--filter_photo / --no-filter_photo	Applica filtri ai dati fotometrici.
filter_spec	bool	True	--filter_spec / --no-filter_spec	Applica filtri ai dati spettroscopici.
fit_outliers_photo	bool	False	--fit_outliers_photo (etc.)	Se True, include la modellazione degli outlier fotometrici.
fit_outliers_spec	bool	False	--fit_outliers_spec (etc.)	Se True, include la modellazione degli outlier spettroscopici.
⚛️ 3. Parametri Fisici e Modello

Definizione delle componenti fisiche del modello stellare/galattico.
Parametro Interno	Tipo	Default	Flag CLI	Descrizione
model_type	str	"ContinuitySFH"	--model_type	Tipo di modello (es. ContinuitySFH, ParametricSFH).
redshift	float	None	--redshift	Redshift della galassia (se None, letto dai metadati).
fixed_z	bool	False	--fixed_z / --no-fixed_z	Se True, il redshift è bloccato (non è un parametro libero).
nbins	int	8	--nbins	Numero di bin temporali per la SFH (Star Formation History).
z_continuous	int	1	--z_continuous	Parametro per FSPS (gestione interpolazione metallicità).
add_nebular	bool	True	--nebular / --no-nebular	Include emissione nebulare nel modello.
add_duste	bool	True	--duste / --no-duste	Include emissione della polvere (Dust emission).
add_dust1	bool	True	--dust1 / --no-dust1	Include attenuazione della nube natale (Birth cloud dust).
add_agn	bool	False	--agn / --no-agn	Include la componente AGN.
add_sigmav	bool	True	--sigmav / --no-sigmav	Applica smoothing spettrale (resolution matching).
⚙️ 4. Configurazione del Fitting (Engine)

Impostazioni per l'algoritmo di campionamento/ottimizzazione.
Parametro Interno	Tipo	Default	Flag CLI	Descrizione
optimize	bool	False	--optimize / --no-optimize	Abilita ottimizzazione Levenberg-Marquardt.
emcee	bool	False	--emcee / --no-emcee	Abilita il campionamento MCMC.
dynesty	bool	True	--dynesty / --no-dynesty	Abilita Dynamic Nested Sampling (consigliato).
dynesty_kwargs	dict	Vedi sotto	(Non esposto in CLI)	Parametri avanzati per l'engine Dynesty.

Default per dynesty_kwargs:
JSON

{
  "nested_nlive_init": 300,
  "nested_sample": "rwalk",
  "nested_target_n_effective": 300,
  "nested_dlogz_init": 0.01
}

📊 5. Debug, Plotting e Interattività

Comportamento a runtime dello script.
Parametro Interno	Tipo	Default	Flag CLI	Descrizione
verbose	bool	True	--verbose / --no-verbose	Stampa output dettagliati a schermo.
interactive	bool	False	--interactive / --no-interactive	Mostra plot e richiede input manuali all'utente.
lines	dict	{}	(Non esposto in CLI)	Linee spettrali per mascheramento o plotting.
Esempi Pratici di Utilizzo

Esecuzione Standard (usa i default, eccetto per i path richiesti):
Bash

python run.py --galaxy_name NGC1234 --input_file ./data/NGC1234.h5 --output_folder ./results

Esecuzione Veloce per Testing (solo spettroscopia, no dust, no verbose):
Bash

python run.py --galaxy_name NGC1234 --input_file ./data.h5 --output_folder ./out \
              --no-photometry --no-duste --no-dust1 --no-verbose

Modifica dei parametri fisici:
Bash

python run.py --galaxy_name NGC1234 --input_file ./data.h5 --output_folder ./out \
              --model_type ParametricSFH --nbins 4 --fixed_z --redshift 0.05
"""
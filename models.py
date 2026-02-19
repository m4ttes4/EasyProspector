from typing import Dict, Any
import numpy as np
from prospect.models import priors, transforms
from prospect.models.templates import TemplateLibrary, adjust_continuity_agebins

import logging
import numpy as np
from typing import Dict, Any

import rich.box
from rich.table import Table
from rich.console import Console

# Inizializziamo il logger per questo modulo
logger = logging.getLogger(__name__)


def format_dict(d: Dict) -> str:
    lines = []
    for key, value in d.items():
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        lines.append(f"{key} --> {formatted_value}")
    return "\n".join(lines)


def format_long_list(value: Any, max_items_per_line: int = 3) -> str:
    """Formatta liste lunghe su più righe."""
    if isinstance(value, (list, np.ndarray)) and len(value) > max_items_per_line:
        # Assicuriamoci che sia una lista per fare lo slicing in modo sicuro
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
    Tabula i parametri di un modello di Prospector usando Rich.
    Stampa la tabella in modo pulito e sicuro usando il modulo logging globale.
    """
    # 1. Creazione della tabella (la logica rimane identica alla tua)
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

        # Rich interpreterà questi tag se la console ha i colori attivi,
        # altrimenti li rimuoverà in automatico se stampiamo su file.
        is_free_display = "[bold green]True[/]" if is_free else "[bold red]False[/]"

        if not isinstance(prior, (str, type(None))):
            # Aggiunto controllo sicuro per evitare crash se l'oggetto prior non ha .params
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

    # 2. Cattura dell'output e Logging (LA MAGIA AVVIENE QUI)

    # Creiamo una console "virtuale".
    # Usare color_system=None è FONDAMENTALE quando si logga su file,
    # altrimenti il file .log si riempie di codici illeggibili come "\033[92mTrue\033[0m"
    console = Console(width=120, color_system=None)

    with console.capture() as capture:
        console.print(table)

    table_str = capture.get()

    # Passiamo la stringa gigante al nostro logger.
    # Mettiamo "\n" all'inizio così la tabella parte su una riga pulita sotto il timestamp del log.
    logger.info("\n" + table_str)
    
class ProspectorModelBuilder:
    """
    Classe base per costruire i parametri del modello Prospector.
    Include funzionalità di visualizzazione con Rich.
    """

    def __init__(self):
        self.model_params = {}

    def get_params(self) -> Dict[str, Any]:
        return self.model_params

    # --- Metodi di Visualizzazione (Rich) ---

    @staticmethod
    def _format_dict(d: Dict) -> str:
        """Helper statico: Formatta dizionari (es. parametri prior)."""
        lines = []
        for key, value in d.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            lines.append(f"{key} -> {formatted_value}")
        return "\n".join(lines)

    @staticmethod
    def _format_long_list(value: Any, max_items_per_line: int = 3) -> str:
        """Helper statico: Formatta liste o array lunghi su più righe."""
        if isinstance(value, (list, np.ndarray)) and len(value) > max_items_per_line:
            # Converte in lista se è numpy
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

    def show_summary(self):
        pass
        # tabula_modello_prospector_rich(self.model_params)


class BaseModel(ProspectorModelBuilder):
    """
    Costruisce un modello con storia di formazione stellare non parametrica (Continuity SFH).
    Gestisce configurazioni complesse come nebular emission, dust, smoothing e outlier.
    """

    def __init__(self, config):
        
        super().__init__()
        self.config = config


        # 1. Base SFH (Continuity)
        self._setup_sfh()

        # 2. Parametri Fisici (Metallicity, IMF, etc.)
        self._setup_physical_params()

        # 3. Polvere
        self._setup_dust()

        # 4. Nebular Emission (Linee)
        if self.config.add_nebular:
            self._setup_nebular()

        # 5. Calibrazione Spettroscopica (Smoothing, Normalizzazione)
        if self.config.use_spectroscopy:
            self._setup_spectroscopy()

        # 6. Modelli di Outlier (Noise modeling)
        self._setup_outliers()

        # 7. Marginalizzazione Linee (Avanzato)
        if getattr(self.config, "margin_elines", False):
            self._setup_line_marginalization()

    def _setup_sfh(self):
        """Imposta i parametri per la Continuity SFH (bin di massa e tempo)."""
        # Carica il template base
        self.model_params.update(TemplateLibrary["continuity_sfh"])

        # Redshift (fondamentale per calcolare l'età dell'universo)
        z = self.config.redshift if self.config.redshift is not None else 0.0

        # Regola i bin temporali in base all'età dell'universo a z
        # Nota: tuniv=13.7 è un'approssimazione sicura, ma usare cosmo.age(z) è meglio se disponibile
        self.model_params.update(
            adjust_continuity_agebins(
                self.model_params, tuniv=13.7, nbins=self.config.nbins
            )
        )

        # Massa Totale
        self.model_params["logmass"] = {
            "N": 1,
            "isfree": True,
            "init": 10.5,
            "units": "Solar masses formed",
            "prior": priors.TopHat(
                mini=6.0, maxi=13.0
            ),  # Ho allargato un po' il mini per sicurezza
        }

        # Masse per Bin (Calcolate da logmass + ratios)
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
                # Prior stretto attorno al valore fotometrico/spettroscopico iniziale
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
        """Metallicità e IMF."""
        # Metallicità Stellare
        self.model_params["logzsol"] = {
            "N": 1,
            "isfree": True,
            "init": -0.3,
            "units": r"$\log (Z/Z_\odot)$",
            "prior": priors.TopHat(mini=-2.0, maxi=0.5),  # Allargato leggermente maxi
        }

        # IMF (Chabrier = 1)
        self.model_params["imf_type"] = {
            "N": 1,
            "isfree": False,
            "init": 1,
            "units": "FSPS index",
        }

    def _setup_dust(self):
        """Configurazione attenuazione e emissione polvere."""
        # Attenuazione Continuum (Dust2)
        self.model_params["dust_type"] = {
            "N": 1,
            "isfree": False,
            "init": 4,
            "units": "FSPS index",
        }  # Calzetti+

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

        # Emissione Polvere (IR)
        if self.config.add_duste:
            self.model_params.update(TemplateLibrary["dust_emission"])
            # Sovrascrivi priors se necessario
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
        """Emissioni Nebulari (HII regions)."""
        self.model_params.update(TemplateLibrary["nebular"])

        # Default: non includere linee nello spettro stellare (le gestiamo noi o FSPS)
        self.model_params["nebemlineinspec"] = {"N": 1, "isfree": False, "init": False}

        # Fisica del gas
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
        # Dispersione velocità linee intrinseche
        self.model_params["eline_sigma"] = {
            "N": 1,
            "isfree": True,
            "init": 150.0,
            "units": "km/s",
            "prior": priors.TopHat(
                mini=50, maxi=500
            ),  # 3000 era forse troppo per linee nebular
        }

    def _setup_spectroscopy(self):
        """Parametri strumentali: Smoothing e Polinomi di calibrazione."""

        # 1. Smoothing Spettrale (Sigma Smooth)
        self.model_params.update(TemplateLibrary["spectral_smoothing"])
        self.model_params["sigma_smooth"] = {
            "N": 1,
            "isfree": True,
            "init": 1000.0,
            "units": "km/s",
            "prior": priors.TopHat(mini=200.0, maxi=2000.0),
        }

        # 2. Ottimizzazione Continuum (Polinomi)
        self.model_params.update(TemplateLibrary["optimize_speccal"])

        # Normalizzazione Spettro (Scaling factor)
        self.model_params["spec_norm"] = {
            "N": 1,
            "isfree": True,
            "init": 1.0,
            "prior": priors.Normal(
                mean=1.0, sigma=0.2
            ),  # Sigma 0.1 un po' stretto a volte
        }

        # Jitter (Noise floor underestimation)
        self.model_params["spec_jitter"] = {
            "N": 1,
            "isfree": True,
            "init": 1.0,  # Init 1.0 = add 1*sigma error?
            "prior": priors.TopHat(mini=0.0, maxi=5.0),  # jitter >= 0 di solito
        }
        # Polinomio
        self.model_params["polyorder"] = {
            "N": 1,
            "isfree": False,
            "init": 10,
        }  # 10 ordine alto, occhio all'overfitting

    def _setup_outliers(self):
        """Modelli di mixture per gestire dati 'cattivi' (outliers)."""

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
                "init": 0.00,  # Inizia a 0
                "prior": priors.TopHat(mini=0, maxi=0.1),  # Max 10% fotometria outlier
            }
            self.model_params["nsigma_outlier_phot"] = {
                "N": 1,
                "isfree": False,
                "init": 50.0,
            }

    def _setup_line_marginalization(self):
        """
        Logica complessa per marginalizzare le linee di emissione.
        Richiede 'lines' nel config e dati osservati validi.
        """
        if not self.config.add_nebular:
            raise ValueError("Marginalization requires add_nebular=True")

        # Recupera le linee da fittare dalla config
        # Assumiamo che config.lines sia un dict: {'[OIII]': 5007, ...}
        lines_dict = getattr(self.config, "lines", {})
        if not lines_dict:
            print("Warning: margin_elines=True but no lines found in config.")
            return

        to_fit_names = list(lines_dict.keys())

        # --- Qui potresti inserire la logica di controllo pixel (da tuo codice) ---
        # Per ora prendiamo tutto quello che c'è nella config
        valid_lines = to_fit_names

        if not valid_lines:
            return

        # Carica template marginalization
        self.model_params.update(TemplateLibrary["nebular_marginalization"])

        self.model_params["elines_to_fit"] = {
            "N": 1,
            "isfree": False,
            "init": np.array(valid_lines),
        }

        # Ampiezza prior (sigma velocità per queste linee)
        self.model_params["eline_sigma"] = {
            "N": 1,
            "isfree": True,
            "init": 300.0,
            "prior": priors.TopHat(mini=50.0, maxi=1000.0),
        }

        # Fit Redshift specifico per le linee? (Offset rispetto alle stelle)
        # Nota: nel tuo config originale era 'fit_eline_redshift'
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
                "init": np.zeros(n_lines),  # Delta z inizialmente 0
                "prior": priors.TopHat(
                    mini=-0.01, maxi=0.01
                ),  # Piccolo shift consentito
            }
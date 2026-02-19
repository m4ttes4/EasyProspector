

import numpy as np
import logging
import os
from astropy.io import fits
from scipy import interpolate

try:
    from prospect.sources import FastStepBasis, CSPSpecBasis
except ImportError:
    FastStepBasis, CSPSpecBasis = None, None

logger = logging.getLogger(__name__)


class ProspectorSPSBuilder:
    """
    Gestisce l'inizializzazione dell'oggetto SPS (Stellar Population Synthesis).
    Applica lo smoothing strumentale (LSF) leggendo i dati direttamente dal file
    indicato in configurazione, forzando l'uso della libreria MILES.
    """

    def __init__(self, config, data_handler, model):
        self.config = config
        self.data_handler = data_handler
        self.model = model

    def build_sps(self):
        """Costruisce e restituisce l'oggetto SPS configurato."""
        if FastStepBasis is None:
            logger.error("Librerie prospect.models.sedmodel mancanti.")
            raise ImportError("Prospector non è installato correttamente.")

        # Estrae il parametro per l'interpolazione della metallicità
        zcont = getattr(self.config, "z_continuous", 1)

        # 1. Selezione del tipo SPS in base ai parametri liberi del modello
        if "agebins" in self.model.model_params:
            sps = FastStepBasis(zcontinuous=zcont)
            logger.info("Costruito SPS: FastStepBasis (Non-parametric SFH).")
            
        elif "tau" in self.model.model_params:
            sps = CSPSpecBasis(zcontinuous=zcont)
            logger.info("Costruito SPS: CSPSpecBasis (Parametric SFH).")
            
        else:
            logger.error("Il modello Prospector non contiene né 'agebins' né 'tau'.")
            raise ValueError("Model parameters must include either 'agebins' or 'tau'.")

        # 2. Applicazione dello smoothing strumentale se richiesto
        if self.config.add_sigmav and ("agebins" in self.model.model_params):
            if self.config.dispersion_file is not None:
                self._apply_smoothing(sps)
            else:
                logger.info(
                    "Smoothing (add_sigmav) richiesto, ma dispersion_file è None. LSF ignorato."
                )

        return sps
    
    def _read_jwst_dispersion(self, path):
        """Legge il file FITS di JWST e restituisce una funzione di interpolazione."""
        with fits.open(path) as hdul:
            nirspec_wave = hdul[1].data["WAVELENGTH"]  # Solitamente in micron!
            nirspec_R = hdul[1].data["R"]


            func_nirspec = interpolate.interp1d(
                nirspec_wave,
                2.998e5 / (nirspec_R * 2.355),
                bounds_error=False,
                fill_value="extrapolate",
            )
            return func_nirspec



    def _apply_smoothing(self, sps):
        """Legge il file di dispersione e inietta l'LSF nell'SPS."""
        disp_file = self.config.dispersion_file

        if not os.path.exists(disp_file):
            logger.error(f"Il file di dispersione non esiste: {disp_file}")
            raise FileNotFoundError(f"Dispersion file non trovato: {disp_file}")

        logger.debug(f"Lettura file di dispersione LSF: {disp_file}")
        try:
            # 1. Recuperiamo le lunghezze d'onda osservate (in Angstrom)
            wave_obs = self.data_handler.spectroscopy["wavelength"]

            # 2. Otteniamo la FUNZIONE di interpolazione
            interp_func = self._read_jwst_dispersion(disp_file)

            # 3. VALUTIAMO la funzione per ottenere l'array sigma_v
            # IMPORTANTE: Dividiamo per 10000 per convertire gli Angstrom in Micron
            # in modo che combacino con le lunghezze d'onda del file FITS di JWST.
            # (Rimuovi il / 10000.0 se i tuoi wave_obs sono già in micron!)
            sigma_v = interp_func(wave_obs / 10000.0)

        except Exception as e:
            logger.error(f"Errore durante la creazione o valutazione dell'LSF: {e}")
            raise ValueError(
                f"Impossibile applicare il file di dispersione {disp_file}. Dettagli: {e}"
            ) from e

        logger.debug("Calcolo LSF forzando la libreria MILES.")
        wave_lsf, delta_v = self._get_lsf(wave_obs, sigma_v)

        # Abilita lo smoothing in FSPS
        sps.ssp.params["smooth_lsf"] = True

        # Inietta la matrice di smoothing
        sps.ssp.set_lsf(wave_lsf, delta_v)
        logger.info(
            f"Smoothing strumentale (LSF) applicato con successo dal file {os.path.basename(disp_file)}."
        )

    def _get_lsf(self, wave_obs, sigma_v):
        """
        Calcola la differenza in quadratura tra dispersione strumentale e libreria MILES.
        Restituisce lunghezza d'onda rest-frame e delta_v in km/s.
        """
        zred = self.config.redshift if self.config.redshift is not None else 0.0
        lightspeed = 2.998e5  # km/s

        # 1. Filtriamo subito i valori dove la dispersione strumentale è zero o negativa
        valid_dispersion = sigma_v > 0
        wave_obs_clean = wave_obs[valid_dispersion]
        sigma_v_clean = sigma_v[valid_dispersion]

        # Passiamo al rest-frame
        wave_rest = wave_obs_clean / (1.0 + zred)

        # 2. Otteniamo la risoluzione della libreria teorica MILES (rest-frame)
        miles_fwhm_aa = 2.54
        # formula: c * FWHM / (2.355 * lambda)
        sigma_v_lib = lightspeed * miles_fwhm_aa / (2.355 * wave_rest)

        # Dominio di validità MILES
        valid_rest_wave = (wave_rest > 3525.0) & (wave_rest < 7500.0)

        # 3. Calcolo della differenza in quadratura
        # clip a 0 previene sqrt(numeri negativi) se la libreria ha risoluzione peggiore dello strumento
        dsv = np.sqrt(np.clip(sigma_v_clean**2 - sigma_v_lib**2, 0, np.inf))

        # Restituiamo solo le porzioni di array che rientrano nei limiti della libreria
        return wave_rest[valid_rest_wave], dsv[valid_rest_wave]
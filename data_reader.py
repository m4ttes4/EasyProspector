import h5py
import os
import logging
from typing import Optional, Dict, Any
import numpy as np

try:
    from sedpy import observate
except ImportError:
    observate = None

# Inizializza il logger per questo modulo
logger = logging.getLogger(__name__)


class GalaxyDataManager:
    def __init__(self, config):
        self.config = config

        if not self.config.file:
            raise ValueError(
                "Errore: 'file' non è definito nella configurazione."
            )

        self.filepath = self.config.file
        self.version = getattr(self.config, "version", None) or "V1"

        self.photometry: Optional[Dict[str, Any]] = None
        self.spectroscopy: Optional[Dict[str, Any]] = None
        self.metadata: Optional[Dict[str, Any]] = None

    def load_data(self):
        """Carica i dati e avvia la validazione."""
        if not os.path.exists(self.filepath):
            logger.error(f"File non trovato: {self.filepath}")
            raise FileNotFoundError(f"Il file dati non esiste: {self.filepath}")

        logger.info(f"Apertura del file HDF5: {self.filepath}")

        with h5py.File(self.filepath, "r") as h5_file:
            if self.version not in h5_file:
                logger.error(f"Versione '{self.version}' mancante nel file.")
                raise KeyError(
                    f"Versione '{self.version}' non trovata in {self.filepath}."
                )

            logger.debug(f"Accesso al gruppo della versione: {self.version}")
            version_group = h5_file[self.version]

            # --- Lettura Fotometria ---
            if self.config.use_photometry:
                if "Photometry" in version_group:
                    logger.info("Estrazione dati fotometrici in corso...")
                    self.photometry = self._extract_to_memory(
                        version_group["Photometry"]
                    )
                    self._validate_photometry()
                    self._build_photometric_filters()
                else:
                    logger.warning(
                        "Fotometria richiesta dal config, ma non presente nel file HDF5."
                    )

            # --- Lettura Spettroscopia ---
            if self.config.use_spectroscopy:
                if "Spectroscopy" in version_group:
                    logger.info("Estrazione dati spettroscopici in corso...")
                    self.spectroscopy = self._extract_to_memory(
                        version_group["Spectroscopy"]
                    )
                    self._validate_spectroscopy()
                else:
                    logger.warning(
                        "Spettroscopia richiesta dal config, ma non presente nel file HDF5."
                    )

            # --- Lettura Metadati ---
            if "Metadata" in version_group:
                logger.info("Estrazione metadati in corso...")
                self.metadata = self._extract_to_memory(version_group["Metadata"])
                logger.debug(f"Metadati trovati: {list(self.metadata.keys())}")

                self._update_config_from_metadata()
            else:
                logger.debug("Nessun gruppo 'Metadata' trovato nel file.")

        logger.info("Caricamento e validazione dati completati con successo.")

    def _extract_to_memory(self, h5_object) -> Dict[str, Any]:
        """Converte ricorsivamente i dataset HDF5 in dict di array numpy in RAM."""
        data_dict = {}
        if isinstance(h5_object, h5py.Group):
            for key, item in h5_object.items():
                if isinstance(item, h5py.Dataset):
                    data_dict[key] = item[()]
                else:
                    data_dict[key] = self._extract_to_memory(item)
        elif isinstance(h5_object, h5py.Dataset):
            data_dict["data"] = h5_object[()]
        return data_dict

    def _validate_photometry(self):
        """Controlla la consistenza della fotometria e applica filtri opzionali."""
        if not self.photometry:
            return

        logger.debug("Avvio validazione fotometria e generazione maschera...")
        req_keys = ["flux", "flux_err", "filters"]
        for k in req_keys:
            if k not in self.photometry:
                logger.error(f"Fotometria corrotta: chiave '{k}' mancante.")
                raise ValueError(f"Dati fotometrici corrotti: manca la chiave '{k}'.")

        flux = self.photometry["flux"]
        flux_err = self.photometry["flux_err"]
        n_elements = len(flux)

        if self.config.use_mask and "mask" in self.photometry:
            logger.debug("Utilizzo della maschera fotometrica dal file HDF5.")
            mask = np.array(self.photometry["mask"], dtype=bool)
            if len(mask) != n_elements:
                raise ValueError(
                    f"Lunghezza mask fotometrica ({len(mask)}) diversa dai dati ({n_elements})."
                )
        else:
            logger.debug("Creazione maschera fotometrica di default.")
            mask = np.ones(n_elements, dtype=bool)

        if self.config.filter_photo:
            logger.debug("Applicazione filtri autonomi per NaN e Inf sulla fotometria.")
            mask &= np.isfinite(flux)
            mask &= np.isfinite(flux_err)
            mask &= flux_err > 0

        self.photometry["mask"] = mask
        logger.debug(
            f"Maschera fotometrica creata: {mask.sum()}/{n_elements} pixel validi."
        )

    def _validate_spectroscopy(self):
        """Controlla la consistenza della spettroscopia e applica filtri opzionali."""
        if not self.spectroscopy:
            return

        logger.debug("Avvio validazione spettroscopia e generazione maschera...")
        req_keys = ["wavelength", "flux", "flux_err"]
        for k in req_keys:
            if k not in self.spectroscopy:
                logger.error(f"Spettroscopia corrotta: chiave '{k}' mancante.")
                raise ValueError(
                    f"Dati spettroscopici corrotti: manca la chiave '{k}'."
                )

        wave = self.spectroscopy["wavelength"]
        flux = self.spectroscopy["flux"]
        flux_err = self.spectroscopy["flux_err"]
        n_elements = len(flux)

        if self.config.use_mask and "mask" in self.spectroscopy:
            logger.debug("Utilizzo della maschera spettroscopica dal file HDF5.")
            mask = np.array(self.spectroscopy["mask"], dtype=bool)
            if len(mask) != n_elements:
                raise ValueError(
                    f"Lunghezza mask spettroscopica ({len(mask)}) diversa dai dati ({n_elements})."
                )
        else:
            logger.debug("Creazione maschera spettroscopica di default.")
            mask = np.ones(n_elements, dtype=bool)

        if self.config.filter_spec:
            logger.debug(
                "Applicazione filtri autonomi per NaN e Inf sulla spettroscopia."
            )
            mask &= np.isfinite(flux)
            mask &= np.isfinite(flux_err)
            mask &= flux_err > 0
            mask &= np.isfinite(wave)
            mask &= wave > 0

        self.spectroscopy["mask"] = mask
        logger.debug(
            f"Maschera spettroscopica creata: {mask.sum()}/{n_elements} pixel validi."
        )

    def _build_photometric_filters(self):
        """Converte i nomi dei filtri dal file H5 in oggetti sedpy.observate.Filter."""
        if not self.photometry:
            return

        if observate is None:
            logger.error("La libreria 'sedpy' non è installata ma è richiesta.")
            raise ImportError(
                "La libreria 'sedpy' non è installata, ma è richiesta per la fotometria."
            )

        raw_filters = self.photometry["filters"]
        logger.debug(f"Costruzione oggetti sedpy per {len(raw_filters)} filtri...")

        filter_names = []
        for f in raw_filters:
            if isinstance(f, bytes):
                filter_names.append(f.decode("utf-8"))
            else:
                filter_names.append(str(f))

        try:
            filters_list = [observate.Filter(f) for f in filter_names]
        except Exception as e:
            logger.error(f"Errore di sedpy durante il caricamento dei filtri: {e}")
            raise ValueError(
                f"Errore in sedpy: impossibile trovare uno o più filtri. Dettagli: {e}"
            )

        filters_wave = np.array([flt.wave_effective for flt in filters_list])

        self.photometry["sedpy_filters"] = filters_list
        self.photometry["wave_effective"] = filters_wave
        logger.debug("Filtri sedpy costruiti correttamente.")
    
    def _update_config_from_metadata(self):
        """
        Aggiorna l'oggetto FitConfig usando i metadati letti dal file HDF5,
        rispettando le priorità della riga di comando.
        """
        if not self.metadata:
            return

        if "redshift" in self.metadata:
            # Il nostro metodo _extract_to_memory salva i dataset singoli dentro la chiave "data"
            z_file = self.metadata["redshift"]
            if isinstance(z_file, dict) and "data" in z_file:
                z_file = z_file["data"]

            # Assicuriamoci che sia un float (HDF5 a volte restituisce array di 1 elemento o float32)
            try:
                z_file = float(z_file)
            except (ValueError, TypeError):
                logger.warning(f"Valore di redshift nel file non valido: {z_file}")
                return

            # Applica il redshift SOLO se l'utente non lo ha forzato da terminale (cioè se è None)
            if self.config.redshift is None:
                self.config.redshift = z_file
                logger.info(
                    f"Redshift aggiornato automaticamente dai metadati: z = {self.config.redshift:.4f}"
                )
            else:
                logger.info(
                    f"Redshift CLI (z={self.config.redshift}) mantenuto. Ignorato z={z_file:.4f} del file."
                )
    
    def to_dict(self):
        observation = {
            "wavelength": self.spectroscopy["wavelength"],
            "spectrum": self.spectroscopy["flux"],
            "unc": self.spectroscopy["flux_err"],
            "mask": self.spectroscopy["mask"],
            "filters": self.photometry["sedpy_filters"],
            "maggies": self.photometry["flux"],
            "maggies_unc": self.photometry["flux_err"],
            "phot_mask": self.photometry["mask"],
            "phot_wave": self.photometry["wave_effective"],
        }
        return observation

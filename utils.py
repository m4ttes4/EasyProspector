import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button, CheckButtons
import logging

import plotext as plttxt

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


def plot_emission_lines(ax, wavelengths, flux, lines_dict):
    """
    Aggiunge le linee di emissione al grafico.
    Trova il picco locale dei dati attorno alla riga e posiziona il testo
    appena sopra, collegato da una linea tratteggiata.
    """
    # Filtra i dati validi per calcolare il range globale del grafico
    valid_flux = flux[np.isfinite(flux)]
    if len(valid_flux) == 0:
        return
        
    f_min, f_max = np.nanmin(valid_flux), np.nanmax(valid_flux)
    f_range = f_max - f_min
    
    # Imposta un offset dinamico per il testo (es. 5% del range totale)
    text_offset = f_range * 0.05
    
    w_min, w_max = np.nanmin(wavelengths), np.nanmax(wavelengths)
    
    for name, ww in lines_dict.items():
        # Estrae la lunghezza d'onda (gestisce sia la tupla che il singolo float)
        wavelength = ww[0] if isinstance(ww, (tuple, list)) else ww
        
        if w_min <= wavelength <= w_max:
            # Trova l'indice corrispondente alla lunghezza d'onda
            idx = np.nanargmin(np.abs(wavelengths - wavelength))
            
            # Crea una piccola finestra di pixel (es. +/- 15 pixel) per trovare il picco locale reale
            window_size = 15
            idx_start = max(0, idx - window_size)
            idx_end = min(len(flux), idx + window_size)
            
            local_flux = flux[idx_start:idx_end]
            
            # Calcola il picco locale (ignorando eventuali NaN)
            if len(local_flux) > 0 and not np.all(np.isnan(local_flux)):
                local_peak = np.nanmax(local_flux)
            else:
                local_peak = f_max  # Fallback di sicurezza
            
            # Disegna una linea verticale leggera dal fondo del grafico fino al picco locale
            ax.vlines(
                wavelength,
                ymin=f_min,
                ymax=local_peak + (text_offset * 0.5),
                color="indianred",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                zorder=1
            )
            
            # Posiziona il testo appena sopra il picco locale
            ax.text(
                wavelength,
                local_peak + text_offset,  
                name,
                color="darkred",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="medium",
                rotation=90,
                zorder=2
            )


def interactive_masking(config, data_dict):
    """
    Funzione per mascherare interattivamente lo spettro di una galassia.
    
    Parametri
    ---------
    config : FitConfig
        Oggetto di configurazione da cui estrarre redshift e dizionario delle linee.
    data_dict : dict
        Dizionario generato da GalaxyDataManager.to_dict() contenente 'wavelength', 
        'spectrum', e opzionalmente 'mask'.
        
    Ritorna
    -------
    mask : np.ndarray
        La nuova maschera booleana aggiornata.
    lines_to_fit : dict
        Il dizionario delle righe di emissione selezionate per la marginalizzazione.
    """
    # 1. Estrazione dati da config e data_dict
    z = config.redshift if config.redshift is not None else 0.0
    line_dict = config.lines if hasattr(config, "lines") and config.lines else {}
    
    wavelengths = data_dict["wavelength"]
    flux = data_dict["spectrum"]
    n_elements = len(flux)
    
    # Inizializza la maschera con quella esistente nel data_dict, se presente
    if "mask" in data_dict and data_dict["mask"] is not None:
        initial_mask = mask = np.ones(n_elements, dtype=bool)#np.copy(data_dict["mask"])
    else:
        initial_mask = np.ones_like(wavelengths, dtype=bool)
        
    mask = np.copy(initial_mask)
    selected_regions = []

    # 2. Setup della figura con estetica "formale"
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.1)
    ax = fig.add_subplot(gs[0])
    ax_check = fig.add_subplot(gs[1])
    
    # Spazio per il pulsante in basso
    plt.subplots_adjust(bottom=0.25)
    
    # Sposta a rest-frame
    ww = wavelengths / (1 + z)

    # Dati da visualizzare
    masked_flux = np.where(mask, flux, np.nan)       # Dati validi
    visible_flux = np.where(mask, np.nan, flux)      # Dati mascherati

    (original_line,) = ax.plot(ww, masked_flux, color="k", linewidth=1.2, label="Valid Spectrum")
    (masked_line,) = ax.plot(ww, visible_flux, color="crimson", alpha=0.7, linewidth=1.2, label="Masked Regions")

    # 3. Formattazione formale dell'asse dello spettro
    ax.set_xlabel(r"Rest-frame Wavelength ($\mathrm{\AA}$)", fontsize=14)
    ax.set_ylabel(r"Flux", fontsize=14)
    ax.set_title("Interactive Spectrum Masking", fontsize=16, weight="bold")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", length=4, color="gray")
    
    # Rimuovi bordi superflui per un look più pulito
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # 4. Funzioni di callback per l'interattività
    def update_plot():
        masked_flux_update = np.where(mask, flux, np.nan)
        visible_flux_update = np.where(mask, np.nan, flux)
        original_line.set_ydata(masked_flux_update)
        masked_line.set_ydata(visible_flux_update)
        # Usa draw_idle per un rendering più efficiente
        fig.canvas.draw_idle()

    def onselect(xmin, xmax):
        selected = (ww >= xmin) & (ww <= xmax)
        mask[selected] = False
        selected_regions.append((xmin, xmax))
        update_plot()

    def clear_selection(event):
        nonlocal mask
        # Il reset riporta la maschera allo stato originale letto da data_dict
        mask[:] = np.copy(initial_mask)
        selected_regions.clear()
        update_plot()

    # SpanSelector configurato con un colore in evidenza (rosso traslucido)
    span = SpanSelector(
        ax,
        onselect,
        "horizontal",
        useblit=True,
        props=dict(alpha=0.2, facecolor="red"),
        interactive=True,
    )

    # Pulsante di Reset
    ax_button = plt.axes([0.15, 0.05, 0.12, 0.06])
    button = Button(ax_button, "Reset Mask", color='lightgray', hovercolor='0.85')
    button.on_clicked(clear_selection)
    
    # 5. Creazione pannello di selezione per le righe di emissione
    lines_to_fit = {}
    ax_check.set_title("Emission Lines\nto Marginalize", fontsize=12, weight="bold")
    ax_check.set_xticks([])
    ax_check.set_yticks([])
    ax_check.set_frame_on(False)
    
    if line_dict:
        check_labels = list(line_dict.keys())
        check_states = [False] * len(check_labels)
        
        # CheckButtons di default
        check_buttons = CheckButtons(ax_check, check_labels, check_states)

        def toggle_line(label):
            if label in lines_to_fit:
                del lines_to_fit[label]
            else:
                lines_to_fit[label] = line_dict[label]

        check_buttons.on_clicked(toggle_line)
        
        # Plot delle linee (la tua funzione in utils.py)
        plot_emission_lines(ax, ww, flux, line_dict)
    else:
        # Messaggio placeholder se non ci sono linee nel config
        ax_check.text(0.5, 0.5, "No emission lines\nprovided", 
                      ha='center', va='center', color="gray", fontsize=10)

    ax.legend(loc="upper right", fontsize=12, frameon=True, framealpha=0.9)
    
    plt.show()

    return mask, lines_to_fit


def plot_unicode_spectrum(obs, model_spec=None):
    # 1. Estrazione dati dal dizionario obs
    wave = obs["wavelength"]
    spec = obs["spectrum"]
    mask = obs.get("mask", np.ones_like(wave, dtype=bool))

    # 2. Rimuoviamo preventivamente i NaN per evitare crash di Plotext
    valid_data = ~np.isnan(spec)
    wave = wave[valid_data]
    spec = spec[valid_data]
    mask = mask[valid_data]

    # Separazione pixel buoni (True) e mascherati (False)
    good_wave, good_spec = wave[mask], spec[mask]
    bad_wave, bad_spec = wave[~mask], spec[~mask]

    plttxt.clear_figure()

    # 3. Calcolo dei limiti intelligenti (Smart Y-limits)
    # 2. Calcolo dei limiti intelligenti
    if len(good_spec) > 0:
        y_min = np.percentile(good_spec, 1)
        y_max = np.percentile(good_spec, 99)

        margin = (y_max - y_min) * 0.1
        if margin == 0:
            margin = 0.1 * abs(y_min) if y_min != 0 else 1e-10

        y_bottom = y_min - margin
        y_top = y_max + margin
        plttxt.ylim(y_bottom, y_top)

        # --- NOVITÀ: Notazione scientifica forzata sull'asse Y ---
        # Creiamo 5 tacche equidistanti
        y_ticks = np.linspace(y_bottom, y_top, 5)
        # Le formattiamo in notazione scientifica (es. "1.50e-08")
        y_labels = [f"{val:.2e}" for val in y_ticks]
        # Le applichiamo al grafico
        plttxt.yticks(y_ticks, y_labels)

    plttxt.theme("clear")

    # 4. Plotting (RIMOSSO marker="x" per evitare l'IndexError)
    if len(bad_spec) > 0:
        # Usa plttxt.scatter ma con il marker di default (hd/braille)
        plttxt.scatter(bad_wave, bad_spec, color="cyan", label="Masked/Bad")

    if len(good_spec) > 0:
        # plttxt.plot(good_wave, good_spec, color="white", label="Observed Data")
        plttxt.scatter(good_wave, good_spec, color="white", label="Observed Data")

    if model_spec is not None:
        # Assicuriamoci che anche il modello sia filtrato dai NaN
        mod_valid = model_spec[valid_data]
        plttxt.plot(wave, mod_valid, color="yellow", label="Best Fit Model")

    plttxt.title("Prospector Fit - Osservazione vs Modello")
    plttxt.xlabel("Wavelength (Angstrom)")
    plttxt.ylabel("Flux (Maggies)")

    plttxt.plotsize(100, 30)
    plttxt.show()


def plot_spectrum(
    wavelengths,
    flux,
    flux_error,
    redshift=0,
    mask=None,
    title="Galaxy Spectrum",
    # Parametri opzionali per la fotometria
    phot_wavelengths=None,
    phot_flux=None,
    phot_flux_error=None,
    phot_label="Photometry",
    phot_marker="o",
    phot_color="red",
    phot_alpha=1.0,
    # Parametri opzionali per personalizzare linee di assorbimento
    absorption_lines=None,
    absorption_line_color="gray",
    absorption_line_style="--",
    absorption_line_alpha=0.7,
):
    """
    Plotta lo spettro di una galassia con bande di errore e punti fotometrici opzionali,
    corretti per lo spostamento verso il rosso.

    Se viene fornita una maschera booleana, il codice suddivide lo spettro in segmenti contigui:
      - I segmenti dove la maschera è True vengono plottati con il colore principale (dalla colormap "plasma")
      - I segmenti dove la maschera è False vengono plottati in blue
    """
    import matplotlib.pyplot as plt

    # Funzione helper per ottenere i segmenti contigui dati gli indici in cui la maschera assume un certo valore target
    def get_segments(mask, target):
        segments = []
        current_seg = []
        for i, val in enumerate(mask):
            if val == target:
                # Se l'indice corrente non è contiguo a quello precedente, inizia un nuovo segmento
                if current_seg and i != current_seg[-1] + 1:
                    segments.append(current_seg)
                    current_seg = []
                current_seg.append(i)
            else:
                if current_seg:
                    segments.append(current_seg)
                    current_seg = []
        if current_seg:
            segments.append(current_seg)
        return segments

    # Determina quali dati sono disponibili
    has_spectrum = (
        wavelengths is not None and flux is not None and flux_error is not None
    )
    has_photometry = phot_wavelengths is not None and phot_flux is not None

    # Correggi le linee di assorbimento per lo spostamento verso il rosso (se definite)
    if absorption_lines is not None:
        corrected_lines = {
            line: wl * (1 + redshift) for line, wl in absorption_lines.items()
        }
    else:
        corrected_lines = {}

    # Scegli la colormap e definisci i colori:
    cmap = plt.colormaps["plasma"]
    main_color = cmap(0.4)  # colore per i dati con mask True
    error_color = cmap(0.7)  # colore per la banda d'errore dei dati validi
    masked_color = "blue"  # colore per i segmenti con mask False

    plt.figure(figsize=(10, 6))

    if has_spectrum:
        # Se è definita una maschera, suddividi i dati in segmenti contigui
        if mask is not None:
            segments_true = get_segments(mask, True)
            segments_false = get_segments(mask, False)

            # Plot dei segmenti con mask True (dati "validi")
            first_true = True
            for seg in segments_true:
                seg_w = wavelengths[seg]
                seg_flux = flux[seg]
                seg_flux_err = flux_error[seg]
                if first_true:
                    plt.plot(
                        seg_w, seg_flux, label="Flux", color=main_color, linewidth=2
                    )
                    plt.fill_between(
                        seg_w,
                        seg_flux - seg_flux_err,
                        seg_flux + seg_flux_err,
                        color=error_color,
                        alpha=0.5,
                        label="Flux Error",
                    )
                    first_true = False
                else:
                    plt.plot(seg_w, seg_flux, color=main_color, linewidth=2)
                    plt.fill_between(
                        seg_w,
                        seg_flux - seg_flux_err,
                        seg_flux + seg_flux_err,
                        color=error_color,
                        alpha=0.5,
                    )

            # Plot dei segmenti con mask False (dati "mascherati")
            first_false = True
            for seg in segments_false:
                seg_w = wavelengths[seg]
                seg_flux = flux[seg]
                seg_flux_err = flux_error[seg]
                if first_false:
                    plt.plot(
                        seg_w,
                        seg_flux,
                        label="Masked Flux",
                        color=masked_color,
                        linewidth=2,
                    )
                    plt.fill_between(
                        seg_w,
                        seg_flux - seg_flux_err,
                        seg_flux + seg_flux_err,
                        color=masked_color,
                        alpha=0.5,
                        label="Masked Flux Error",
                    )
                    first_false = False
                else:
                    plt.plot(seg_w, seg_flux, color=masked_color, linewidth=2)
                    plt.fill_between(
                        seg_w,
                        seg_flux - seg_flux_err,
                        seg_flux + seg_flux_err,
                        color=masked_color,
                        alpha=0.5,
                    )
        else:
            # Se non viene fornita alcuna maschera, plottare lo spettro completo
            plt.plot(wavelengths, flux, label="Flux", color=main_color, linewidth=2)
            plt.fill_between(
                wavelengths,
                flux - flux_error,
                flux + flux_error,
                color=error_color,
                alpha=0.5,
                label="Flux Error",
            )

    # Plot della fotometria se disponibile
    if has_photometry:
        plt.errorbar(
            phot_wavelengths,
            phot_flux,
            yerr=phot_flux_error,
            fmt=phot_marker,
            color=phot_color,
            alpha=phot_alpha,
            label=phot_label,
            linestyle="none",
            markersize=6,
            capsize=4,
        )

    # Plot delle linee di assorbimento (se definite)
    for line_name, wl in corrected_lines.items():
        plt.axvline(
            wl,
            color=absorption_line_color,
            linestyle=absorption_line_style,
            linewidth=1,
            alpha=absorption_line_alpha,
            label=f"{line_name} ({wl:.1f} Å)",
        )

    # Etichette degli assi, titolo e formattazione del grafico
    plt.xlabel("Wavelength (Å)", fontsize=14)
    plt.ylabel("Flux", fontsize=14)
    plt.title(title, fontsize=16, weight="bold", color="black")
    plt.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.tick_params(axis="both", which="minor", length=4, color="gray")
    plt.legend(fontsize=12, loc="best", frameon=True, framealpha=0.9)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout()
    plt.show()
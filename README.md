# Prospector SED Fitting Pipeline

A Python pipeline for fitting galaxy spectral energy distributions (SEDs) using [Prospector](https://github.com/bd-j/prospector) with a non-parametric Continuity SFH model. Supports joint spectroscopy + photometry fitting, MPI parallelisation, and interactive spectrum masking.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Input Data Format](#input-data-format)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration Reference](#configuration-reference)
- [MPI / Batch Mode](#mpi--batch-mode)
- [Output Files](#output-files)
- [Code Structure](#code-structure)

---

## Overview

The pipeline reads galaxy data from HDF5 files, builds a Prospector model (Continuity SFH by default), and runs nested sampling with Dynesty. It supports:

- **Joint fitting** of photometry and spectroscopy
- **Non-parametric Continuity SFH** with configurable number of age bins
- **Nebular emission**, **dust emission**, **AGN**, and **birth-cloud dust** components
- **Instrumental LSF** correction using a JWST dispersion FITS file
- **Emission line marginalization**
- **Interactive spectrum masking** via a matplotlib GUI
- **MPI batch processing**: one galaxy per rank, or a single galaxy split across all ranks

---

## Requirements

```
prospector
dynesty
fsps / python-fsps
sedpy
astropy
scipy
numpy
matplotlib
rich
h5py
plotext
mpi4py        # optional, for parallel runs
```

---

## Input Data Format

Each galaxy must be stored in an **HDF5 file** with the following structure:

```
galaxy.h5
└── V1/                        ← version group (configurable via --version)
    ├── Photometry/
    │   ├── flux               ← array of flux values (Maggies)
    │   ├── flux_err           ← flux uncertainties
    │   ├── filters            ← array of sedpy filter name strings (e.g. b"hst_acs_wfc_f606w")
    │   └── mask               ← boolean mask (optional, 1 = valid)
    ├── Spectroscopy/
    │   ├── wavelength         ← observed-frame wavelengths in Ångström
    │   ├── flux               ← flux array
    │   ├── flux_err           ← flux uncertainty array
    │   └── mask               ← boolean mask (optional, 1 = valid)
    └── Metadata/
        └── redshift           ← scalar redshift value (optional; can also pass via CLI)
```

> **Notes:**
> - Filter names must be valid `sedpy` filter identifiers.
> - If a `redshift` is found in `Metadata`, it is used automatically unless `--redshift` is passed on the command line (CLI takes priority).
> - `mask` datasets are optional. If absent, all pixels are treated as valid.

---

## Quick Start

**Fit a single galaxy (photometry + spectroscopy):**

```bash
python run.py --file data/galaxy_A.h5
```

**Fit with a known redshift, disabling spectroscopy:**

```bash
python run.py --file data/galaxy_A.h5 --redshift 1.23 --no-spectroscopy
```

**Fit a list of galaxies in batch mode (serial):**

```bash
python run.py --file_list targets.txt --out_folder results/out
```

where `targets.txt` is a plain-text file with one HDF5 path per line.

---

## Usage Examples

### Minimal fit (photometry only)

```bash
python run.py \
    --file data/galaxy_B.h5 \
    --no-spectroscopy \
    --redshift 0.85 \
    --out_folder results/phot_only
```

### Full fit with JWST LSF correction

```bash
python run.py \
    --file data/galaxy_C.h5 \
    --dispersion_file jwst/nirspec_prism_disp.fits \
    --sigmav \
    --redshift 3.1 \
    --nbins 10 \
    --out_folder results/full_fit
```

### Interactive masking before fitting

Launches a matplotlib GUI where you can drag to mask spectral regions and optionally select emission lines to marginalise.

```bash
python run.py --file data/galaxy_D.h5 --interactive
```

### Log to file instead of terminal

```bash
python run.py --file data/galaxy_A.h5 --logging_file --log_folder results/logs
```

### Batch mode with MPI (one galaxy per rank)

```bash
mpirun -n 8 python run.py --file_list targets.txt --out_folder results/batch
```

### Single-galaxy parallel fit (all ranks cooperate on one target)

```bash
mpirun -n 16 python run.py --file data/big_galaxy.h5
```

---

## Configuration Reference

All options can be set via the command line. Boolean flags follow a `--flag` / `--no-flag` convention.

### Paths and I/O

| Argument | Type | Default | Description |
|---|---|---|---|
| `--file` | `str` | — | Path to a single HDF5 input file |
| `--file_list` | `str` | — | Path to a text file listing HDF5 paths (one per line) |
| `--out_folder` | `str` | `results/out` | Directory for output `.h5` result files |
| `--version` | `str` | `V1` | Version group name inside the HDF5 file |
| `--ext` | `str` | — | Optional suffix appended to the output filename |
| `--log_folder` | `str` | `results/log` | Directory for log files (requires `--logging_file`) |
| `--logging_file` | flag | `False` | Write logs to a per-galaxy file instead of the terminal |

### Data Selection

| Argument | Type | Default | Description |
|---|---|---|---|
| `--spectroscopy` / `--no-spectroscopy` | flag | `True` | Include/exclude spectroscopy |
| `--photometry` / `--no-photometry` | flag | `True` | Include/exclude photometry |
| `--use_mask` / `--no-use_mask` | flag | `True` | Apply the mask stored in the HDF5 file |
| `--filter_spec` / `--no-filter_spec` | flag | `True` | Auto-mask NaN/Inf/zero-error spectral pixels |
| `--filter_photo` / `--no-filter_photo` | flag | `True` | Auto-mask NaN/Inf/negative photometric pixels |

### Physical Model

| Argument | Type | Default | Description |
|---|---|---|---|
| `--redshift` | `float` | — | Galaxy redshift (overrides metadata value in HDF5) |
| `--fixed_z` / `--no-fixed_z` | flag | `False` | Fix metallicity (`logzsol`) instead of fitting it |
| `--nbins` | `int` | `8` | Number of age bins in the Continuity SFH |
| `--z_continuous` | `int` | `1` | FSPS `zcontinuous` parameter |
| `--nebular` / `--no-nebular` | flag | `True` | Include nebular emission |
| `--duste` / `--no-duste` | flag | `True` | Include dust emission (IR) |
| `--dust1` / `--no-dust1` | flag | `True` | Include birth-cloud dust (`dust1`) |
| `--agn` / `--no-agn` | flag | `False` | Include AGN component |
| `--sigmav` / `--no-sigmav` | flag | `True` | Apply instrumental LSF smoothing (requires `--dispersion_file`) |
| `--dispersion_file` | `str` | — | Path to a JWST NIRSpec dispersion FITS file |

### Fitting Engine

| Argument | Type | Default | Description |
|---|---|---|---|
| `--fit_outliers_spec` / `--no-fit_outliers_spec` | flag | `False` | Fit spectral outlier fraction |
| `--fit_outliers_photo` / `--no-fit_outliers_photo` | flag | `False` | Fit photometric outlier fraction |
| `--optimize` / `--no-optimize` | flag | `False` | Run optimisation step before sampling |

### Interactivity and Logging

| Argument | Type | Default | Description |
|---|---|---|---|
| `--interactive` / `--no-interactive` | flag | `False` | Open interactive masking GUI before fitting |
| `--verbose` / `--no-verbose` | flag | `True` | Print model parameters and data summary tables |

---

## MPI / Batch Mode

The pipeline automatically selects the correct execution mode based on the number of MPI ranks and targets:

| Situation | Behaviour |
|---|---|
| 1 target, 1 rank | Simple serial fit |
| 1 target, N ranks | Parallel Dynesty fit across all N ranks |
| M targets, N ranks (N ≤ M) | Each rank processes `M/N` galaxies sequentially |
| M targets, N ranks (N > M) | Excess ranks print a warning and exit cleanly |

---

## Output Files

Results are saved as HDF5 files in `--out_folder`. The filename is derived from the input file basename (plus `--ext` if provided):

```
results/out/
└── galaxy_A.h5        ← Prospector output (posteriors, model, obs, SPS)
results/log/
└── galaxy_A.log       ← per-galaxy log (only when --logging_file is set)
```

The output HDF5 contains the full Dynesty sampling chain, model parameters, and the observed data dictionary, and is compatible with standard Prospector post-processing tools (e.g. `prospect.io.read_results`).

---

## Code Structure

| File | Description |
|---|---|
| `run.py` | Entry point — MPI setup, CLI parsing, orchestrates the fitting pipeline |
| `config.py` | `FitConfig` dataclass — all configuration options and CLI argument parsing |
| `data_reader.py` | `GalaxyDataManager` — HDF5 reading, mask generation, sedpy filter building |
| `models.py` | `ContinuitySFH`, `BaseModel` — Prospector model parameter construction |
| `sps.py` | `ProspectorSPSBuilder` — SPS object initialisation and JWST LSF injection |
| `utils.py` | Interactive masking GUI, spectrum plotting utilities |
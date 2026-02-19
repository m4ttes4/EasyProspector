# EasyProspector
Configuring Prospector shouldn't be harder than the astrophysics itself.

# 🚀 Prospector MPI Wrapper: Scalable SED Fitting Made Easy

While [Prospector](https://github.com/bd-j/prospector) is an incredibly powerful tool for Spectral Energy Distribution (SED) fitting, setting it up from scratch can be notoriously tedious and error-prone. Building the `obs` dictionary, initializing the `sps` object, handling Line Spread Functions (LSF), and manually filtering out bad data (like NaNs or negative errors) usually requires hundreds of lines of boilerplate code.

This repository provides a **robust, MPI-ready wrapper** around Prospector. It automates data loading, smart masking, configuration management, and parallel execution. You focus on the science; the pipeline handles the software engineering.

## 🧠 Architecture Overview

1. **`FitConfig`**: The central "brain" of the pipeline. It stores all default values and dynamically updates them based on your Command Line Interface (CLI) inputs.
    
2. **`GalaxyDataManager`**: An intelligent data reader. It safely extracts data from HDF5 files, validates keys, creates boolean masks to filter out unphysical values, and builds the `sedpy` filters on the fly.
    
3. **`BaseModel` & `ProspectorSPSBuilder`**: These handle the physical model (Star Formation History, Dust, Nebular emission) and the Stellar Population Synthesis (SPS) setup, including instrumental resolution matching.
    
4. **`run.py`**: The orchestrator. It manages the MPI parallelization, assigns tasks to available cores, sets up isolated logging for each galaxy, and triggers the fitting engine (e.g., Dynesty).
    

---

## 📂 Data Input

The pipeline expects your pre-processed data to be stored in `.h5` files. To ensure seamless integration with the `GalaxyDataManager`, the HDF5 file should have the following internal structure:

```
📁 YourGalaxy.h5
 └─ 📁 V1 (or any custom version name, defined by --version)
     ├─ 📁 Metadata
     │   ├─ 🔢 redshift (float)
     │   └─ 🔢 ra, dec, etc. (optional)
     ├─ 📁 Photometry
     │   ├─ 🔢 flux (array)
     │   ├─ 🔢 flux_err (array)
     │   ├─ 🔢 filters (array of strings, e.g., b"sdss_u")
     │   └─ 🔢 mask (optional boolean array)
     └─ 📁 Spectroscopy
         ├─ 🔢 wavelength (array)
         ├─ 🔢 flux (array)
         ├─ 🔢 flux_err (array)
         └─ 🔢 mask (optional boolean array)
```

### Running Multiple Files (MPI Batch Mode)

If you have a large sample of galaxies, you don't need to run them one by one. You can provide a `.txt` file containing a list of `.h5` file paths using the `--file_list` argument.

When launched with `mpirun`, the pipeline will automatically distribute the galaxies across the available CPU cores using a Round-Robin approach. If a fit crashes due to bad data, the pipeline will log the error, isolate it, and safely move on to the next galaxy without killing the MPI environment.

---
## The Physical Model

The Prospector model is implemented as a class inside a separate `.py` file (e.g., `models.py`).

By default, the pipeline uses a highly flexible **Continuity SFH Model** (`BaseModel`). It is designed to be adaptable: you don't need to rewrite the code to change the physics. Features like dust emission, nebular lines, AGN components, and spectral smoothing can be toggled on or off directly from the terminal.

If you need a completely different parameterization (e.g., a parametric Tau-model), you can easily subclass the base model in the `.py` file and plug it in.

---

## ⚡ Execution & CLI Arguments

This script is designed to be executed primarily from the terminal.

**Single File (No MPI, or parallelizing the sampler engine across cores):**

```
python run.py --file ./data/NGC1234.h5 --out_folder ./results
# Or with MPI:
mpirun -n 4 python run.py --file ./data/NGC1234.h5 --out_folder ./results
```

**Multiple Files (Batch mode, 1 galaxy per core):**

```
mpirun -n 10 python run.py --file_list ./my_sample.txt --out_folder ./results
```

### Command Line Arguments

For boolean values (True/False), the system uses tristate flags. To set it to True, use `--flag`. To set it to False, use `--no-flag`. If omitted, the default defined in the class is used.

#### 1. I/O and Paths

|Argument|Type|Default|Description|
|---|---|---|---|
|`--name`|`str`|`"result"`|Identifier for the run/galaxy.|
|`--file`|`str`|`None`|Full path to the input `.h5` file.|
|`--file_list`|`str`|`None`|Path to a `.txt` file containing a list of `.h5` files.|
|`--out_folder`|`str`|`"results/out"`|Destination folder for the results.|
|`--log_folder`|`str`|`"results/log"`|Folder to store log files.|
|`--logging_file`|`bool`|`False`|(`--no-logging_file`) If True, saves logs to a file instead of just stdout.|
|`--version`|`str`|`"V1"`|The root group name inside the `.h5` file.|
|`--use_mask`|`bool`|`True`|(`--no-use_mask`) Use the boolean mask saved inside the H5 file.|
|`--dispersion_file`|`str`|`None`|Optional path to an LSF dispersion file (e.g., for JWST).|

#### 2. Data Selection

|Argument|Type|Default|Description|
|---|---|---|---|
|`--photometry`|`bool`|`True`|(`--no-photometry`) Enable/disable photometry fitting.|
|`--spectroscopy`|`bool`|`True`|(`--no-spectroscopy`) Enable/disable spectroscopy fitting.|
|`--filter_photo`|`bool`|`True`|(`--no-filter_photo`) Filter unphysical photometric values (NaNs, inf).|
|`--filter_spec`|`bool`|`True`|(`--no-filter_spec`) Filter unphysical spectroscopic values.|
|`--fit_outliers_photo`|`bool`|`False`|(`--no-fit_outliers_photo`) Model photometric outliers.|
|`--fit_outliers_spec`|`bool`|`False`|(`--no-fit_outliers_spec`) Model spectroscopic outliers.|

#### 3. Physics & Model

|Argument|Type|Default|Description|
|---|---|---|---|
|`--model_type`|`str`|`"ContinuitySFH"`|Type of physical model to build.|
|`--redshift`|`float`|`None`|Galaxy redshift (if None, reads from H5 metadata).|
|`--fixed_z`|`bool`|`False`|(`--no-fixed_z`) If True, redshift is fixed (not a free parameter).|
|`--nbins`|`int`|`8`|Number of age bins for the Star Formation History.|
|`--z_continuous`|`int`|`1`|FSPS parameter for metallicity interpolation.|
|`--nebular`|`bool`|`True`|(`--no-nebular`) Include nebular emission in the model.|
|`--duste`|`bool`|`True`|(`--no-duste`) Include dust thermal emission.|
|`--dust1`|`bool`|`True`|(`--no-dust1`) Include birth cloud dust attenuation.|
|`--agn`|`bool`|`False`|(`--no-agn`) Include AGN component.|
|`--sigmav`|`bool`|`True`|(`--no-sigmav`) Apply spectral smoothing (instrumental matching).|

#### 4. Fitting Engine

|Argument|Type|Default|Description|
|---|---|---|---|
|`--optimize`|`bool`|`False`|(`--no-optimize`) Enable Levenberg-Marquardt optimization.|
|`--emcee`|`bool`|`False`|(`--no-emcee`) Enable MCMC sampling.|
|`--dynesty`|`bool`|`True`|(`--no-dynesty`) Enable Dynamic Nested Sampling (recommended).|

#### 5. Debugging & Interactivity

|Argument|Type|Default|Description|
|---|---|---|---|
|`--verbose`|`bool`|`True`|(`--no-verbose`) Print detailed output and Rich tables.|
|`--interactive`|`bool`|`False`|(`--no-interactive`) Show plots and require manual user inputs.|

### Advanced Example

Running a quick test using only spectroscopy, disabling dust emission, and fixing the redshift to a specific value:

```
python run.py --file ./data.h5 --out_folder ./quick_test \
              --no-photometry --no-duste --no-dust1 --fixed_z --redshift 0.05
```
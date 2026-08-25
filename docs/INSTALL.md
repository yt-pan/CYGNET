# Installation

## Conda environment

From the package directory:

```bash
conda env create -f environment.yml
conda activate cygnet
python -m pip install --no-deps --no-build-isolation -e .
```

Conda supplies the compiled numerical dependencies and R `mgcv` backend.
Installing the local package with `--no-deps` prevents pip from replacing
those binary packages.

## Optional SpotClean backend

SpotClean is needed only for the optional Visium decontamination wrapper:

```bash
Rscript scripts/install_spotclean.R
python -c "import cygnet; print(cygnet.spotclean_available())"
```

The statistical tests, AnnData adapter, pipeline, permutation calibration, and
plotting helpers do not require SpotClean.

## Verification

```bash
python -m pytest -q
python scripts/benchmark_reference.py
Rscript -e "suppressPackageStartupMessages(library(mgcv)); packageVersion('mgcv')"
```

See [Windows notes](WINDOWS_INSTALL.md) if pip attempts to compile `chi2comb`
or if CYGNET resolves the wrong `Rscript` executable.

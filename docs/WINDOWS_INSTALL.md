# Windows installation notes

Use the supplied conda environment so compiled dependencies are installed from
binary packages:

```powershell
conda env create -f environment.yml
conda activate cygnet
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
```

## `chi2comb` build errors

If pip tries to compile `chi2comb`, reinstall the numerical dependencies from
conda-forge and then install CYGNET without dependency resolution:

```powershell
conda install -n cygnet -c conda-forge chiscore chi2comb -y
conda run -n cygnet python -m pip install --no-deps --no-build-isolation -e .
```

## R executable selection

CYGNET resolves `Rscript` from an explicit function argument, the
`CYGNET_RSCRIPT` environment variable, `PATH`, or the active conda environment.
If another R installation is selected, point CYGNET to the environment copy:

```powershell
$env:CYGNET_RSCRIPT = "$env:CONDA_PREFIX\Scripts\Rscript.exe"
```

Verify the `mgcv` backend with:

```powershell
Rscript -e "suppressPackageStartupMessages(library(mgcv)); packageVersion('mgcv')"
```

## Optional SpotClean backend

```powershell
Rscript scripts/install_spotclean.R
python -c "import cygnet; print(cygnet.spotclean_available())"
```

The Python analysis pipeline does not require SpotClean.

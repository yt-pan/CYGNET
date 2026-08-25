# Tutorial

Two compact lung AnnData fixtures are included for smoke testing the single-cell
and spot-level interfaces. They contain reduced, normalized inputs intended for
software validation rather than biological inference.

Run both examples from the package directory:

```bash
python examples/run_lung_fixture_pipeline.py --hide-progress
```

The script writes observed results, permutation results, calibrated results,
skipped-name tables, and run metadata under `docs/tutorial_outputs/`.

## Minimal AnnData example

```python
import anndata as ad
import cygnet

adata = ad.read_h5ad("docs/tutorial_data/lung_xenium_single_cell_fixture.h5ad")

result = cygnet.run_cygnet_pipeline_from_anndata(
    adata,
    genes=["MSR1", "CHIT1"],
    environment="pollutant",
    celltypes="celltype_label",
    celltype_names=["Macrophage"],
    n_permutations=8,
    n_jobs=1,
    store_key="cygnet",
)

print(result.results)
```

Continue with the [pipeline reference](PIPELINE.md) for input validation,
kernel settings, spot-level preprocessing, and result saving.

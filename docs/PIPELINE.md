# Analysis pipeline

`cygnet.run_cygnet_pipeline` is the high-level DataFrame interface.
`cygnet.run_cygnet_pipeline_from_anndata` provides the same workflow for
AnnData objects.

## DataFrame inputs

The four input tables must share observation identifiers:

- `locations`: spatial-coordinate columns.
- `celltypes`: cell-type indicators or composition components.
- `normalized_counts`: normalized expression, with genes in columns.
- `environment`: one or more columns representing one environmental feature
  or a basis expansion of that feature.

```python
result = cygnet.run_cygnet_pipeline(
    locations,
    celltypes,
    normalized_counts,
    environment,
    genes=None,
    celltype_names=None,
    n_permutations=10,
    n_jobs=8,
)
```

The result object contains `observed_results`, `permutation_results`, calibrated
`results`, `skipped_genes`, `skipped_celltypes`, and `metadata`.

## Permutation calibration

By default, CYGNET permutes the expression response and uses seeds
`0, ..., n_permutations - 1`. The default empirical probability is

```text
(1 + count(null <= observed)) / (1 + number of null values)
```

followed by Benjamini-Hochberg adjustment. The article analysis policy is
available with `permutation_ecdf_method="manuscript_interpolation"`.

Disable permutations with `permutation=False`, or permute cell-type rows with
`permutation_kind="celltype"`.

## Kernels

Environment and spatial matrices use RBF random Fourier features by default.
When gamma is omitted, CYGNET uses:

```text
gamma = 1 / median(pairwise squared distances)
```

The spatial setting used by most article analyses is selected with
`gamma_spatial_method="manuscript_range"`, which uses the reciprocal median
coordinate-column range. `gamma_spatial_method="median_half"` uses half the
inverse median squared-distance value and matches the corrected large Xenium
analysis when paired with its recorded deterministic subsample size. An
explicit numeric gamma takes precedence over an automatic method.

Pass `environment_kernel="values"` or `spatial_kernel="values"` when an input
is already a basis matrix.

## AnnData inputs

```python
result = cygnet.run_cygnet_pipeline_from_anndata(
    adata,
    environment="pollutant",
    celltypes="celltype_label",
    spatial_key="spatial",
    n_jobs=8,
    store_key="cygnet",
)
```

A categorical `obs` column is one-hot encoded. Numeric `obs` columns or an
`obsm` matrix are used directly. With `store_key="cygnet"`, serializable result
tables and metadata are attached to `adata.uns["cygnet"]`.

## Spot-level compositions

Use `prepare_spot_mode_inputs` to align inputs, apply CLR to a composition
matrix, and remove one redundant composition component:

```python
locations, celltypes_clr, counts, environment = cygnet.prepare_spot_mode_inputs(
    locations,
    celltype_proportions,
    normalized_counts,
    environment,
)
```

## Saving results

```python
paths = cygnet.save_cygnet_results(result, "cygnet_outputs", prefix="analysis")
```

The helper writes calibrated, observed, and permutation tables; skipped-name
tables; and run metadata.

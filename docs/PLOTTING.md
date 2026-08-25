# Plotting

CYGNET includes reusable plotting helpers for model results, spatial inputs,
partial-dependence curves, and variance decomposition.

## Result plots

- `plot_cygnet_qq`: compare observed and permutation-null p-values.
- `plot_feature_pdp`: visualize a fitted expression surface over cell-type and
  environmental values.
- `plot_pve`: compare trace-aware variance components.

## Spatial plots

- `plot_celltype_map`: categorical cell map or spot composition map.
- `plot_spatial_values`: spatial scatter plot for numeric values.
- `plot_spot_celltype_pies`: spot-level composition pies.
- `plot_celltype_env_overlay`: cell-type map over an environmental field.

```python
from cygnet._plot import plot_celltype_map, plot_spatial_values

plot_celltype_map(locations, celltypes, mode="auto")
plot_spatial_values(locations, environment["pollutant"])
```

Most helpers return Matplotlib figure and axes objects so titles, labels,
themes, and export settings can be adjusted without modifying CYGNET.

## Variance explained

Use `calculate_pve` with the same kernel features and fitted variance
components used by the model:

```python
pve = cygnet.calculate_pve(
    variance_components,
    interaction_features=interaction_features,
    environment_features=environment_features,
    spatial_features=spatial_features,
)
```

The calculation weights each variance component by the trace of its kernel and
includes residual variance in the denominator.

## Example gallery

Generate the compact example gallery with:

```bash
python examples/make_plot_previews.py
```

The article figure repository is distributed separately because it depends on
dataset-specific result tables and external source data.

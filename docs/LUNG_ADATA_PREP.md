# Lung AnnData preparation

CYGNET expects normalized expression, spatial coordinates, an environmental
feature, and cell-type information aligned to the same cells or spots.

```text
adata.X or adata.layers[layer]   observations x genes expression matrix
adata.var_names                  gene identifiers
adata.obs_names                  observation identifiers
adata.obsm["spatial"]            observations x 2 coordinates
adata.obs["environment"]         numeric environmental feature
```

Single-cell datasets may store categorical labels in `adata.obs["celltype"]`.
Spot-level datasets should store numeric composition columns or an `obsm`
matrix. Apply `prepare_spot_mode_inputs` before modeling spot compositions.

## Single-cell example

```python
import anndata as ad
import pandas as pd

expression = pd.read_csv("data/expression.csv", index_col=0)
locations = pd.read_csv("data/locations.csv", index_col=0)
metadata = pd.read_csv("data/metadata.csv", index_col=0)

common = expression.index.intersection(locations.index).intersection(metadata.index)
adata = ad.AnnData(expression.loc[common])
adata.obs = metadata.loc[common].copy()
adata.obsm["spatial"] = locations.loc[common, ["x", "y"]].to_numpy()
adata.write_h5ad("data/cygnet_input.h5ad")
```

## Spot-level example

```python
import anndata as ad
import cygnet
import pandas as pd

expression = pd.read_csv("data/expression.csv", index_col=0)
locations = pd.read_csv("data/locations.csv", index_col=0)
compositions = pd.read_csv("data/celltype_proportions.csv", index_col=0)
environment = pd.read_csv("data/environment.csv", index_col=0)

locations, compositions, expression, environment = cygnet.prepare_spot_mode_inputs(
    locations, compositions, expression, environment
)

adata = ad.AnnData(expression)
adata.obsm["spatial"] = locations[["x", "y"]].to_numpy()
adata.obs = environment.copy()
for column in compositions:
    adata.obs[column] = compositions[column]
```

The fixtures in `docs/tutorial_data/` demonstrate the expected serialized
layout on compact reduced inputs.

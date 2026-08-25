"""AnnData adapters for CYGNET."""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings

from .core import construct_inter_kernels, construct_rbf_kernel, run_cygnet


def _as_dense_matrix(matrix):
    """Return a dense numpy array from dense or scipy sparse inputs."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _align_pandas_to_obs(values, obs_names, name):
    """Align a pandas Series or DataFrame to AnnData observation names."""
    if values.index.equals(obs_names):
        return values

    missing = obs_names.difference(values.index)
    extra = values.index.difference(obs_names)
    if len(missing) or len(extra):
        raise ValueError(
            f"{name} cannot be aligned to adata.obs_names: "
            f"{len(missing)} missing cells and {len(extra)} extra cells."
        )

    warnings.warn(
        f"{name} index order differs from adata.obs_names; values were rearranged to match AnnData cells.",
        UserWarning,
        stacklevel=3,
    )
    return values.loc[obs_names]


def _resolve_observation_matrix(adata, value, *, name, allow_vector=True):
    """Resolve an AnnData field or external matrix into an obs-aligned numpy matrix.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix that defines the cell order.
    value : str, pandas.Series, pandas.DataFrame, array-like
        Input to resolve. Strings are interpreted as ``obs`` columns by default.
        Explicit AnnData namespaces can be addressed with ``obs:key``,
        ``obsm:key``, ``layers:key`` or ``X``.
    name : str
        Human-readable input name used in warnings and errors.
    allow_vector : bool, default True
        Whether one-dimensional inputs are accepted and reshaped to ``n_obs x 1``.

    Returns
    -------
    numpy.ndarray
        Numeric matrix with ``adata.n_obs`` rows.
    """
    obs_names = adata.obs_names

    if isinstance(value, str):
        namespace, sep, key = value.partition(":")
        if not sep:
            if value in adata.obs:
                source = adata.obs[[value]]
            elif value in adata.obsm:
                source = adata.obsm[value]
            elif value in adata.layers:
                source = adata.layers[value]
            elif value == "X":
                source = adata.X
            else:
                raise KeyError(
                    f"{name}={value!r} was not found in adata.obs, adata.obsm, adata.layers, or as 'X'."
                )
        elif namespace == "obs":
            if key not in adata.obs:
                raise KeyError(f"{name} obs field not found: {key}")
            source = adata.obs[[key]]
        elif namespace == "obsm":
            if key not in adata.obsm:
                raise KeyError(f"{name} obsm field not found: {key}")
            source = adata.obsm[key]
        elif namespace == "layers":
            if key not in adata.layers:
                raise KeyError(f"{name} layer not found: {key}")
            source = adata.layers[key]
        elif namespace == "X" and key == "":
            source = adata.X
        else:
            raise ValueError(
                f"Unsupported {name} reference {value!r}. Use an obs column, obsm key, "
                "'obs:<key>', 'obsm:<key>', 'layers:<key>', 'X', or an external vector/matrix."
            )
    else:
        source = value

    if isinstance(source, pd.Series):
        source = _align_pandas_to_obs(source, obs_names, name).to_frame()
    elif isinstance(source, pd.DataFrame):
        source = _align_pandas_to_obs(source, obs_names, name)

    matrix = _as_dense_matrix(source)
    if matrix.ndim == 1:
        if not allow_vector:
            raise ValueError(f"{name} must be a matrix, not a vector.")
        matrix = matrix.reshape(-1, 1)
    elif matrix.ndim != 2:
        raise ValueError(f"{name} must be a vector or 2D matrix; got shape {matrix.shape}.")

    if matrix.shape[0] != adata.n_obs:
        raise ValueError(
            f"{name} must have one row per AnnData observation: expected {adata.n_obs}, got {matrix.shape[0]}."
        )

    try:
        matrix = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc

    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite numeric values.")

    return matrix


def matrices_from_anndata(
    adata,
    *,
    gene,
    environment,
    celltype,
    covariates=None,
    spatial_key="spatial",
    layer=None,
    n_components=100,
    gamma_environment=None,
    gamma_spatial=None,
    gamma_environment_method="median",
    gamma_spatial_method="median",
    gamma_max_samples=1000,
    random_state=1,
):
    """Build CYGNET input matrices from an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with observations as cells or spots.
    gene : str
        Gene name in ``adata.var_names`` to use as the response.
    environment : str or array-like
        Per-cell environmental values. A string may refer to an ``obs`` column,
        ``obsm`` key, ``layers`` key, ``X``, or an explicit namespace such as
        ``"obs:env"`` or ``"obsm:env_matrix"``. External pandas objects are
        aligned to ``adata.obs_names``; external numpy arrays must already be in
        AnnData cell order.
    celltype : str or array-like
        Per-cell cell-type/context values. Accepted forms are the same as
        ``environment``.
    covariates : sequence of str, optional
        Observation columns to include as fixed covariates. If omitted, an
        intercept-only design is used.
    spatial_key : str, default "spatial"
        Key in ``adata.obsm`` containing spatial coordinates.
    layer : str, optional
        Layer to use for expression values instead of ``adata.X``.
    n_components : int, default 100
        Number of random Fourier features per RBF kernel.
    gamma_environment : float, optional
        Explicit RBF gamma for the environment kernel.
    gamma_spatial : float, optional
        Explicit RBF gamma for the spatial kernel.
    gamma_environment_method, gamma_spatial_method : str, default "median"
        Automatic gamma rule passed to :func:`construct_rbf_kernel` when the
        corresponding explicit gamma is omitted. Use ``"manuscript_range"``
        for the article range policy or ``"median_half"`` for the half-median
        squared-distance option.
    gamma_max_samples : int or None, default 1000
        Maximum observations used by distance-based automatic gamma rules.
    random_state : int, default 1
        Random state for Fourier feature construction.

    Returns
    -------
    dict
        Dictionary with ``y``, ``X``, ``x``, ``E``, ``S``, ``inter``,
        ``null_kernels`` and ``full_kernels`` entries.
    """
    if gene not in adata.var_names:
        raise KeyError(f"Gene not found in adata.var_names: {gene}")
    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial coordinates not found in adata.obsm: {spatial_key}")

    covariates = list(covariates or [])
    missing_covariates = [name for name in covariates if name not in adata.obs]
    if missing_covariates:
        raise KeyError(f"Covariate columns not found in adata.obs: {missing_covariates}")

    expression = adata.layers[layer] if layer is not None else adata.X
    expression = _as_dense_matrix(expression)
    gene_idx = adata.var_names.get_loc(gene)
    y = np.asarray(expression[:, gene_idx], dtype=float).reshape(-1)
    y = y - np.mean(y)

    if covariates:
        X = pd.get_dummies(adata.obs[covariates], drop_first=False, dtype=float).to_numpy()
        X = np.column_stack([np.ones(adata.n_obs), X])
    else:
        X = np.ones((adata.n_obs, 1), dtype=float)

    x = _resolve_observation_matrix(adata, celltype, name="celltype")
    environment_values = _resolve_observation_matrix(adata, environment, name="environment")
    spatial_values = np.asarray(adata.obsm[spatial_key], dtype=float)

    E, _ = construct_rbf_kernel(
        environment_values,
        gamma=gamma_environment,
        n=n_components,
        random_state=random_state,
        gamma_method=gamma_environment_method,
        gamma_max_samples=gamma_max_samples,
    )
    S, _ = construct_rbf_kernel(
        spatial_values,
        gamma=gamma_spatial,
        n=n_components,
        random_state=random_state,
        gamma_method=gamma_spatial_method,
        gamma_max_samples=gamma_max_samples,
    )
    inter = construct_inter_kernels(E, x)

    return {
        "y": y,
        "X": X,
        "x": x,
        "E": E,
        "S": S,
        "inter": inter,
        "null_kernels": [E, S],
        "full_kernels": [inter, E, S],
    }


def cygnet_from_anndata(adata, *, gene, environment, celltype, **kwargs):
    """Run the default CYGNET test for one gene from an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    gene : str
        Gene name to test.
    environment : str
        Observation column used to construct the environment kernel.
    celltype : str
        Observation column used as the tested cell-type/context vector.
    **kwargs
        Extra arguments forwarded to :func:`matrices_from_anndata`, plus
        optional ``maxiter`` for :func:`run_cygnet`.

    Returns
    -------
    tuple
        ``(eigenvalues, score, p_value, variance_components, convergence)``.
    """
    maxiter = kwargs.pop("maxiter", 100)
    matrices = matrices_from_anndata(
        adata,
        gene=gene,
        environment=environment,
        celltype=celltype,
        **kwargs,
    )
    return run_cygnet(
        matrices["y"],
        matrices["X"],
        matrices["x"],
        matrices["E"],
        matrices["null_kernels"],
        matrices["full_kernels"],
        maxiter=maxiter,
    )

"""High-level CYGNET analysis pipeline helpers."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .anndata import _as_dense_matrix, _resolve_observation_matrix
from .core import construct_inter_kernels, construct_rbf_kernel, run_cygnet, run_cygnet_permu
from .utils import add_empirical_fdr


@dataclass
class CygnetPipelineResult:
    """Container returned by :func:`run_cygnet_pipeline`."""

    results: pd.DataFrame
    observed_results: pd.DataFrame
    permutation_results: pd.DataFrame
    skipped_genes: list[str]
    skipped_celltypes: list[str]
    metadata: dict

    def to_dict(self):
        """Return the result as a dictionary suitable for AnnData ``uns`` storage."""
        return {
            "results": self.results,
            "observed_results": self.observed_results,
            "permutation_results": self.permutation_results,
            "skipped_genes": list(self.skipped_genes),
            "skipped_celltypes": list(self.skipped_celltypes),
            "metadata": dict(self.metadata),
        }


@contextmanager
def _tqdm_joblib(progress_bar):
    """Update a tqdm bar from joblib batch completions."""
    if progress_bar is None:
        yield
        return

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            progress_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        progress_bar.close()


def _as_frame(value, name):
    if isinstance(value, pd.Series):
        return value.to_frame()
    if isinstance(value, pd.DataFrame):
        # The pipeline treats inputs as read-only.  Returning the original
        # frame avoids duplicating large expression matrices supplied by the
        # AnnData adapter.
        return value
    matrix = np.asarray(value)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a vector or 2D matrix.")
    return pd.DataFrame(matrix)


def _align_frames(frames, *, only_common=False):
    reference = frames[0]
    if reference.index.has_duplicates:
        raise ValueError("Input DataFrame indexes cannot contain duplicates.")

    if only_common:
        common = reference.index
        for frame in frames[1:]:
            common = common.intersection(frame.index)
        if common.empty:
            raise ValueError("There are no common observation IDs across CYGNET inputs.")
        return [frame.loc[common] for frame in frames]

    if all(frame.index.equals(reference.index) for frame in frames[1:]):
        return frames

    reference_set = set(reference.index)
    aligned = []
    for frame in frames:
        if frame.index.has_duplicates:
            raise ValueError("Input DataFrame indexes cannot contain duplicates.")
        missing = reference_set - set(frame.index)
        extra = set(frame.index) - reference_set
        if missing or extra:
            raise ValueError(
                "CYGNET input DataFrames must contain the same observation IDs. "
                f"Missing: {len(missing)}; extra: {len(extra)}."
            )
        aligned.append(frame.loc[reference.index])
    return aligned


def _select_existing(requested, available, kind):
    available = list(available)
    if requested is None:
        return available, []
    if isinstance(requested, str):
        requested = [requested]

    selected = []
    skipped = []
    seen = set()
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        if name in available:
            selected.append(name)
        else:
            skipped.append(name)
    if skipped:
        warnings.warn(
            f"Skipping {len(skipped)} requested {kind} not found in the input: {skipped}",
            UserWarning,
            stacklevel=3,
        )
    return selected, skipped


def _numeric_frame(frame, name):
    if all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        out = frame
    else:
        try:
            out = frame.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only numeric values.") from exc

    values = out.to_numpy(copy=False)
    # A full np.isfinite(values) mask can itself be large. Checking column
    # blocks bounds peak memory while preserving the finite-value check.
    for start in range(0, values.shape[1], 64):
        if not np.isfinite(values[:, start : start + 64]).all():
            raise ValueError(f"{name} must contain only finite numeric values.")
    return out


def _kernel_from_frame(
    frame,
    method,
    *,
    n_components,
    gamma,
    gamma_method,
    gamma_max_samples,
    random_state,
    name,
):
    method = method.lower()
    values = frame.to_numpy(dtype=float)
    if method in {"rbf", "fourier"}:
        matrix, resolved_gamma = construct_rbf_kernel(
            values,
            gamma=gamma,
            n=n_components,
            random_state=random_state,
            gamma_method=gamma_method,
            gamma_max_samples=gamma_max_samples,
        )
        return matrix, resolved_gamma
    if method in {"values", "identity", "basis", "precomputed"}:
        return values, None
    raise ValueError(f"{name}_kernel must be 'rbf' or 'values'.")


def _ecdf_uses_plus_one(method):
    """Resolve a public permutation ECDF method to its low-level option."""
    method = str(method).lower()
    if method == "plus_one":
        return True
    if method == "manuscript_interpolation":
        return False
    raise ValueError(
        "permutation_ecdf_method must be 'plus_one' or "
        "'manuscript_interpolation'."
    )


def _drop_collinear_fixed_columns(fixed, target, names):
    fixed = np.asarray(fixed, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1, 1)
    if fixed.size == 0:
        return fixed.reshape(target.shape[0], 0), []

    kept = []
    dropped = []
    current = np.empty((target.shape[0], 0))
    current_rank = np.linalg.matrix_rank(np.column_stack([current, target]))
    for idx, name in enumerate(names):
        candidate = np.column_stack([current, fixed[:, idx], target])
        candidate_rank = np.linalg.matrix_rank(candidate)
        if candidate_rank > current_rank:
            kept.append(idx)
            current = fixed[:, kept]
            current_rank = candidate_rank
        else:
            dropped.append(name)
    return fixed[:, kept] if kept else np.empty((target.shape[0], 0)), dropped


def _serialize_value(value):
    if value is None:
        return None
    if isinstance(value, BaseException):
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        try:
            return float(arr)
        except (TypeError, ValueError):
            return str(value)
    return arr.tolist()


def _prepare_fit_context(
    celltypes,
    celltype_idx,
    covariates,
    covariate_names,
    include_intercept,
    E,
):
    """Build gene-independent design terms for one target cell type."""
    x = celltypes[:, celltype_idx]
    fixed_blocks = []
    fixed_names = []
    other_idx = [idx for idx in range(celltypes.shape[1]) if idx != celltype_idx]
    if other_idx:
        fixed_blocks.append(celltypes[:, other_idx])
        fixed_names.extend(
            [
                f"celltype:{name}"
                for idx, name in enumerate(covariate_names["celltypes"])
                if idx != celltype_idx
            ]
        )
    if covariates is not None and covariates.size:
        fixed_blocks.append(covariates)
        fixed_names.extend([f"covariate:{name}" for name in covariate_names["covariates"]])
    if include_intercept:
        fixed_blocks.insert(0, np.ones((celltypes.shape[0], 1)))
        fixed_names.insert(0, "intercept")

    fixed = np.column_stack(fixed_blocks) if fixed_blocks else np.empty((celltypes.shape[0], 0))
    fixed, dropped_fixed = _drop_collinear_fixed_columns(fixed, x, fixed_names)
    inter = construct_inter_kernels(E, x)
    return x, fixed, inter, dropped_fixed


def _run_single_task(
    gene_idx,
    gene_name,
    celltype_idx,
    celltype_name,
    seed,
    permutation_kind,
    counts,
    celltypes,
    covariates,
    covariate_names,
    include_intercept,
    E,
    S,
    maxiter,
    random_state,
    prepared_context=None,
):
    y = counts[:, gene_idx].astype(float)
    y = y - np.mean(y)

    if seed is not None and permutation_kind == "y":
        # A local RandomState provides a deterministic permutation without
        # mutating NumPy's process-wide RNG state.
        y = np.random.RandomState(seed).permutation(y)
    if prepared_context is not None:
        x, fixed, inter, dropped_fixed = prepared_context
    else:
        working_celltypes = celltypes
        if seed is not None and permutation_kind == "celltype":
        # Match the deterministic stream used by DataFrame sampling.
            order = np.random.RandomState(seed).permutation(celltypes.shape[0])
            working_celltypes = celltypes[order, :]
        elif seed is not None and permutation_kind not in {"y", "celltype"}:
            raise ValueError("permutation_kind must be 'y' or 'celltype'.")
        x, fixed, inter, dropped_fixed = _prepare_fit_context(
            working_celltypes,
            celltype_idx,
            covariates,
            covariate_names,
            include_intercept,
            E,
        )

    try:
        if seed is None:
            eigenvalues, score, p_value, varcom, cc_par = run_cygnet(
                y,
                fixed,
                x,
                E,
                [E, S],
                [inter, E, S],
                maxiter=maxiter,
            )
        else:
            eigenvalues, score, p_value, varcom, cc_par = run_cygnet_permu(
                y,
                fixed,
                x,
                E,
                [E, S],
                [inter, E, S],
                maxiter=maxiter,
            )
        error = None
        if isinstance(p_value, BaseException):
            error = f"{type(p_value).__name__}: {p_value}"
            p_value = np.nan
    except Exception as exc:
        eigenvalues, score, p_value, varcom, cc_par = None, np.nan, np.nan, None, np.nan
        error = f"{type(exc).__name__}: {exc}"

    return {
        "Gene": gene_name,
        "Celltype": celltype_name,
        "Analysis": "observed" if seed is None else "permutation",
        "seed": 0 if seed is None else int(seed),
        "p_value": float(p_value) if np.isscalar(p_value) and np.isfinite(p_value) else np.nan,
        "score": float(score) if np.isscalar(score) and np.isfinite(score) else np.nan,
        "varcom": _serialize_value(varcom),
        "eigenvalues": _serialize_value(eigenvalues),
        "cc_par": float(cc_par) if np.isscalar(cc_par) and np.isfinite(cc_par) else np.nan,
        "error": error,
        "dropped_fixed_columns": ";".join(dropped_fixed),
    }


def run_cygnet_pipeline(
    locations,
    celltypes,
    normalized_counts,
    environment,
    *,
    genes=None,
    celltype_names=None,
    covariates=None,
    n_permutations=8,
    permutation=True,
    permutation_seeds=None,
    permutation_kind="y",
    n_jobs=1,
    n_components=100,
    gamma_environment=None,
    gamma_spatial=None,
    gamma_environment_method="median",
    gamma_spatial_method="median",
    gamma_max_samples=1000,
    environment_kernel="rbf",
    spatial_kernel="rbf",
    include_intercept=False,
    random_state=1,
    maxiter=100,
    only_common=False,
    show_progress=True,
    fdr_fallback="bh",
    permutation_ecdf_method="plus_one",
):
    """Run CYGNET across selected genes and cell types.

    Missing requested genes or cell types are recorded and skipped. When
    permutation is disabled or ``n_permutations`` is zero, raw BH-adjusted
    p-values are still returned in the standard result table, but empirical
    permutation p-values remain unavailable. Automatic RBF kernels use the
    standard median squared-distance heuristic by default. The article range
    policy combines ``"manuscript_range"`` with
    ``permutation_ecdf_method="manuscript_interpolation"``. The
    ``"median_half"`` policy remains available independently.
    """
    locations_df = _numeric_frame(_as_frame(locations, "locations"), "locations")
    celltype_df = _numeric_frame(_as_frame(celltypes, "celltypes"), "celltypes")
    counts_df = _numeric_frame(_as_frame(normalized_counts, "normalized_counts"), "normalized_counts")
    environment_df = _numeric_frame(_as_frame(environment, "environment"), "environment")
    frames = [locations_df, celltype_df, counts_df, environment_df]
    if covariates is not None:
        covariates_df = _numeric_frame(_as_frame(covariates, "covariates"), "covariates")
        frames.append(covariates_df)
    else:
        covariates_df = None

    aligned = _align_frames(frames, only_common=only_common)
    locations_df, celltype_df, counts_df, environment_df = aligned[:4]
    covariates_df = aligned[4] if covariates_df is not None else None

    selected_genes, skipped_genes = _select_existing(genes, counts_df.columns, "genes")
    selected_celltypes, skipped_celltypes = _select_existing(celltype_names, celltype_df.columns, "cell types")
    if not selected_genes:
        raise ValueError("No requested genes are present in normalized_counts.")
    if not selected_celltypes:
        raise ValueError("No requested cell types are present in celltypes.")

    if selected_genes == list(counts_df.columns):
        counts_selected = counts_df
    else:
        counts_selected = counts_df.loc[:, selected_genes]
    celltype_column_index = {name: idx for idx, name in enumerate(celltype_df.columns)}
    E, resolved_gamma_environment = _kernel_from_frame(
        environment_df,
        environment_kernel,
        n_components=n_components,
        gamma=gamma_environment,
        gamma_method=gamma_environment_method,
        gamma_max_samples=gamma_max_samples,
        random_state=random_state,
        name="environment",
    )
    S, resolved_gamma_spatial = _kernel_from_frame(
        locations_df,
        spatial_kernel,
        n_components=n_components,
        gamma=gamma_spatial,
        gamma_method=gamma_spatial_method,
        gamma_max_samples=gamma_max_samples,
        random_state=random_state,
        name="spatial",
    )

    # Keep the stored expression precision here.  Each one-gene task converts
    # its response vector to float64 before fitting, so numerical behavior is
    # unchanged while float32 AnnData matrices use substantially less memory.
    counts_array = counts_selected.to_numpy(copy=False)
    celltype_array = celltype_df.to_numpy(dtype=float)
    covariate_array = covariates_df.to_numpy(dtype=float) if covariates_df is not None else None
    covariate_names = {
        "celltypes": list(celltype_df.columns),
        "covariates": list(covariates_df.columns) if covariates_df is not None else [],
    }
    prepared_contexts = None
    if permutation_kind == "y":
        prepared_contexts = {
            celltype_column_index[celltype_name]: _prepare_fit_context(
                celltype_array,
                celltype_column_index[celltype_name],
                covariate_array,
                covariate_names,
                include_intercept,
                E,
            )
            for celltype_name in selected_celltypes
        }
    observed_tasks = [
        (gene_idx, gene_name, celltype_column_index[celltype_name], celltype_name, None)
        for celltype_name in selected_celltypes
        for gene_idx, gene_name in enumerate(selected_genes)
    ]
    observed = _run_tasks(
        observed_tasks,
        "CYGNET observed",
        n_jobs,
        show_progress,
        counts_array,
        celltype_array,
        covariate_array,
        covariate_names,
        include_intercept,
        E,
        S,
        maxiter,
        random_state,
        permutation_kind,
        prepared_contexts,
    )
    observed_df = pd.DataFrame(observed)

    if permutation and n_permutations:
        if permutation_seeds is None:
            permutation_seeds = list(range(int(n_permutations)))
        else:
            permutation_seeds = [int(seed) for seed in permutation_seeds]
        permutation_tasks = [
            (gene_idx, gene_name, celltype_column_index[celltype_name], celltype_name, seed)
            for seed in permutation_seeds
            for celltype_name in selected_celltypes
            for gene_idx, gene_name in enumerate(selected_genes)
        ]
        permutation_records = _run_tasks(
            permutation_tasks,
            "CYGNET permutations",
            n_jobs,
            show_progress,
            counts_array,
            celltype_array,
            covariate_array,
            covariate_names,
            include_intercept,
            E,
            S,
            maxiter,
            random_state,
            permutation_kind,
            prepared_contexts,
        )
        permutation_df = pd.DataFrame(permutation_records)
    else:
        permutation_seeds = []
        permutation_df = pd.DataFrame(columns=observed_df.columns)

    results_df = add_empirical_fdr(
        observed_df,
        permutation_df,
        group_cols=("Celltype",),
        fallback=fdr_fallback,
        plus_one=_ecdf_uses_plus_one(permutation_ecdf_method),
    )
    metadata = {
        "genes_requested": None if genes is None else list(genes if not isinstance(genes, str) else [genes]),
        "genes_used": selected_genes,
        "celltypes_requested": None
        if celltype_names is None
        else list(celltype_names if not isinstance(celltype_names, str) else [celltype_names]),
        "celltypes_used": selected_celltypes,
        "celltype_columns_used_as_fixed_effects": list(celltype_df.columns),
        "n_observations": int(counts_df.shape[0]),
        "n_permutations": int(len(permutation_seeds)),
        "permutation_enabled": bool(permutation and permutation_seeds),
        "permutation_kind": permutation_kind,
        "n_jobs": n_jobs,
        "n_components": int(n_components),
        "environment_kernel": environment_kernel,
        "spatial_kernel": spatial_kernel,
        "gamma_environment": resolved_gamma_environment,
        "gamma_spatial": resolved_gamma_spatial,
        "gamma_environment_method": gamma_environment_method,
        "gamma_spatial_method": gamma_spatial_method,
        "gamma_max_samples": gamma_max_samples,
        "permutation_ecdf_method": permutation_ecdf_method,
        "include_intercept": bool(include_intercept),
        "random_state": int(random_state),
        "maxiter": int(maxiter),
    }
    return CygnetPipelineResult(
        results=results_df,
        observed_results=observed_df,
        permutation_results=permutation_df,
        skipped_genes=skipped_genes,
        skipped_celltypes=skipped_celltypes,
        metadata=metadata,
    )


def _run_tasks(
    tasks,
    description,
    n_jobs,
    show_progress,
    counts,
    celltypes,
    covariates,
    covariate_names,
    include_intercept,
    E,
    S,
    maxiter,
    random_state,
    permutation_kind,
    prepared_contexts=None,
):
    progress = tqdm(total=len(tasks), desc=description, disable=not show_progress)
    with _tqdm_joblib(progress):
        return Parallel(n_jobs=n_jobs)(
            delayed(_run_single_task)(
                gene_idx,
                gene_name,
                celltype_idx,
                celltype_name,
                seed,
                permutation_kind,
                counts,
                celltypes,
                covariates,
                covariate_names,
                include_intercept,
                E,
                S,
                maxiter,
                random_state,
                None if prepared_contexts is None else prepared_contexts[celltype_idx],
            )
            for gene_idx, gene_name, celltype_idx, celltype_name, seed in tasks
        )


def run_cygnet_pipeline_from_anndata(
    adata,
    *,
    genes=None,
    environment,
    celltypes,
    covariates=None,
    spatial_key="spatial",
    layer=None,
    store_key=None,
    **kwargs,
):
    """Run :func:`run_cygnet_pipeline` using matrices stored in AnnData."""
    expression = adata.layers[layer] if layer is not None else adata.X
    expression = _as_dense_matrix(expression)
    counts_df = pd.DataFrame(expression, index=adata.obs_names, columns=adata.var_names)

    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial coordinates not found in adata.obsm: {spatial_key}")
    locations_df = pd.DataFrame(
        np.asarray(adata.obsm[spatial_key], dtype=float),
        index=adata.obs_names,
        columns=[f"{spatial_key}_{i + 1}" for i in range(np.asarray(adata.obsm[spatial_key]).shape[1])],
    )
    environment_matrix = _resolve_observation_matrix(adata, environment, name="environment")
    environment_df = pd.DataFrame(
        environment_matrix,
        index=adata.obs_names,
        columns=_resolved_matrix_columns(adata, environment, environment_matrix, "environment"),
    )
    celltype_df = _celltype_frame_from_anndata(adata, celltypes)
    covariate_df = None
    if covariates is not None:
        covariates = [covariates] if isinstance(covariates, str) else list(covariates)
        missing = [name for name in covariates if name not in adata.obs]
        if missing:
            raise KeyError(f"Covariate columns not found in adata.obs: {missing}")
        covariate_df = pd.get_dummies(adata.obs[covariates], drop_first=False, dtype=float)

    result = run_cygnet_pipeline(
        locations_df,
        celltype_df,
        counts_df,
        environment_df,
        genes=genes,
        covariates=covariate_df,
        **kwargs,
    )
    if store_key is not None:
        attach_cygnet_results_to_anndata(adata, result, key=store_key)
    return result


def _resolved_matrix_columns(adata, key, matrix, prefix):
    if isinstance(key, str) and key in adata.obs and matrix.shape[1] == 1:
        return [key]
    return [f"{prefix}_{i + 1}" for i in range(matrix.shape[1])]


def _celltype_frame_from_anndata(adata, celltypes):
    if isinstance(celltypes, pd.DataFrame):
        return celltypes.loc[adata.obs_names]
    if isinstance(celltypes, pd.Series):
        values = celltypes.loc[adata.obs_names]
        if pd.api.types.is_numeric_dtype(values):
            return values.to_frame()
        return pd.get_dummies(values, dtype=float)
    if isinstance(celltypes, str):
        if celltypes in adata.obs:
            values = adata.obs[celltypes]
            if pd.api.types.is_numeric_dtype(values):
                return values.to_frame()
            return pd.get_dummies(values, dtype=float)
        if celltypes in adata.obsm:
            matrix = np.asarray(adata.obsm[celltypes], dtype=float)
            return pd.DataFrame(
                matrix,
                index=adata.obs_names,
                columns=[f"{celltypes}_{i + 1}" for i in range(matrix.shape[1])],
            )
        raise KeyError(f"celltypes={celltypes!r} was not found in adata.obs or adata.obsm.")

    if isinstance(celltypes, (list, tuple)):
        missing = [name for name in celltypes if name not in adata.obs]
        if missing:
            raise KeyError(f"Cell-type columns not found in adata.obs: {missing}")
        frame = adata.obs[list(celltypes)]
        if all(pd.api.types.is_numeric_dtype(frame[col]) for col in frame.columns):
            return frame.astype(float)
        return pd.get_dummies(frame, drop_first=False, dtype=float)

    matrix = np.asarray(celltypes, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.shape[0] != adata.n_obs:
        raise ValueError(f"celltypes must have {adata.n_obs} rows; got {matrix.shape[0]}.")
    return pd.DataFrame(matrix, index=adata.obs_names, columns=[f"celltype_{i + 1}" for i in range(matrix.shape[1])])


def attach_cygnet_results_to_anndata(adata, result, key="cygnet"):
    """Attach a :class:`CygnetPipelineResult` to ``adata.uns[key]``."""
    adata.uns[key] = result.to_dict() if isinstance(result, CygnetPipelineResult) else result
    return adata


def save_cygnet_results(result, output_dir, prefix=""):
    """Save CYGNET pipeline result tables and metadata to a directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = f"{prefix}_" if prefix and not str(prefix).endswith("_") else str(prefix)

    paths = {
        "results": output_path / f"{prefix}results.csv",
        "observed_results": output_path / f"{prefix}observed_results.csv",
        "permutation_results": output_path / f"{prefix}permutation_results.csv",
        "skipped_genes": output_path / f"{prefix}skipped_genes.csv",
        "skipped_celltypes": output_path / f"{prefix}skipped_celltypes.csv",
        "metadata": output_path / f"{prefix}metadata.json",
    }
    result.results.to_csv(paths["results"], index=False)
    result.observed_results.to_csv(paths["observed_results"], index=False)
    result.permutation_results.to_csv(paths["permutation_results"], index=False)
    pd.DataFrame({"Gene": result.skipped_genes}).to_csv(paths["skipped_genes"], index=False)
    pd.DataFrame({"Celltype": result.skipped_celltypes}).to_csv(paths["skipped_celltypes"], index=False)
    with paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump(result.metadata, handle, indent=2)
    return paths

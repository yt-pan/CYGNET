"""Multiple-testing utilities for CYGNET results."""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def empirical_p_values(observed_values, null_values, tail="lower", plus_one=True):
    """Estimate empirical p-values from observed values and a permutation null.

    Use ``tail="lower"`` for p-values, where smaller values are more
    significant, and ``tail="upper"`` for score statistics, where larger values
    are more significant.
    """
    observed = np.asarray(observed_values, dtype=float)
    null = np.asarray(null_values, dtype=float)
    observed_flat = observed.reshape(-1)
    null = null[np.isfinite(null)]
    if null.size == 0:
        raise ValueError("null_values cannot be empty")
    if tail not in {"lower", "upper"}:
        raise ValueError("tail must be 'lower' or 'upper'")

    out = np.full(observed_flat.shape, np.nan, dtype=float)
    valid = np.isfinite(observed_flat)
    offset = 1 if plus_one else 0
    denom = null.size + offset
    null_sorted = np.sort(null)
    if tail == "lower":
        counts = np.searchsorted(null_sorted, observed_flat[valid], side="right")
    else:
        counts = null.size - np.searchsorted(null_sorted, observed_flat[valid], side="left")
    out[valid] = (counts + offset) / denom
    return out.reshape(observed.shape)


def p_value_adjust_with_permutation_ecdf(p_values, permuted_p_values, plus_one=True):
    """Estimate empirical p-values from permutation-null p-values.

    The default uses the standard finite-permutation correction
    ``(1 + count(perm_p <= observed_p)) / (1 + n_perm)``. Set
    ``plus_one=False`` to linearly interpolate ranks ``1 / n, ..., n / n``
    with boundaries ``(0, 0)`` and
    ``(1, 1)``.
    """
    p_values = np.asarray(p_values)
    permuted_p_values = np.asarray(permuted_p_values)

    if np.any((p_values < 0) | (p_values > 1)) or np.any((permuted_p_values < 0) | (permuted_p_values > 1)):
        raise ValueError("All p-values must be between 0 and 1")
    if len(permuted_p_values) == 0:
        raise ValueError("permuted_p_values cannot be empty")

    if plus_one:
        return empirical_p_values(p_values, permuted_p_values, tail="lower", plus_one=True)

    sorted_perm = np.sort(permuted_p_values)
    ecdf = np.arange(1, len(sorted_perm) + 1) / len(sorted_perm)
    sorted_perm = np.concatenate([[0], sorted_perm, [1]])
    ecdf = np.concatenate([[0], ecdf, [1]])
    return np.interp(p_values, sorted_perm, ecdf)


def fdr_control(p_values, method="fdr_bh", permutation=False, permuted_p_values=None):
    """Control false discovery rate for observed p-values.

    Returns ``(adjusted_p_values, input_or_permutation_adjusted_p_values)``.
    """
    p_values = np.asarray(p_values)
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("All p-values must be between 0 and 1")

    if permutation:
        if permuted_p_values is None:
            raise ValueError("permuted_p_values cannot be None")
        p_values = p_value_adjust_with_permutation_ecdf(p_values, permuted_p_values)
    elif permuted_p_values is not None:
        print("Warning: permuted_p_values will be ignored since permutation is set to False")

    adjusted_p_values = multipletests(p_values, method=method)[1]
    return adjusted_p_values, p_values


def add_empirical_fdr(
    observed_results,
    permutation_results=None,
    value_col="p_value",
    group_cols=("Celltype",),
    tail="lower",
    method="fdr_bh",
    empirical_col="permutation_p_value",
    bh_col="FDR_permutation_adjusted_p_value",
    raw_bh_col="raw_bh_p_value",
    analytical_fdr_col="fdr_adjusted_p_value",
    permutation_fdr_col="permutation_fdr_adjusted_p_value",
    used_col="Used_Permutation",
    source_col="P_Value_Source",
    n_perm_col="n_permuted_null",
    fallback="bh",
    plus_one=True,
):
    """Attach analytical and permutation-calibrated multiple-testing columns.

    For the standard CYGNET p-value workflow, each group uses the plus-one
    finite-permutation correction and then applies BH. Set ``plus_one=False``
    to reproduce the interpolation-based manuscript ECDF. If a group lacks
    permutation null values, ``fallback="bh"``
    keeps the ordinary BH-adjusted analytical p-values.

    The returned table includes both descriptive column names
    ``fdr_adjusted_p_value`` and ``permutation_fdr_adjusted_p_value`` and the
    compatibility names ``raw_bh_p_value`` and
    ``FDR_permutation_adjusted_p_value``.
    """
    observed = pd.DataFrame(observed_results).copy()
    permutations = pd.DataFrame() if permutation_results is None else pd.DataFrame(permutation_results).copy()
    if value_col not in observed.columns:
        raise ValueError(f"observed_results is missing value_col: {value_col}")
    has_permutations = not permutations.empty
    if has_permutations and value_col not in permutations.columns:
        raise ValueError(f"permutation_results is missing value_col: {value_col}")
    if fallback not in {"bh", "raise"}:
        raise ValueError("fallback must be 'bh' or 'raise'")

    group_cols = tuple(group_cols or ())
    missing = [
        col
        for col in group_cols
        if col not in observed.columns or (has_permutations and col not in permutations.columns)
    ]
    if missing:
        raise ValueError(f"group_cols missing from observed or permutation results: {missing}")

    observed[raw_bh_col] = np.nan
    observed[analytical_fdr_col] = np.nan
    observed[empirical_col] = np.nan
    observed[bh_col] = np.nan
    observed[permutation_fdr_col] = np.nan
    observed[used_col] = False
    observed[source_col] = "raw_p_value_BH_no_permutation_file"
    observed[n_perm_col] = 0

    if group_cols and has_permutations:
        observed_groups = observed.groupby(list(group_cols), dropna=False, sort=False).indices.items()
        permutation_groups = {
            key if isinstance(key, tuple) else (key,): group
            for key, group in permutations.groupby(list(group_cols), dropna=False, sort=False)
        }
    else:
        observed_groups = (
            observed.groupby(list(group_cols), dropna=False, sort=False).indices.items()
            if group_cols
            else [((), observed.index.to_numpy())]
        )
        permutation_groups = {(): permutations} if has_permutations else {}

    for key, idx in observed_groups:
        key = key if isinstance(key, tuple) else (key,)
        raw_values = observed.loc[idx, value_col].to_numpy(dtype=float)
        valid = np.isfinite(raw_values)
        raw_bh = np.full(raw_values.shape, np.nan, dtype=float)
        if np.any(valid):
            raw_bh[valid] = multipletests(raw_values[valid], method=method)[1]
        observed.loc[idx, raw_bh_col] = raw_bh
        observed.loc[idx, analytical_fdr_col] = raw_bh

        permutation_group = permutation_groups.get(key)
        if permutation_group is None or permutation_group.empty:
            if fallback == "raise":
                raise ValueError(f"No permutation null values found for group {key}")
            observed.loc[idx, bh_col] = raw_bh
            observed.loc[idx, permutation_fdr_col] = raw_bh
            continue

        null_values = permutation_group[value_col].to_numpy(dtype=float)
        if tail == "lower":
            empirical = p_value_adjust_with_permutation_ecdf(
                raw_values,
                null_values,
                plus_one=plus_one,
            )
        else:
            empirical = empirical_p_values(raw_values, null_values, tail=tail, plus_one=plus_one)
        empirical_bh = np.full(empirical.shape, np.nan, dtype=float)
        valid_empirical = np.isfinite(empirical)
        if np.any(valid_empirical):
            empirical_bh[valid_empirical] = multipletests(empirical[valid_empirical], method=method)[1]

        observed.loc[idx, empirical_col] = empirical
        observed.loc[idx, bh_col] = empirical_bh
        observed.loc[idx, permutation_fdr_col] = empirical_bh
        observed.loc[idx, used_col] = True
        observed.loc[idx, source_col] = "permutation_p_value_BH"
        observed.loc[idx, n_perm_col] = np.isfinite(null_values).sum()

    return observed


# Re-export the file-based post-processing API.
from ._manuscript_utils import extract_significant_genes, run_celltype_fdr_analysis

# Preserve the documented utility import path.
from .core import check_and_rearrange_dataframes, load_simulation_data

"""Tests for the AnnData compatibility layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")

import cygnet


def _make_adata():
    """Create a small AnnData object with stable cell names."""
    rng = np.random.default_rng(11)
    obs = {
        "env": rng.normal(size=12),
        "celltype_score": rng.random(size=12),
        "batch": ["a"] * 6 + ["b"] * 6,
    }
    adata = anndata.AnnData(rng.normal(size=(12, 3)), obs=obs)
    adata.obs_names = [f"cell{i}" for i in range(12)]
    adata.var_names = ["g0", "g1", "g2"]
    adata.obsm["spatial"] = rng.normal(size=(12, 2))
    return adata


def test_matrices_from_anndata_builds_expected_shapes():
    """AnnData inputs should convert into CYGNET matrix arguments."""
    adata = _make_adata()

    matrices = cygnet.matrices_from_anndata(
        adata,
        gene="g1",
        environment="env",
        celltype="celltype_score",
        covariates=["batch"],
        n_components=5,
    )

    assert matrices["y"].shape == (12,)
    assert matrices["X"].shape[0] == 12
    assert matrices["x"].shape == (12, 1)
    assert matrices["E"].shape == (12, 5)
    assert matrices["S"].shape == (12, 5)
    assert len(matrices["null_kernels"]) == 2
    assert len(matrices["full_kernels"]) == 3


def test_external_pandas_inputs_are_aligned_with_warning():
    """External metadata with matching labels should be rearranged to AnnData order."""
    adata = _make_adata()
    reversed_names = list(reversed(adata.obs_names))
    environment = pd.Series(np.arange(12.0), index=reversed_names)
    celltype = pd.DataFrame({"score": np.linspace(0.1, 1.0, 12)}, index=reversed_names)

    with pytest.warns(UserWarning, match="rearranged"):
        matrices = cygnet.matrices_from_anndata(
            adata,
            gene="g1",
            environment=environment,
            celltype=celltype,
            n_components=4,
        )

    expected_environment = environment.loc[adata.obs_names].to_numpy().reshape(-1, 1)
    assert matrices["x"].shape == (12, 1)
    np.testing.assert_allclose(matrices["E"], cygnet.construct_rbf_kernel(expected_environment, n=4)[0])


def test_external_pandas_inputs_fail_when_cells_do_not_match():
    """External metadata should stop when labels cannot be aligned to AnnData cells."""
    adata = _make_adata()
    bad_environment = pd.Series(np.arange(12.0), index=[f"bad{i}" for i in range(12)])

    with pytest.raises(ValueError, match="cannot be aligned"):
        cygnet.matrices_from_anndata(
            adata,
            gene="g1",
            environment=bad_environment,
            celltype="celltype_score",
            n_components=4,
        )


def test_environment_can_come_from_obsm_matrix():
    """Environment can be a matrix stored inside AnnData, not only an obs column."""
    adata = _make_adata()
    adata.obsm["env_matrix"] = np.column_stack([adata.obs["env"], np.arange(adata.n_obs)])

    matrices = cygnet.matrices_from_anndata(
        adata,
        gene="g1",
        environment="obsm:env_matrix",
        celltype=adata.obs["celltype_score"],
        n_components=4,
    )

    assert matrices["E"].shape == (12, 4)
    assert matrices["x"].shape == (12, 1)

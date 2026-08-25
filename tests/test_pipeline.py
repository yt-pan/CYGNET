"""Tests for the high-level CYGNET pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

import cygnet


def _make_pipeline_inputs():
    rng = np.random.default_rng(20260603)
    n = 18
    index = [f"cell{i}" for i in range(n)]
    locations = pd.DataFrame(rng.normal(size=(n, 2)), index=index, columns=["x", "y"])
    environment = pd.DataFrame(rng.normal(size=(n, 1)), index=index, columns=["pollutant"])
    celltypes = pd.DataFrame(
        rng.normal(size=(n, 3)),
        index=index,
        columns=["Macrophage", "T_cell", "Tumor"],
    )
    counts = pd.DataFrame(
        rng.normal(size=(n, 4)),
        index=index,
        columns=["MSR1", "CHIT1", "CD14", "C5AR1"],
    )
    return locations, celltypes, counts, environment


def test_pipeline_runs_genes_celltypes_permutations_and_skips_missing(tmp_path):
    """The full pipeline should run selected valid inputs and record missing names."""
    locations, celltypes, counts, environment = _make_pipeline_inputs()

    with pytest.warns(UserWarning, match="Skipping"):
        result = cygnet.run_cygnet_pipeline(
            locations,
            celltypes,
            counts,
            environment,
            genes=["MSR1", "missing_gene", "CHIT1"],
            celltype_names=["Macrophage", "missing_ct"],
            n_permutations=2,
            n_components=4,
            n_jobs=1,
            maxiter=3,
            show_progress=False,
        )

    assert result.skipped_genes == ["missing_gene"]
    assert result.skipped_celltypes == ["missing_ct"]
    assert result.observed_results.shape[0] == 2
    assert result.permutation_results.shape[0] == 4
    assert set(result.permutation_results["seed"]) == {0, 1}
    assert set(result.results["Gene"]) == {"MSR1", "CHIT1"}
    assert set(result.results["Celltype"]) == {"Macrophage"}
    assert result.results["Used_Permutation"].all()
    assert (result.results["n_permuted_null"] == 4).all()
    expected_spatial_gamma = 1.0 / np.median(
        pdist(locations.to_numpy(), metric="sqeuclidean")
    )
    assert result.metadata["gamma_spatial"] == expected_spatial_gamma
    assert result.metadata["gamma_spatial_method"] == "median"
    assert result.metadata["permutation_ecdf_method"] == "plus_one"
    assert result.metadata["random_state"] == 1

    paths = cygnet.save_cygnet_results(result, tmp_path, prefix="lung")
    assert paths["results"].exists()
    assert paths["metadata"].exists()


def test_pipeline_can_reproduce_manuscript_gamma_and_ecdf():
    """The tutorial-facing compatibility settings remain explicit options."""
    locations, celltypes, counts, environment = _make_pipeline_inputs()
    result = cygnet.run_cygnet_pipeline(
        locations,
        celltypes,
        counts,
        environment,
        genes=["MSR1"],
        celltype_names=["Macrophage"],
        n_permutations=2,
        n_components=4,
        n_jobs=1,
        maxiter=3,
        show_progress=False,
        gamma_spatial_method="manuscript_range",
        permutation_ecdf_method="manuscript_interpolation",
    )
    expected = 1.0 / np.median(
        locations.max(axis=0).to_numpy() - locations.min(axis=0).to_numpy()
    )
    assert result.metadata["gamma_spatial"] == expected
    assert result.metadata["gamma_spatial_method"] == "manuscript_range"
    assert result.metadata["permutation_ecdf_method"] == "manuscript_interpolation"


def test_large_numeric_frames_keep_float32_storage_when_already_aligned():
    """Read-only fast paths should not duplicate or widen expression frames."""
    import cygnet.pipeline as pipeline

    index = pd.Index(["cell1", "cell2"])
    counts = pd.DataFrame(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        index=index,
        columns=["gene1", "gene2"],
    )
    locations = pd.DataFrame(
        np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64),
        index=index,
    )

    assert pipeline._as_frame(counts, "counts") is counts
    assert pipeline._numeric_frame(counts, "counts") is counts
    aligned = pipeline._align_frames([locations, counts])
    assert aligned[1] is counts
    assert counts.to_numpy(copy=False).dtype == np.float32

def test_pipeline_without_permutations_uses_raw_bh_fallback():
    """Permutation can be disabled while preserving the result table schema."""
    locations, celltypes, counts, environment = _make_pipeline_inputs()

    result = cygnet.run_cygnet_pipeline(
        locations,
        celltypes,
        counts,
        environment,
        genes=["MSR1", "CHIT1"],
        celltype_names=["Macrophage"],
        permutation=False,
        n_components=4,
        n_jobs=1,
        maxiter=3,
        show_progress=False,
    )

    assert result.permutation_results.empty
    assert not result.results["Used_Permutation"].any()
    assert "raw_bh_p_value" in result.results
    assert "fdr_adjusted_p_value" in result.results
    assert "FDR_permutation_adjusted_p_value" in result.results
    assert "permutation_fdr_adjusted_p_value" in result.results
    pd.testing.assert_series_equal(
        result.results["FDR_permutation_adjusted_p_value"],
        result.results["raw_bh_p_value"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result.results["permutation_fdr_adjusted_p_value"],
        result.results["fdr_adjusted_p_value"],
        check_names=False,
    )


def test_pipeline_permutation_uses_manuscript_randomstate(monkeypatch):
    import cygnet.pipeline as pipeline_module

    captured = {}

    def fake_run(y, fixed, x, E, null_kernels, full_kernels, maxiter=100):
        captured["y"] = np.asarray(y).reshape(-1)
        return np.array([1.0]), 1.0, 0.5, None, 0.0

    monkeypatch.setattr(pipeline_module, "run_cygnet_permu", fake_run)
    counts = np.arange(8.0).reshape(-1, 1)
    celltypes = np.column_stack([np.arange(8.0), np.arange(8.0) ** 2 + 1])
    E = np.column_stack([np.linspace(0.0, 1.0, 8), np.ones(8)])
    S = np.column_stack([np.linspace(1.0, 2.0, 8), np.ones(8)])
    pipeline_module._run_single_task(
        0,
        "gene",
        0,
        "Ct1",
        3,
        "y",
        counts,
        celltypes,
        None,
        {"celltypes": ["Ct1", "Ct2"], "covariates": []},
        False,
        E,
        S,
        5,
        1,
    )
    centered = counts[:, 0] - counts[:, 0].mean()
    expected = np.random.RandomState(3).permutation(centered)
    np.testing.assert_array_equal(captured["y"], expected)


def test_selected_celltype_keeps_other_celltypes_as_fixed_effects(monkeypatch):
    """Selecting target cell types should not remove background cell-type effects."""
    import cygnet.pipeline as pipeline

    locations, celltypes, counts, environment = _make_pipeline_inputs()
    captured = {}

    def fake_run_cygnet(y, fixed, x, E, kernel_lst, full_kernel_lst, maxiter=100):
        captured["fixed_shape"] = fixed.shape
        captured["x"] = x.copy()
        return np.array([0.5, 1.0]), 1.0, 0.2, np.ones(4), 0.0

    monkeypatch.setattr(pipeline, "run_cygnet", fake_run_cygnet)

    result = cygnet.run_cygnet_pipeline(
        locations,
        celltypes,
        counts,
        environment,
        genes=["MSR1"],
        celltype_names=["Macrophage"],
        permutation=False,
        n_components=4,
        n_jobs=1,
        maxiter=3,
        show_progress=False,
    )

    assert result.results.loc[0, "p_value"] == 0.2
    assert captured["fixed_shape"] == (celltypes.shape[0], celltypes.shape[1] - 1)
    np.testing.assert_allclose(captured["x"], celltypes["Macrophage"].to_numpy())
    assert result.metadata["celltypes_used"] == ["Macrophage"]
    assert result.metadata["celltype_columns_used_as_fixed_effects"] == list(celltypes.columns)


def test_pipeline_from_anndata_can_store_results():
    """The AnnData wrapper should build matrices, run the pipeline, and store outputs."""
    anndata = pytest.importorskip("anndata")
    locations, celltypes, counts, environment = _make_pipeline_inputs()
    adata = anndata.AnnData(counts.to_numpy(), obs=pd.DataFrame(index=counts.index))
    adata.var_names = counts.columns
    adata.obsm["spatial"] = locations.to_numpy()
    adata.obs["pollutant"] = environment["pollutant"].to_numpy()
    for col in celltypes.columns:
        adata.obs[col] = celltypes[col].to_numpy()

    result = cygnet.run_cygnet_pipeline_from_anndata(
        adata,
        genes=["MSR1"],
        environment="pollutant",
        celltypes=["Macrophage", "T_cell"],
        celltype_names=["Macrophage"],
        permutation=False,
        n_components=3,
        maxiter=3,
        show_progress=False,
        store_key="cygnet_test",
    )

    assert result.observed_results.shape[0] == 1
    assert "cygnet_test" in adata.uns
    assert "results" in adata.uns["cygnet_test"]


def test_rank_errors_raise_value_error_instead_of_exiting():
    """Low-level rank failures should be regular exceptions for callers to handle."""
    y = np.arange(6.0)
    X = np.ones((6, 1))
    x = np.ones(6)
    E = np.random.default_rng(1).normal(size=(6, 2))

    with pytest.raises(ValueError, match="not column full rank"):
        cygnet.cygnet_davies(y, X, x, E, [E], maxiter=2)

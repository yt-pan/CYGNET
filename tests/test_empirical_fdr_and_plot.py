from __future__ import annotations

import numpy as np
import pandas as pd

import cygnet
from cygnet._plot import (
    plot_celltype_env_overlay,
    plot_celltype_map,
    plot_feature_pdp,
    plot_pve,
    plot_qq,
    plot_spatial_values,
    plot_spot_celltype_pies,
)


def test_empirical_p_values_use_plus_one_lower_tail():
    observed = np.array([0.01, 0.20, 0.80])
    null = np.array([0.005, 0.02, 0.50, 0.90])
    result = cygnet.empirical_p_values(observed, null, tail="lower")
    np.testing.assert_allclose(result, np.array([2 / 5, 3 / 5, 4 / 5]))


def test_permutation_ecdf_defaults_to_plus_one():
    observed = np.array([0.01, 0.20, 0.80])
    null = np.array([0.005, 0.02, 0.50, 0.90])
    np.testing.assert_allclose(
        cygnet.p_value_adjust_with_permutation_ecdf(observed, null),
        np.array([2 / 5, 3 / 5, 4 / 5]),
    )


def test_permutation_ecdf_can_use_manuscript_interpolation():
    observed = np.array([0.01, 0.20, 0.80])
    null = np.array([0.005, 0.02, 0.50, 0.90])
    expected = np.interp(
        observed,
        np.array([0.0, 0.005, 0.02, 0.50, 0.90, 1.0]),
        np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.0]),
    )
    np.testing.assert_allclose(
        cygnet.p_value_adjust_with_permutation_ecdf(
            observed, null, plus_one=False
        ),
        expected,
    )


def test_add_empirical_fdr_groups_by_celltype_and_falls_back_to_bh():
    observed = pd.DataFrame(
        {
            "Gene": ["g1", "g2", "g3", "g4"],
            "Celltype": ["ct1", "ct1", "ct2", "ct2"],
            "p_value": [0.01, 0.20, 0.03, 0.40],
        }
    )
    permutations = pd.DataFrame(
        {
            "Gene": ["p1", "p2", "p3", "p4"],
            "Celltype": ["ct1", "ct1", "ct1", "ct1"],
            "p_value": [0.005, 0.02, 0.50, 0.90],
        }
    )
    result = cygnet.add_empirical_fdr(observed, permutations)
    np.testing.assert_allclose(result.loc[0, "permutation_p_value"], 2 / 5)
    np.testing.assert_allclose(result.loc[1, "permutation_p_value"], 3 / 5)
    np.testing.assert_allclose(result.loc[0, "FDR_permutation_adjusted_p_value"], 3 / 5)
    np.testing.assert_allclose(result.loc[0, "permutation_fdr_adjusted_p_value"], 3 / 5)
    assert result.loc[0, "fdr_adjusted_p_value"] == result.loc[0, "raw_bh_p_value"]
    assert bool(result.loc[0, "Used_Permutation"]) is True
    assert bool(result.loc[2, "Used_Permutation"]) is False
    assert result.loc[2, "FDR_permutation_adjusted_p_value"] == result.loc[2, "raw_bh_p_value"]
    assert result.loc[2, "permutation_fdr_adjusted_p_value"] == result.loc[2, "fdr_adjusted_p_value"]


def test_plot_helpers_return_axes_with_agg_backend():
    import matplotlib

    matplotlib.use("Agg", force=True)
    ax = plot_qq([0.1, 0.4, 0.8], labels=["CYGNET"])
    assert ax.get_xlabel()

    locations = pd.DataFrame({"x": [0.0, 1.0, 0.0, 1.0, 0.5], "y": [0.0, 0.0, 1.0, 1.0, 0.5]})
    celltypes = pd.DataFrame(
        {
            "Macrophage": [0.8, 0.2, 0.6, 0.1, 0.4],
            "Tumor": [0.2, 0.8, 0.4, 0.9, 0.6],
        }
    )
    env = pd.DataFrame({"pollutant": [0.9, 0.7, 0.3, 0.1, 0.5]})
    counts = pd.DataFrame({"MSR1": [2.0, 1.1, 1.6, 0.8, 1.3]})
    labels = pd.Series(["Macrophage", "Tumor", "Macrophage", "Tumor", "Macrophage"], name="celltype")

    assert plot_spatial_values(locations, env, "pollutant").get_xlabel() == "x"
    assert plot_celltype_map(locations, labels).get_xlabel() == "x"
    assert len(plot_celltype_map(locations, celltypes, legend=False).patches) > 0
    assert len(plot_celltype_map(locations, celltypes[["Macrophage"]], legend=False).patches) > 0
    one_hot = pd.DataFrame(
        {
            "Macrophage": [1, 0, 1, 0, 1],
            "Tumor": [0, 1, 0, 1, 0],
        }
    )
    assert len(plot_celltype_map(locations, one_hot, legend=False).collections) >= 2
    assert len(plot_celltype_map(locations, one_hot[["Macrophage"]], legend=False).collections) >= 2
    assert plot_spot_celltype_pies(locations, celltypes, legend=False).get_xlabel() == "x"
    assert plot_celltype_env_overlay(locations, celltypes[["Macrophage"]], env, "Macrophage", "pollutant").get_xlabel() == "x"
    spot_overlay_ax = plot_celltype_env_overlay(locations, celltypes, env, env_col="pollutant")
    assert len(spot_overlay_ax.patches) > 0
    one_hot_overlay_ax = plot_celltype_env_overlay(locations, one_hot, env, env_col="pollutant")
    assert len(one_hot_overlay_ax.collections) >= 2
    overlay_ax = plot_celltype_env_overlay(
        locations,
        celltypes[["Macrophage"]],
        env,
        "Macrophage",
        "pollutant",
        celltype_labels=labels,
    )
    assert overlay_ax.get_xlabel() == "x"
    axes = plot_feature_pdp(counts, celltypes, env, gene="MSR1", celltype="Macrophage", env="pollutant", n_neighbors=2)
    assert len(np.atleast_1d(axes).ravel()) == 3


def test_trace_aware_pve_uses_component_traces():
    z1 = np.ones((3, 2))
    z2 = np.eye(3)
    result = cygnet.calculate_pve([2.0, 3.0, 4.0], [z1, z2], effect_names=["CxE", "Spatial"])
    np.testing.assert_allclose(result["Contribution"].to_numpy(), np.array([12.0, 9.0, 12.0]))
    np.testing.assert_allclose(result["PVE"].sum(), 1.0)
    assert plot_pve(result).get_ylabel() == "Proportion of variance explained"

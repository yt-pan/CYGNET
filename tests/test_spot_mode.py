"""Tests for spot-level preprocessing helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import cygnet
import cygnet.spot as spot


def test_prepare_spot_mode_inputs_aligns_and_applies_clr():
    """Spot mode should align matrices, CLR-transform, and drop the smallest cell type."""
    spots = ["s1", "s2", "s3"]
    locations = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [3.0, 4.0, 5.0]}, index=spots)
    celltypes = pd.DataFrame(
        {"ct_a": [0.2, 0.5, 0.7], "ct_b": [0.8, 0.5, 0.3]},
        index=["s3", "s1", "s2"],
    )
    counts = pd.DataFrame({"gene1": [1.0, 2.0, 3.0]}, index=["s2", "s3", "s1"])
    env = pd.DataFrame({"env1": [2.0, 4.0, 6.0]}, index=["s1", "s3", "s2"])

    out_locations, out_celltypes, out_counts, out_env = cygnet.prepare_spot_mode_inputs(
        locations,
        celltypes,
        counts,
        env,
    )

    assert out_locations.index.tolist() == spots
    assert out_celltypes.index.tolist() == spots
    assert out_counts.index.tolist() == spots
    assert out_env.index.tolist() == spots

    expected = cygnet.celltype_clr_transform_from_df(celltypes.loc[spots], drop_column="auto")
    pd.testing.assert_frame_equal(out_celltypes, expected)
    assert out_celltypes.columns.tolist() == ["ct_b"]


def test_prepare_spot_mode_inputs_can_drop_named_clr_column_or_keep_all():
    """Spot mode should allow an explicit CLR drop column or the full CLR matrix."""
    spots = ["s1", "s2", "s3"]
    locations = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [3.0, 4.0, 5.0]}, index=spots)
    celltypes = pd.DataFrame(
        {
            "ct_a": [0.2, 0.4, 0.6],
            "ct_b": [0.7, 0.5, 0.3],
            "ct_c": [0.1, 0.1, 0.1],
        },
        index=spots,
    )
    counts = pd.DataFrame({"gene1": [1.0, 2.0, 3.0]}, index=spots)
    env = pd.DataFrame({"env1": [2.0, 4.0, 6.0]}, index=spots)

    _, named_drop, _, _ = cygnet.prepare_spot_mode_inputs(
        locations,
        celltypes,
        counts,
        env,
        clr_drop_column="ct_b",
    )
    assert named_drop.columns.tolist() == ["ct_a", "ct_c"]

    _, full_clr, _, _ = cygnet.prepare_spot_mode_inputs(
        locations,
        celltypes,
        counts,
        env,
        clr_drop_column=None,
    )
    assert full_clr.columns.tolist() == ["ct_a", "ct_b", "ct_c"]
    np.testing.assert_allclose(full_clr.mean(axis=1), 0.0, atol=1e-12)


def test_load_simulation_data_spot_mode_applies_clr(tmp_path):
    """mode='spot' should make simulated cell-type indicators CLR-transformed."""
    simulation_dir = tmp_path / "simulation_1"
    simulation_dir.mkdir()
    spots = [f"Spot{i}" for i in range(1, 7)]

    pd.DataFrame({"x": np.arange(6), "y": np.arange(6) + 1}, index=spots).to_csv(
        simulation_dir / "sim_1_1_locations.csv"
    )
    pd.DataFrame({"celltype": ["Ct1", "Ct1", "Ct1", "Ct1", "Ct2", "Ct2"]}, index=spots).to_csv(
        simulation_dir / "sim_1_1_celltypes.csv"
    )
    pd.DataFrame({"gene1": np.arange(6.0), "gene2": np.arange(6.0) + 2}, index=spots).to_csv(
        simulation_dir / "sim_1_1_scater_normalized_counts.csv"
    )
    pd.DataFrame({"env1": np.linspace(-1.0, 1.0, 6)}, index=spots).to_csv(
        simulation_dir / "sim_1_1_values_env.csv"
    )

    with pytest.warns(UserWarning, match="Some values are 0"):
        _, celltype_df, _, _ = cygnet.load_simulation_data(
            str(simulation_dir),
            seed=1,
            normalized_type="scater",
            mode="spot",
        )

    assert celltype_df.shape == (6, 1)
    assert celltype_df.index.tolist() == spots
    assert celltype_df.columns.tolist() == ["Ct1"]
    assert not np.isin(celltype_df.to_numpy(), [0.0, 1.0]).all()


def test_missing_spotclean_r_packages_parses_rscript_output(monkeypatch):
    """The R package checker should report packages missing from R."""
    captured = {}

    def fake_run(command, capture_output, text, check):
        captured["command"] = command
        return SimpleNamespace(returncode=1, stdout="SpotClean\nSeurat\n", stderr="")

    monkeypatch.setattr(spot.subprocess, "run", fake_run)

    missing = cygnet.missing_spotclean_r_packages(rscript="Rscript")

    assert missing == ["SpotClean", "Seurat"]
    assert captured["command"][0] == "Rscript"
    assert captured["command"][1] == "-e"
    assert "requireNamespace" in captured["command"][2]


def test_run_spotclean_10x_builds_standard_r_workflow(monkeypatch, tmp_path):
    """The SpotClean wrapper should call Rscript with the standard workflow."""
    raw_dir = tmp_path / "raw_feature_bc_matrix"
    spatial_dir = tmp_path / "spatial"
    output_dir = tmp_path / "spotclean"
    raw_dir.mkdir()
    spatial_dir.mkdir()
    captured = {}

    def fake_run(command, capture_output, text, check):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(spot.subprocess, "run", fake_run)

    outputs = cygnet.run_spotclean_10x(
        raw_dir,
        spatial_dir,
        output_dir=output_dir,
        rscript="Rscript",
        min_cells=7,
        min_features=3,
        save_rds=False,
    )

    command = captured["command"]
    assert command[0] == "Rscript"
    assert command[1] == "-e"
    assert "spotclean(slide_obj)" in command[2]
    assert "SCTransform" in command[2]
    assert command[3] == str(raw_dir)
    assert command[4] == str(spatial_dir)
    assert command[5] == str(output_dir)
    assert command[6:9] == ["7", "3", "1.3"]
    assert command[9] == "0"
    assert outputs == {
        "raw_counts": output_dir / "raw_counts_spotclean.csv",
        "sct_scaled_counts": output_dir / "SCTscaled_counts_spotclean.csv",
    }

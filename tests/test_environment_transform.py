"""Tests for environment spline transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cygnet
from cygnet.core import load_simulation_data


def test_statsmodels_bspline_transform_shape_and_index():
    """The optional statsmodels backend should preserve rows and create k columns."""
    env = pd.DataFrame({"env1": np.linspace(-1.0, 1.0, 20)}, index=[f"cell{i}" for i in range(20)])

    transformed = cygnet.transform_environment(env, method="statsmodels", k=10)

    assert transformed.shape == (20, 10)
    assert transformed.index.equals(env.index)
    assert transformed.columns.tolist() == [f"env1_bspline_{i}" for i in range(1, 11)]


def test_transform_environment_values_preserves_multiple_columns():
    """The raw values backend should not drop extra environment columns."""
    env = pd.DataFrame(
        {
            "env1": np.linspace(-1.0, 1.0, 20),
            "env2": np.linspace(1.0, -1.0, 20),
        },
        index=[f"cell{i}" for i in range(20)],
    )

    transformed = cygnet.transform_environment(env, method="values")

    pd.testing.assert_frame_equal(transformed, env)


def test_statsmodels_bspline_transform_supports_multiple_columns():
    """The statsmodels backend should build a basis for every environment column."""
    env = pd.DataFrame(
        {
            "env1": np.linspace(-1.0, 1.0, 20),
            "env2": np.linspace(1.0, -1.0, 20),
        },
        index=[f"cell{i}" for i in range(20)],
    )

    transformed = cygnet.transform_environment(env, method="statsmodels", k=6)

    assert transformed.shape == (20, 12)
    assert transformed.index.equals(env.index)
    assert transformed.columns.tolist() == [
        *[f"env1_bspline_{i}" for i in range(1, 7)],
        *[f"env2_bspline_{i}" for i in range(1, 7)],
    ]


def test_mgcv_tp_transform_shape_when_r_mgcv_is_available():
    """The default nonlinear backend should call R mgcv and return its tp basis."""
    env = pd.DataFrame({"env1": np.linspace(-1.0, 1.0, 20)}, index=[f"cell{i}" for i in range(20)])

    try:
        transformed = cygnet.transform_environment(env, method="mgcv", k=10)
    except RuntimeError as exc:
        pytest.skip(f"R mgcv is not available: {exc}")

    assert transformed.shape == (20, 10)
    assert transformed.index.equals(env.index)
    assert transformed.columns.tolist() == [f"env1_mgcv_{i}" for i in range(1, 11)]


def test_mgcv_tp_transform_supports_multiple_columns_when_r_mgcv_is_available():
    """Multiple environment columns should be passed to one multivariate mgcv smooth."""
    env = pd.DataFrame(
        {
            "env1": np.linspace(-1.0, 1.0, 30),
            "env2": np.sin(np.linspace(-1.0, 1.0, 30)),
        },
        index=[f"cell{i}" for i in range(30)],
    )

    try:
        transformed = cygnet.transform_environment(env, method="mgcv", k=10)
    except RuntimeError as exc:
        pytest.skip(f"R mgcv is not available: {exc}")

    assert transformed.shape == (30, 10)
    assert transformed.index.equals(env.index)
    assert transformed.columns.tolist() == [f"env1_env2_mgcv_{i}" for i in range(1, 11)]


def test_load_simulation_data_transforms_mgcv_from_values_file(tmp_path):
    """env_type='mgcv' should read raw values_env and transform them on load."""
    simulation_dir = tmp_path / "simulation_1"
    simulation_dir.mkdir()
    cells = [f"Cell{i}" for i in range(1, 7)]

    pd.DataFrame({"x": np.arange(6), "y": np.arange(6) + 1}, index=cells).to_csv(
        simulation_dir / "sim_1_1_locations.csv"
    )
    pd.DataFrame({"celltype": ["Ct1", "Ct2", "Ct1", "Ct2", "Ct1", "Ct2"]}, index=cells).to_csv(
        simulation_dir / "sim_1_1_celltypes.csv"
    )
    pd.DataFrame({"gene1": np.arange(6.0), "gene2": np.arange(6.0) + 2}, index=cells).to_csv(
        simulation_dir / "sim_1_1_scater_normalized_counts.csv"
    )
    pd.DataFrame({"env1": np.linspace(-1.0, 1.0, 6)}, index=cells).to_csv(
        simulation_dir / "sim_1_1_values_env.csv"
    )

    _, _, _, env_values_df = load_simulation_data(
        str(simulation_dir),
        seed=1,
        normalized_type="scater",
        env_type="mgcv",
        env_transform_backend="statsmodels",
        env_spline_k=5,
    )

    assert env_values_df.shape == (6, 5)
    assert env_values_df.index.tolist() == cells
    assert env_values_df.columns.tolist() == [f"env1_bspline_{i}" for i in range(1, 6)]


def test_load_simulation_data_values_uses_first_column(tmp_path):
    """env_type='values' should retain only the first column, as before."""
    simulation_dir = tmp_path / "simulation_1"
    simulation_dir.mkdir()
    cells = [f"Cell{i}" for i in range(1, 7)]

    pd.DataFrame({"x": np.arange(6), "y": np.arange(6) + 1}, index=cells).to_csv(
        simulation_dir / "sim_1_1_locations.csv"
    )
    pd.DataFrame({"celltype": ["Ct1", "Ct2", "Ct1", "Ct2", "Ct1", "Ct2"]}, index=cells).to_csv(
        simulation_dir / "sim_1_1_celltypes.csv"
    )
    pd.DataFrame({"gene1": np.arange(6.0), "gene2": np.arange(6.0) + 2}, index=cells).to_csv(
        simulation_dir / "sim_1_1_scater_normalized_counts.csv"
    )
    env = pd.DataFrame(
        {"env1": np.linspace(-1.0, 1.0, 6), "env2": np.linspace(1.0, -1.0, 6)},
        index=cells,
    )
    env.to_csv(simulation_dir / "sim_1_1_values_env.csv")

    _, _, _, env_values_df = load_simulation_data(
        str(simulation_dir),
        seed=1,
        normalized_type="scater",
        env_type="values",
    )

    pd.testing.assert_frame_equal(env_values_df, env.iloc[:, [0]])


def test_load_simulation_data_reads_precomputed_environment_by_default(tmp_path):
    """A non-values env_type selects its precomputed file without transforming."""
    simulation_dir = tmp_path / "simulation_1"
    simulation_dir.mkdir()
    cells = [f"Cell{i}" for i in range(1, 7)]
    pd.DataFrame({"x": np.arange(6), "y": np.arange(6) + 1}, index=cells).to_csv(
        simulation_dir / "sim_1_1_locations.csv"
    )
    pd.DataFrame({"celltype": ["Ct1", "Ct2"] * 3}, index=cells).to_csv(
        simulation_dir / "sim_1_1_celltypes.csv"
    )
    pd.DataFrame({"gene1": np.arange(6.0)}, index=cells).to_csv(
        simulation_dir / "sim_1_1_scater_normalized_counts.csv"
    )
    expected = pd.DataFrame(
        {"mgcv_1": np.arange(6.0), "mgcv_2": np.arange(6.0) ** 2},
        index=cells,
    )
    expected.to_csv(simulation_dir / "sim_1_1_mgcv_env.csv")

    _, _, _, observed = load_simulation_data(
        str(simulation_dir), seed=1, normalized_type="scater", env_type="mgcv"
    )
    pd.testing.assert_frame_equal(observed, expected)


def test_default_simulation_loader_matches_manuscript_loader(tmp_path):
    from cygnet import _manuscript_utils

    simulation_dir = tmp_path / "simulation_1"
    simulation_dir.mkdir()
    cells = [f"Cell{i}" for i in range(1, 7)]
    pd.DataFrame({"x": np.arange(6), "y": np.arange(6) + 1}, index=cells).to_csv(
        simulation_dir / "sim_1_1_locations.csv"
    )
    pd.DataFrame({"celltype": ["Ct1", "Ct2"] * 3}, index=cells).to_csv(
        simulation_dir / "sim_1_1_celltypes.csv"
    )
    pd.DataFrame({"gene1": np.arange(6.0)}, index=cells).to_csv(
        simulation_dir / "sim_1_1_scater_normalized_counts.csv"
    )
    pd.DataFrame(
        {"env1": np.arange(6.0), "unused": np.arange(6.0) ** 2}, index=cells
    ).to_csv(simulation_dir / "sim_1_1_values_env.csv")

    actual = load_simulation_data(
        str(simulation_dir), seed=1, normalized_type="scater", env_type="values"
    )
    expected = _manuscript_utils.load_simulation_data(
        str(simulation_dir), seed=1, normalized_type="scater", env_type="values"
    )
    for actual_frame, expected_frame in zip(actual, expected):
        pd.testing.assert_frame_equal(actual_frame, expected_frame)

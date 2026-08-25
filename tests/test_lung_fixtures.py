"""Smoke tests for lung AnnData tutorial fixtures."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

anndata = pytest.importorskip("anndata")

import cygnet


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "docs" / "tutorial_data"
PACKAGE_DIR = Path(__file__).resolve().parents[1]


def test_lung_tutorial_fixtures_run_pipeline_smoke():
    """The packaged lung fixtures should be usable by the AnnData pipeline."""
    spot_path = FIXTURE_DIR / "lung_visium_spot_fixture.h5ad"
    single_path = FIXTURE_DIR / "lung_xenium_single_cell_fixture.h5ad"
    if not spot_path.exists() or not single_path.exists():
        pytest.skip("Lung tutorial h5ad fixtures have not been built.")

    spot = anndata.read_h5ad(spot_path)
    single = anndata.read_h5ad(single_path)

    assert spot.uns["cygnet_fixture"]["metadata_source"] == "real lung RData workspace"
    assert single.uns["cygnet_fixture"]["metadata_source"] == "real lung RData workspace"
    assert spot.uns["cygnet_fixture"]["expression_source"] == "real_normalized_counts_df_from_RData"
    assert single.uns["cygnet_fixture"]["expression_source"] == "real_normalized_counts_df_from_RData"

    spot_celltypes = [column for column in spot.obs.columns if column != "pollutant"]
    spot_result = cygnet.run_cygnet_pipeline_from_anndata(
        spot,
        genes=["MSR1"],
        environment="pollutant",
        celltypes=spot_celltypes,
        celltype_names=["Macrophage"],
        n_permutations=1,
        n_jobs=1,
        n_components=3,
        maxiter=2,
        show_progress=False,
    )
    single_result = cygnet.run_cygnet_pipeline_from_anndata(
        single,
        genes=["MSR1"],
        environment="pollutant",
        celltypes="celltype_label",
        celltype_names=["Macrophage"],
        n_permutations=1,
        n_jobs=1,
        n_components=3,
        maxiter=2,
        show_progress=False,
    )

    assert spot_result.results.shape[0] == 1
    assert single_result.results.shape[0] == 1
    assert spot_result.results["Used_Permutation"].all()
    assert single_result.results["Used_Permutation"].all()


def test_lung_fixture_tutorial_script_runs(tmp_path):
    """The executable lung fixture tutorial should write standard output files."""
    spot_path = FIXTURE_DIR / "lung_visium_spot_fixture.h5ad"
    single_path = FIXTURE_DIR / "lung_xenium_single_cell_fixture.h5ad"
    if not spot_path.exists() or not single_path.exists():
        pytest.skip("Lung tutorial h5ad fixtures have not been built.")

    script = PACKAGE_DIR / "examples" / "run_lung_fixture_pipeline.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-dir",
            str(FIXTURE_DIR),
            "--output-dir",
            str(tmp_path),
            "--n-jobs",
            "1",
            "--n-permutations",
            "1",
            "--hide-progress",
        ],
        cwd=PACKAGE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "lung_visium_spot_results.csv").exists()
    assert (tmp_path / "lung_xenium_single_cell_results.csv").exists()

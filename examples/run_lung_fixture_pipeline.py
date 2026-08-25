"""Run the CYGNET pipeline on the packaged lung AnnData fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

import cygnet


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PACKAGE_ROOT / "docs" / "tutorial_data"
DEFAULT_OUTPUT = PACKAGE_ROOT / "docs" / "tutorial_outputs"


def _run_spot_fixture(data_dir, output_dir, n_jobs, n_permutations, show_progress):
    adata = ad.read_h5ad(data_dir / "lung_visium_spot_fixture.h5ad")
    celltype_cols = [column for column in adata.obs.columns if column != "pollutant"]
    result = cygnet.run_cygnet_pipeline_from_anndata(
        adata,
        genes=["MSR1", "UBD"],
        environment="pollutant",
        celltypes=celltype_cols,
        celltype_names=["Macrophage"],
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        n_components=4,
        maxiter=3,
        show_progress=show_progress,
        store_key="cygnet",
    )
    paths = cygnet.save_cygnet_results(result, output_dir, prefix="lung_visium_spot")
    return result, paths


def _run_single_fixture(data_dir, output_dir, n_jobs, n_permutations, show_progress):
    adata = ad.read_h5ad(data_dir / "lung_xenium_single_cell_fixture.h5ad")
    result = cygnet.run_cygnet_pipeline_from_anndata(
        adata,
        genes=["MSR1", "CHIT1"],
        environment="pollutant",
        celltypes="celltype_label",
        celltype_names=["Macrophage"],
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        n_components=4,
        maxiter=3,
        show_progress=show_progress,
        store_key="cygnet",
    )
    paths = cygnet.save_cygnet_results(result, output_dir, prefix="lung_xenium_single_cell")
    return result, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--n-permutations", type=int, default=1)
    parser.add_argument("--hide-progress", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spot_result, spot_paths = _run_spot_fixture(
        args.data_dir,
        args.output_dir,
        args.n_jobs,
        args.n_permutations,
        not args.hide_progress,
    )
    single_result, single_paths = _run_single_fixture(
        args.data_dir,
        args.output_dir,
        args.n_jobs,
        args.n_permutations,
        not args.hide_progress,
    )

    print("spot", spot_result.results.shape, spot_paths["results"])
    print("single_cell", single_result.results.shape, single_paths["results"])


if __name__ == "__main__":
    main()

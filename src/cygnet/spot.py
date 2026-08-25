"""Spot-level preprocessing helpers for CYGNET."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from .core import _resolve_rscript, celltype_clr_transform_from_df


_SPOTCLEAN_R_PACKAGES = ("SpatialExperiment", "SpotClean", "Seurat", "sctransform")

_SPOTCLEAN_R_CODE = r"""
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7) {
    stop("Expected arguments: raw_feature_bc_matrix spatial_dir output_dir min_cells min_features variable_features_rv_th save_rds")
}

data_dir <- args[[1]]
spatial_dir <- args[[2]]
output_dir <- args[[3]]
min_cells <- as.integer(args[[4]])
min_features <- as.integer(args[[5]])
variable_features_rv_th <- as.numeric(args[[6]])
save_rds <- as.logical(as.integer(args[[7]]))

required <- c("SpatialExperiment", "SpotClean", "Seurat", "sctransform")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
    stop("Missing required R packages: ", paste(missing, collapse = ", "))
}

suppressPackageStartupMessages({
    library(SpatialExperiment)
    library(SpotClean)
    library(Seurat)
    library(sctransform)
})

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

data_raw <- read10xRaw(data_dir)
slide_info <- read10xSlide(
    tissue_csv_file = file.path(spatial_dir, "tissue_positions.csv"),
    tissue_img_file = file.path(spatial_dir, "tissue_lowres_image.png"),
    scale_factor_file = file.path(spatial_dir, "scalefactors_json.json")
)
slide_obj <- createSlide(data_raw, slide_info)
decont_obj <- spotclean(slide_obj)

decont_counts <- decont_obj@assays@data@listData$decont
write.csv(
    as.matrix(decont_counts),
    file = file.path(output_dir, "raw_counts_spotclean.csv"),
    row.names = TRUE
)

seurat_obj <- convertToSeurat(decont_obj, image_dir = spatial_dir)
seurat_obj <- SCTransform(
    seurat_obj,
    return.only.var.genes = FALSE,
    variable.features.n = NULL,
    variable.features.rv.th = variable_features_rv_th
)
sct_scaled <- tryCatch(
    Seurat::GetAssayData(seurat_obj, assay = "SCT", layer = "scale.data"),
    error = function(e) Seurat::GetAssayData(seurat_obj, assay = "SCT", slot = "scale.data")
)
write.csv(
    as.matrix(sct_scaled),
    file = file.path(output_dir, "SCTscaled_counts_spotclean.csv"),
    row.names = TRUE
)

if (save_rds) {
    saveRDS(decont_obj, file.path(output_dir, "decont_obj.rds"))
    saveRDS(seurat_obj, file.path(output_dir, "seurat_obj.rds"))
}
"""


def _ensure_dataframe(value, name):
    """Validate a spot preprocessing input as a pandas DataFrame."""
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if value.index.has_duplicates:
        raise ValueError(f"{name} index contains duplicate spot IDs.")
    return value


def _align_to_reference(reference, frames, *, only_common=False):
    """Align spot-level frames to a reference index."""
    if only_common:
        common_index = reference.index
        for frame in frames:
            common_index = common_index.intersection(frame.index)
        if common_index.empty:
            raise ValueError("There are no common spot IDs among the input DataFrames.")
        return [frame.loc[common_index] for frame in frames]

    reference_set = set(reference.index)
    aligned = []
    for frame in frames:
        frame_set = set(frame.index)
        missing = reference_set - frame_set
        extra = frame_set - reference_set
        if missing or extra:
            raise ValueError(
                "Spot-level DataFrames must contain the same spot IDs. "
                f"Missing: {len(missing)}; extra: {len(extra)}."
            )
        aligned.append(frame.loc[reference.index])
    return aligned


def prepare_spot_mode_inputs(
    locations_df,
    celltype_df,
    normalized_counts_df,
    env_values_df,
    *,
    apply_clr=True,
    clr_drop_column="auto",
    only_common=False,
    clr_output_file=None,
):
    """Align spot-level matrices and apply CLR to the cell-type composition matrix.

    Parameters
    ----------
    locations_df, celltype_df, normalized_counts_df, env_values_df : pandas.DataFrame
        Spot-indexed input matrices. ``celltype_df`` should contain non-negative
        cell-type proportions or counts with one column per cell type.
    apply_clr : bool, default True
        Whether to apply the standard CLR transform to ``celltype_df``.
    clr_drop_column : str, "auto", or None, default "auto"
        Which CLR-transformed cell-type column to remove before modeling.
        ``"auto"`` removes the cell type with the smallest original column sum.
        None keeps the full CLR matrix.
    only_common : bool, default False
        If True, keep only spot IDs shared by every input. Otherwise every input
        must contain the exact same spot IDs as ``locations_df``.
    clr_output_file : str or path-like, optional
        Optional CSV path for the CLR-transformed cell-type matrix.

    Returns
    -------
    tuple
        ``(locations_df, celltype_df, normalized_counts_df, env_values_df)`` with
        consistent row order. The returned ``celltype_df`` is CLR-transformed
        and has one CLR column removed by default.
    """
    locations_df = _ensure_dataframe(locations_df, "locations_df")
    celltype_df = _ensure_dataframe(celltype_df, "celltype_df")
    normalized_counts_df = _ensure_dataframe(normalized_counts_df, "normalized_counts_df")
    env_values_df = _ensure_dataframe(env_values_df, "env_values_df")

    locations_df, celltype_df, normalized_counts_df, env_values_df = _align_to_reference(
        locations_df,
        [locations_df, celltype_df, normalized_counts_df, env_values_df],
        only_common=only_common,
    )

    if apply_clr:
        celltype_df = celltype_clr_transform_from_df(
            celltype_df,
            output_file=clr_output_file,
            drop_column=clr_drop_column,
        )

    return locations_df, celltype_df, normalized_counts_df, env_values_df


def missing_spotclean_r_packages(rscript=None, packages=_SPOTCLEAN_R_PACKAGES):
    """Return the R packages required for SpotClean that are not importable."""
    rscript_path = _resolve_rscript(rscript)
    quoted = ", ".join(f'"{package}"' for package in packages)
    code = (
        f"packages <- c({quoted}); "
        "missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]; "
        "if (length(missing) > 0) { cat(paste(missing, collapse = '\\n')); quit(status = 1) }"
    )
    result = subprocess.run(
        [rscript_path, "-e", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return []
    missing = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if missing:
        return missing
    return list(packages)


def spotclean_available(rscript=None):
    """Return True when Rscript can load all SpotClean workflow packages."""
    return not missing_spotclean_r_packages(rscript=rscript)


def run_spotclean_10x(
    raw_feature_bc_matrix,
    spatial_dir,
    *,
    output_dir=None,
    rscript=None,
    min_cells=100,
    min_features=20,
    variable_features_rv_th=1.3,
    save_rds=True,
):
    """Run the standard R SpotClean workflow for a 10x Visium dataset.

    The wrapper writes ``raw_counts_spotclean.csv`` and
    ``SCTscaled_counts_spotclean.csv`` to ``output_dir``. When ``save_rds`` is
    True, it also writes ``decont_obj.rds`` and ``seurat_obj.rds``.
    """
    raw_dir = Path(raw_feature_bc_matrix)
    spatial_path = Path(spatial_dir)
    output_path = Path(output_dir) if output_dir is not None else raw_dir.parent

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_feature_bc_matrix does not exist: {raw_dir}")
    if not spatial_path.exists():
        raise FileNotFoundError(f"spatial_dir does not exist: {spatial_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    rscript_path = _resolve_rscript(rscript)
    command = [
        rscript_path,
        "-e",
        _SPOTCLEAN_R_CODE,
        str(raw_dir),
        str(spatial_path),
        str(output_path),
        str(int(min_cells)),
        str(int(min_features)),
        str(float(variable_features_rv_th)),
        "1" if save_rds else "0",
    ]

    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"R SpotClean failed: {message}") from exc

    outputs = {
        "raw_counts": output_path / "raw_counts_spotclean.csv",
        "sct_scaled_counts": output_path / "SCTscaled_counts_spotclean.csv",
    }
    if save_rds:
        outputs["decont_rds"] = output_path / "decont_obj.rds"
        outputs["seurat_rds"] = output_path / "seurat_obj.rds"
    return outputs

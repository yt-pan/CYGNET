required <- c("SpotClean", "SpatialExperiment", "Seurat", "sctransform")

if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
    BiocManager::install(missing, ask = FALSE, update = FALSE)
}

still_missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(still_missing) > 0) {
    stop("Missing required R packages after installation: ", paste(still_missing, collapse = ", "))
}

message("SpotClean workflow R packages are available: ", paste(required, collapse = ", "))

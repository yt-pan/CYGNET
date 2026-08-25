"""Generate synthetic preview figures for CYGNET real-data plotting helpers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

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
    save_figure,
)


def main():
    rng = np.random.default_rng(20260603)
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "plot_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 120
    locations = pd.DataFrame(
        {
            "x": rng.uniform(0, 12, n),
            "y": rng.uniform(0, 8, n),
        }
    )
    x = locations["x"].to_numpy()
    y = locations["y"].to_numpy()
    env = pd.DataFrame(
        {
            "pollutant": np.exp(-((x - 3.0) ** 2 + (y - 5.5) ** 2) / 18.0)
            + 0.20 * np.sin(x / 1.7)
            + rng.normal(0, 0.035, n)
        }
    )
    raw_celltypes = rng.gamma(
        shape=np.column_stack(
            [
                1.4 + 3.2 * env["pollutant"].to_numpy(),
                2.4 - 1.0 * env["pollutant"].to_numpy(),
                np.full(n, 1.2),
            ]
        ),
        scale=1.0,
    )
    celltypes = pd.DataFrame(raw_celltypes / raw_celltypes.sum(axis=1, keepdims=True), columns=["Macrophage", "Tumor", "Stroma"])
    celltype_labels = celltypes.idxmax(axis=1).rename("celltype")
    counts = pd.DataFrame(
        {
            "MSR1": 1.2
            + 1.7 * celltypes["Macrophage"].to_numpy()
            + 1.1 * env["pollutant"].to_numpy()
            + 1.8 * celltypes["Macrophage"].to_numpy() * env["pollutant"].to_numpy()
            + rng.normal(0, 0.18, n)
        }
    )

    ax = plot_spatial_values(locations, env, "pollutant", title="Imported environment values")
    save_figure(ax.figure, out_dir / "spatial_env_values.png", close=True)

    null_p = rng.uniform(0, 1, 900)
    shifted_p = np.concatenate([rng.uniform(0, 0.03, 45), rng.uniform(0, 1, 855)])
    ax = plot_qq([null_p, shifted_p], labels=["Null-like", "Signal-enriched"], title="P-value calibration")
    save_figure(ax.figure, out_dir / "qq_plot.png", close=True)

    ax = plot_celltype_map(locations, celltype_labels, title="Cell type map")
    save_figure(ax.figure, out_dir / "celltype_map.png", close=True)

    gx, gy = np.meshgrid(np.arange(8), np.arange(5))
    pie_locations = pd.DataFrame({"x": gx.ravel(), "y": gy.ravel()})
    pie_signal = np.linspace(0.05, 0.95, pie_locations.shape[0])
    pie_raw = np.column_stack(
        [
            0.6 + 2.5 * pie_signal,
            1.7 - 0.8 * pie_signal,
            0.7 + 0.5 * np.sin(pie_locations["x"].to_numpy()),
        ]
    )
    pie_celltypes = pd.DataFrame(pie_raw / pie_raw.sum(axis=1, keepdims=True), columns=["Macrophage", "Tumor", "Stroma"])
    ax = plot_spot_celltype_pies(
        pie_locations,
        pie_celltypes,
        radius=0.42,
        title="Spot-level cell-type composition",
    )
    save_figure(ax.figure, out_dir / "spot_celltype_pies.png", close=True)

    ax = plot_celltype_env_overlay(
        locations,
        celltypes[["Macrophage"]],
        env,
        celltype_col="Macrophage",
        env_col="pollutant",
        celltype_labels=celltype_labels,
        title="Macrophage abundance over pollutant gradient",
    )
    save_figure(ax.figure, out_dir / "celltype_env_overlay.png", close=True)

    axes = plot_feature_pdp(counts, celltypes, env, gene="MSR1", celltype="Macrophage", env="pollutant", n_neighbors=24)
    axes[0].figure.suptitle("KNN PDP for MSR1", y=1.02)
    save_figure(axes[0].figure, out_dir / "feature_pdp.png", close=True)

    inter = (celltypes[["Macrophage"]].to_numpy() * env[["pollutant"]].to_numpy())
    env_component = env.to_numpy()
    spatial_component = (locations.to_numpy() - locations.to_numpy().mean(axis=0)) / locations.to_numpy().std(axis=0)
    pve = cygnet.calculate_pve(
        [0.8, 0.5, 0.015, 0.35],
        [inter, env_component, spatial_component],
        effect_names=["Celltype x Env", "Env", "Spatial"],
    )
    ax = plot_pve(pve, title="Trace-aware PVE")
    save_figure(ax.figure, out_dir / "trace_aware_pve.png", close=True)


if __name__ == "__main__":
    main()

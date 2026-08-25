"""Real-data plotting utilities for CYGNET inputs and fitted results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist


_CELLTYPE_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#B07AA1",
    "#9C755F",
    "#BAB0AC",
    "#2F4B7C",
    "#A05195",
    "#665191",
    "#FFA600",
]


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _lung_env_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "cygnet_lung_env",
        ["#f8f5ef", "#ead8c7", "#d7a98f", "#b97868", "#6f4c5b"],
    )


def _resolve_cmap(cmap):
    if cmap in {"lung", "lung_pollutant", "cygnet_lung"}:
        return _lung_env_cmap()
    return cmap


def save_figure(fig, output_file, dpi=300, close=False, **kwargs):
    """Save a matplotlib figure and create parent directories when needed."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", **kwargs)
    if close:
        _plt().close(fig)
    return output_file


def _coordinate_array(locations, x_col=None, y_col=None):
    loc = pd.DataFrame(locations)
    if x_col is None or y_col is None:
        if loc.shape[1] < 2:
            raise ValueError("locations must contain at least two coordinate columns.")
        x_col = loc.columns[0] if x_col is None else x_col
        y_col = loc.columns[1] if y_col is None else y_col
    return loc[[x_col, y_col]].to_numpy(dtype=float), x_col, y_col


def _value_vector(values, value_col=None, index=None):
    if isinstance(values, pd.Series):
        series = values
    else:
        frame = pd.DataFrame(values)
        if value_col is None:
            if frame.shape[1] != 1:
                raise ValueError("value_col is required when values has multiple columns.")
            value_col = frame.columns[0]
        series = frame[value_col]
    if index is not None and isinstance(series.index, pd.Index):
        series = series.reindex(index)
    return series.to_numpy(dtype=float)


def _celltype_labels(celltypes, celltype_col=None, index=None):
    if isinstance(celltypes, pd.Series):
        labels = celltypes
    else:
        frame = pd.DataFrame(celltypes)
        if celltype_col is not None:
            labels = frame[celltype_col]
        elif frame.shape[1] == 1:
            numeric = frame.apply(pd.to_numeric, errors="coerce")
            if numeric.notna().all(axis=None) and _is_one_hot_matrix(numeric):
                labels = pd.Series(frame.columns[0], index=frame.index, dtype="object")
                labels[numeric.iloc[:, 0] <= 0] = "Unknown"
            else:
                labels = frame.iloc[:, 0]
        else:
            numeric = frame.apply(pd.to_numeric, errors="coerce")
            if numeric.notna().all(axis=None):
                labels = numeric.idxmax(axis=1).astype("object")
                labels[numeric.max(axis=1) <= 0] = "Unknown"
            else:
                raise ValueError("celltype_col is required when celltypes has multiple non-numeric columns.")
    if index is not None and isinstance(labels.index, pd.Index):
        labels = labels.reindex(index)
    return labels.astype("object").fillna("Unknown")


def _celltype_color_map(labels, colors=None):
    levels = pd.Index(pd.Series(labels).dropna().astype(str).unique())
    if colors is None:
        return {level: _CELLTYPE_COLORS[i % len(_CELLTYPE_COLORS)] for i, level in enumerate(levels)}
    if isinstance(colors, dict):
        return {level: colors.get(level, _CELLTYPE_COLORS[i % len(_CELLTYPE_COLORS)]) for i, level in enumerate(levels)}
    return {level: colors[i % len(colors)] for i, level in enumerate(levels)}


def _is_one_hot_matrix(frame, atol=1e-8):
    numeric = pd.DataFrame(frame).apply(pd.to_numeric, errors="coerce")
    if not numeric.notna().all(axis=None):
        return False
    values = numeric.to_numpy(dtype=float)
    if values.size == 0:
        return False
    binary = np.isclose(values, 0.0, atol=atol) | np.isclose(values, 1.0, atol=atol)
    if not bool(binary.all()):
        return False
    row_sums = values.sum(axis=1)
    return bool(np.all(np.isclose(row_sums, 1.0, atol=atol) | np.isclose(row_sums, 0.0, atol=atol)))


def _infer_celltype_mode(celltypes, mode="auto", celltype_col=None):
    if mode not in {"auto", "single", "spot", "abundance"}:
        raise ValueError("mode must be one of 'auto', 'single', 'spot', or 'abundance'.")
    if mode != "auto":
        return mode
    if isinstance(celltypes, pd.Series) or celltype_col is not None:
        return "single"
    frame = pd.DataFrame(celltypes)
    if frame.shape[1] == 1:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        if numeric.notna().all(axis=None):
            return "single" if _is_one_hot_matrix(numeric) else "spot"
        return "single"
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.notna().all(axis=None):
        return "single" if _is_one_hot_matrix(numeric) else "spot"
    raise ValueError("celltype_col is required when celltypes has multiple non-numeric columns.")


def _infer_overlay_celltype_mode(celltypes, mode="auto", celltype_col=None, celltype_labels=None):
    if celltype_labels is not None:
        return "single"
    if mode != "auto":
        return _infer_celltype_mode(celltypes, mode=mode, celltype_col=celltype_col)
    if celltype_col is not None:
        return "abundance"
    if isinstance(celltypes, pd.Series):
        numeric = pd.to_numeric(celltypes, errors="coerce")
        return "abundance" if numeric.notna().all() else "single"
    frame = pd.DataFrame(celltypes)
    if frame.shape[1] == 1:
        numeric = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
        return "abundance" if numeric.notna().all() else "single"
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.notna().all(axis=None):
        return "single" if _is_one_hot_matrix(numeric) else "spot"
    raise ValueError("celltype_col is required when celltype_values has multiple non-numeric columns.")


def _composition_frame(celltypes, celltype_columns=None, index=None):
    comp = pd.DataFrame(celltypes)
    if index is not None and isinstance(comp.index, pd.Index):
        comp = comp.reindex(index)
    if celltype_columns is None:
        celltype_columns = list(comp.columns)
    comp = comp.loc[:, celltype_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    row_sums = comp.sum(axis=1).replace(0, np.nan)
    return comp.div(row_sums, axis=0).fillna(0.0)


def _pie_color_map(celltypes, colors=None):
    plt = _plt()
    if colors is None:
        palette = plt.get_cmap("tab20")
        return {celltype: palette(i % 20) for i, celltype in enumerate(celltypes)}
    if isinstance(colors, dict):
        return {celltype: colors.get(celltype, _CELLTYPE_COLORS[i % len(_CELLTYPE_COLORS)]) for i, celltype in enumerate(celltypes)}
    return {celltype: colors[i % len(colors)] for i, celltype in enumerate(celltypes)}


def _default_pie_radius(coords):
    distances = pdist(coords)
    distances = distances[np.isfinite(distances) & (distances > 0)]
    return 0.35 * (np.median(distances) if distances.size else 1.0)


def _draw_celltype_scatter(ax, coords, labels, colors=None, point_size=16, alpha=0.9, edgecolors="none", linewidth=0.0):
    color_map = _celltype_color_map(labels, colors=colors)
    label_values = labels.astype(str).to_numpy()
    for label in color_map:
        mask = label_values == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            color=color_map[label],
            label=label,
            alpha=alpha,
            edgecolors=edgecolors,
            linewidth=linewidth,
            zorder=3,
        )
    return color_map


def _draw_spot_pies(ax, coords, comp, radius=None, colors=None):
    from matplotlib.patches import Wedge

    celltypes = list(comp.columns)
    radius = _default_pie_radius(coords) if radius is None else radius
    edge_linewidth = min(0.25, max(0.015, radius * 0.15))
    color_map = _pie_color_map(celltypes, colors=colors)
    for (x, y), (_, row) in zip(coords, comp.iterrows()):
        start = 90.0
        for celltype, fraction in row.items():
            if fraction <= 0:
                continue
            end = start + 360.0 * float(fraction)
            ax.add_patch(
                Wedge(
                    (x, y),
                    radius,
                    start,
                    end,
                    facecolor=color_map[celltype],
                    edgecolor="white",
                    linewidth=edge_linewidth,
                    zorder=3,
                )
            )
            start = end
    return color_map, radius


def _make_legend_markers_readable(legend, marker_size=80):
    if legend is None:
        return
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", [])
    for handle in handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([marker_size])
        if hasattr(handle, "set_markersize"):
            handle.set_markersize(np.sqrt(marker_size))


def _clean_p_values(p_values):
    values = np.asarray(p_values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[(values >= 0) & (values <= 1)]
    if values.size == 0:
        raise ValueError("p_values must contain at least one finite value in [0, 1].")
    return np.clip(values, np.finfo(float).tiny, 1.0)


def plot_qq(p_values, labels=None, ax=None, title=None, point_size=10, show_ci=True):
    """Create a p-value QQ plot with optional beta order-statistic envelopes."""
    plt = _plt()
    if not isinstance(p_values, (list, tuple)) or (len(p_values) > 0 and np.ndim(p_values[0]) == 0):
        p_values = [p_values]
    if labels is None:
        labels = [f"Set {i + 1}" for i in range(len(p_values))]
    if len(labels) != len(p_values):
        raise ValueError("labels must have the same length as p_values.")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    max_x = 0.0
    max_y = 0.0
    for values, label in zip(p_values, labels):
        values = np.sort(_clean_p_values(values))
        n = values.size
        rank = np.arange(1, n + 1)
        expected = rank / (n + 1)
        x = -np.log10(expected)
        y = -np.log10(values)
        max_x = max(max_x, float(np.nanmax(x)))
        max_y = max(max_y, float(np.nanmax(y)))
        if show_ci:
            upper = -np.log10(stats.beta.ppf(0.025, rank, n - rank + 1))
            lower = -np.log10(stats.beta.ppf(0.975, rank, n - rank + 1))
            ax.plot(x, upper, color="0.65", linewidth=0.8, alpha=0.6)
            ax.plot(x, lower, color="0.65", linewidth=0.8, alpha=0.6)
        ax.scatter(x, y, s=point_size, alpha=0.65, label=label)

    lim = max(1.0, max(max_x, max_y))
    ax.plot([0, lim], [0, lim], color="#C44E52", linewidth=1)
    ax.set_xlim(0, lim * 1.02)
    ax.set_ylim(0, lim * 1.02)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_spatial_values(
    locations,
    values,
    value_col=None,
    x_col=None,
    y_col=None,
    ax=None,
    cmap="viridis",
    point_size=18,
    colorbar=True,
    colorbar_label=None,
    title=None,
):
    """Plot imported cell-type, environment, or expression values over space."""
    plt = _plt()
    loc = pd.DataFrame(locations)
    coords, x_col, y_col = _coordinate_array(loc, x_col=x_col, y_col=y_col)
    value = _value_vector(values, value_col=value_col, index=loc.index)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=value, s=point_size, cmap=_resolve_cmap(cmap), edgecolors="none")
    if colorbar:
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    ax.set_xlabel(str(x_col))
    ax.set_ylabel(str(y_col))
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_celltype_map(
    locations,
    celltypes,
    celltype_col=None,
    x_col=None,
    y_col=None,
    ax=None,
    colors=None,
    point_size=16,
    alpha=0.9,
    legend=True,
    title=None,
    mode="auto",
    spot_celltypes=None,
    spot_radius=None,
):
    """Plot cell types over spatial coordinates.

    With ``mode="auto"``, categorical labels and one-hot matrices are drawn as
    one point per cell. Continuous composition matrices are drawn as spot-level
    pie charts.
    """
    plt = _plt()
    loc = pd.DataFrame(locations)
    coords, x_col, y_col = _coordinate_array(loc, x_col=x_col, y_col=y_col)
    mode = _infer_celltype_mode(celltypes, mode=mode, celltype_col=celltype_col)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))

    if mode == "single":
        labels = _celltype_labels(celltypes, celltype_col=celltype_col, index=loc.index).astype(str)
        color_map = _draw_celltype_scatter(
            ax,
            coords,
            labels,
            colors=colors,
            point_size=point_size,
            alpha=alpha,
        )
        if legend:
            legend_obj = ax.legend(frameon=False, fontsize=8, markerscale=1.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
            _make_legend_markers_readable(legend_obj)
    elif mode == "spot":
        comp = _composition_frame(celltypes, celltype_columns=spot_celltypes, index=loc.index)
        color_map, radius = _draw_spot_pies(ax, coords, comp, radius=spot_radius, colors=colors)
        ax.set_xlim(coords[:, 0].min() - radius, coords[:, 0].max() + radius)
        ax.set_ylim(coords[:, 1].min() - radius, coords[:, 1].max() + radius)
        if legend:
            handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[c], label=c) for c in comp.columns]
            legend_obj = ax.legend(handles=handles, frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
            _make_legend_markers_readable(legend_obj)
    else:
        raise ValueError("plot_celltype_map supports mode='auto', 'single', or 'spot'.")

    ax.set_xlabel(str(x_col))
    ax.set_ylabel(str(y_col))
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_spot_celltype_pies(
    locations,
    celltype_df,
    celltypes=None,
    x_col=None,
    y_col=None,
    radius=None,
    colors=None,
    ax=None,
    legend=True,
    title=None,
):
    """Draw each spot as a pie chart showing cell-type composition.

    This is kept as a compatibility wrapper. New code can call
    :func:`plot_celltype_map` with ``mode="spot"`` or ``mode="auto"``.
    """
    return plot_celltype_map(
        locations,
        celltype_df,
        x_col=x_col,
        y_col=y_col,
        ax=ax,
        colors=colors,
        legend=legend,
        title=title,
        mode="spot",
        spot_celltypes=celltypes,
        spot_radius=radius,
    )


def _interpolate_grid(coords, values, grid_size=150, method="linear", max_distance="auto"):
    x = coords[:, 0]
    y = coords[:, 1]
    xi = np.linspace(np.nanmin(x), np.nanmax(x), grid_size)
    yi = np.linspace(np.nanmin(y), np.nanmax(y), grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)
    valid = np.isfinite(values)
    grid = griddata(coords[valid], values[valid], (grid_x, grid_y), method=method)
    if max_distance is not None:
        tree = cKDTree(coords[valid])
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        nearest_distance = tree.query(grid_points, k=1)[0].reshape(grid.shape)
        if max_distance == "auto":
            if coords[valid].shape[0] > 1:
                local_spacing = tree.query(coords[valid], k=2)[0][:, 1]
                spacing = float(np.nanpercentile(local_spacing, 95))
            else:
                spacing = 0.0
            grid_spacing = float(np.hypot(xi[1] - xi[0], yi[1] - yi[0])) if len(xi) > 1 and len(yi) > 1 else 0.0
            distance_limit = max(spacing * 2.5, grid_spacing * 2.0)
        else:
            distance_limit = float(max_distance)
        grid = np.where(nearest_distance <= distance_limit, grid, np.nan)
    return grid_x, grid_y, grid


def plot_celltype_env_overlay(
    locations,
    celltype_values,
    env_values,
    celltype_col=None,
    env_col=None,
    x_col=None,
    y_col=None,
    grid_size=150,
    interpolation="linear",
    ax=None,
    cmap="lung",
    contour=True,
    contour_linewidth=0.55,
    contour_alpha=0.42,
    celltype_labels=None,
    celltype_label_col=None,
    celltype_colors=None,
    show_abundance_points=True,
    point_color="#1f2937",
    point_scale=80,
    point_min=8,
    alpha=0.86,
    env_alpha_by_value=True,
    env_min_alpha=0.05,
    env_max_alpha=0.72,
    max_interpolation_distance="auto",
    colorbar=True,
    legend=True,
    title=None,
    celltype_mode="auto",
    spot_celltypes=None,
    spot_radius=None,
):
    """Plot cell types and an interpolated environment gradient.

    With ``celltype_mode="auto"``, categorical labels and one-hot matrices are
    drawn as one point per cell, continuous composition matrices are drawn as
    spot-level pie charts, and a single numeric vector/column is drawn as
    abundance-sized points. Supplying ``celltype_labels`` forces categorical
    single-cell display.
    """
    plt = _plt()
    loc = pd.DataFrame(locations)
    coords, x_col, y_col = _coordinate_array(loc, x_col=x_col, y_col=y_col)
    celltype_mode = _infer_overlay_celltype_mode(
        celltype_values,
        mode=celltype_mode,
        celltype_col=celltype_col,
        celltype_labels=celltype_labels,
    )
    env = _value_vector(env_values, value_col=env_col, index=loc.index)
    grid_x, grid_y, grid = _interpolate_grid(
        coords,
        env,
        grid_size=grid_size,
        method=interpolation,
        max_distance=max_interpolation_distance,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))

    if env_alpha_by_value:
        finite_grid = grid[np.isfinite(grid)]
        if finite_grid.size:
            grid_min = float(np.nanmin(finite_grid))
            grid_max = float(np.nanmax(finite_grid))
            denom = max(grid_max - grid_min, np.finfo(float).eps)
            image_alpha = env_min_alpha + (env_max_alpha - env_min_alpha) * np.clip((grid - grid_min) / denom, 0, 1)
            image_alpha = np.where(np.isfinite(grid), image_alpha, 0.0)
        else:
            image_alpha = env_max_alpha
    else:
        image_alpha = env_max_alpha
    resolved_cmap = _resolve_cmap(cmap)
    if hasattr(resolved_cmap, "copy"):
        resolved_cmap = resolved_cmap.copy()
        resolved_cmap.set_bad((1, 1, 1, 0))
    image = ax.imshow(
        np.ma.masked_invalid(grid),
        extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
        origin="lower",
        cmap=resolved_cmap,
        alpha=image_alpha,
        aspect="equal",
    )
    if contour:
        ax.contour(
            grid_x,
            grid_y,
            np.ma.masked_invalid(grid),
            levels=7,
            colors="#6d5f5b",
            linewidths=contour_linewidth,
            alpha=contour_alpha,
        )

    if celltype_mode == "single":
        label_source = celltype_labels if celltype_labels is not None else celltype_values
        label_col = celltype_label_col if celltype_labels is not None else celltype_col
        labels = _celltype_labels(label_source, celltype_col=label_col, index=loc.index).astype(str)
        color_map = _draw_celltype_scatter(
            ax,
            coords,
            labels,
            colors=celltype_colors,
            point_size=point_min + point_scale * 0.25,
            alpha=alpha,
            edgecolors="white",
            linewidth=0.25,
        )
        if legend:
            legend_obj = ax.legend(
                frameon=False,
                fontsize=8,
                markerscale=1.4,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=min(len(color_map), 4),
            )
            _make_legend_markers_readable(legend_obj)
    elif celltype_mode == "spot":
        comp = _composition_frame(celltype_values, celltype_columns=spot_celltypes, index=loc.index)
        color_map, radius = _draw_spot_pies(ax, coords, comp, radius=spot_radius, colors=celltype_colors)
        if legend:
            handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[c], label=c) for c in comp.columns]
            legend_obj = ax.legend(
                handles=handles,
                frameon=False,
                fontsize=8,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=min(len(color_map), 4),
            )
            _make_legend_markers_readable(legend_obj)
        ax.set_xlim(coords[:, 0].min() - radius, coords[:, 0].max() + radius)
        ax.set_ylim(coords[:, 1].min() - radius, coords[:, 1].max() + radius)
    elif show_abundance_points:
        celltype = _value_vector(celltype_values, value_col=celltype_col, index=loc.index)
        finite_celltype = celltype[np.isfinite(celltype)]
        max_value = finite_celltype.max() if finite_celltype.size else 1.0
        sizes = point_min + point_scale * np.clip(celltype / max(max_value, np.finfo(float).eps), 0, 1)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=sizes,
            color=point_color,
            edgecolor="white",
            linewidth=0.35,
            alpha=alpha,
            zorder=3,
        )
    if colorbar:
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=env_col or "environment")
    ax.set_xlabel(str(x_col))
    ax.set_ylabel(str(y_col))
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_feature_pdp(
    normalized_counts_df,
    celltype_df,
    env_values_df,
    gene,
    celltype,
    env,
    axes=None,
    n_neighbors=50,
    grid_resolution=80,
    random_state=42,
):
    """Fit a KNN smoother and plot PDPs for cell type, environment, and interaction."""
    plt = _plt()
    from sklearn.inspection import partial_dependence
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler

    counts = pd.DataFrame(normalized_counts_df)
    celltypes = pd.DataFrame(celltype_df)
    envs = pd.DataFrame(env_values_df)
    common_index = counts.index.intersection(celltypes.index).intersection(envs.index)
    if common_index.empty:
        raise ValueError("Input matrices do not share any common index.")
    X = pd.concat([celltypes.loc[common_index], envs.loc[common_index]], axis=1)
    y = counts.loc[common_index, gene].to_numpy(dtype=float)
    y = y - np.nanmean(y)
    if celltype not in X.columns:
        raise ValueError(f"celltype column not found: {celltype}")
    if env not in X.columns:
        raise ValueError(f"environment column not found: {env}")

    max_neighbors = max(1, min(int(n_neighbors), X.shape[0]))
    model = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=max_neighbors)),
        ]
    )
    model.fit(X, y)
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes = np.atleast_1d(axes).ravel()

    pd_cell = partial_dependence(model, X, [celltype], grid_resolution=grid_resolution, kind="average")
    cell_grid = pd_cell["grid_values"][0]
    axes[0].plot(cell_grid, pd_cell["average"][0], color="#4E79A7", linewidth=2)
    axes[0].set_xlabel(celltype)
    axes[0].set_ylabel(f"{gene} partial dependence")

    pd_env = partial_dependence(model, X, [env], grid_resolution=grid_resolution, kind="average")
    env_grid = pd_env["grid_values"][0]
    axes[1].plot(env_grid, pd_env["average"][0], color="#59A14F", linewidth=2)
    axes[1].set_xlabel(env)
    axes[1].set_ylabel(f"{gene} partial dependence")

    celltype_idx = X.columns.get_loc(celltype)
    env_idx = X.columns.get_loc(env)
    pd_inter = partial_dependence(model, X, [(celltype_idx, env_idx)], grid_resolution=grid_resolution, kind="average")
    c_grid, e_grid = pd_inter["grid_values"]
    z = pd_inter["average"][0].T
    image = axes[2].imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=(c_grid.min(), c_grid.max(), e_grid.min(), e_grid.max()),
        cmap="viridis",
    )
    axes[2].set_xlabel(celltype)
    axes[2].set_ylabel(env)
    plt.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return axes


def plot_pve(pve_df, component_col="Component", pve_col="PVE", ax=None, color="#4E79A7", title=None):
    """Plot a trace-aware PVE table returned by ``cygnet.calculate_pve``."""
    plt = _plt()
    df = pd.DataFrame(pve_df).copy()
    missing = [col for col in [component_col, pve_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(df[component_col].astype(str), df[pve_col].astype(float), color=color)
    ax.set_ylabel("Proportion of variance explained")
    ax.set_xlabel("")
    ax.set_ylim(0, max(1.0, float(df[pve_col].max()) * 1.15))
    ax.tick_params(axis="x", rotation=35)
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax

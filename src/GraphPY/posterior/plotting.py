import itertools
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from GraphPY.clustering.predict import predicted_clusters, predicted_clusters_2D
from GraphPY.posterior.posterior import compute_posterior_predictive
from GraphPY.types import NodeId, Nodes, XData


def plot_posterior1D(
    nodes: dict[str, dict[str, Any]],
    x: dict[str, list[float]],
    res: dict[str, Any],
    f_post: dict[str, pd.DataFrame],
    f_true: dict[str, pd.DataFrame] | None = None,
    *,
    grid_mode: bool = False,
    figsize: tuple[int, int] = (900, 400),
) -> go.Figure | None:
    """
    Plot per-node posterior predictive (and optionally true density). If `grid_mode=True`,
    arrange nodes on a grid by level (row = level, columns = nodes at that level).

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns a single Figure in grid mode (for further customization). In non-grid mode,
        shows individual figures and returns None.
    """
    # Predicted cluster labels for coloring observation ticks
    _, pred_part_nodes, _ = predicted_clusters(nodes, x, res)

    if not grid_mode:
        # Old behavior: one figure per node
        for node in nodes:
            fig = go.Figure()
            if f_true is not None and node in f_true:
                fig.add_trace(
                    go.Scatter(
                        x=f_true[node]["x"],
                        y=f_true[node]["y"],
                        mode="lines",
                        name="True distribution",
                        line=dict(color="blue"),
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=f_post[node]["x"],
                    y=f_post[node]["y"],
                    mode="lines",
                    name="Posterior",
                    line=dict(color="red"),
                )
            )

            ymax_true = (
                float(f_true[node]["y"].max()) if (f_true is not None and node in f_true) else 0.0
            )
            ymax_post = float(f_post[node]["y"].max())
            ylim = -0.1 * max(ymax_true, ymax_post, 0.0)

            # Ensure the negative tick baseline is within axis range
            fig.add_trace(
                go.Scatter(
                    x=[f_post[node]["x"].iloc[0]],
                    y=[ylim],
                    mode="markers",
                    marker_opacity=0,
                    showlegend=False,
                )
            )

            # Colored observation ticks
            if len(np.unique(pred_part_nodes[node])) < 24:
                colors = px.colors.qualitative.Light24
                for i, obs in enumerate(x[node]):
                    fig.add_shape(
                        type="line",
                        x0=obs,
                        y0=0,
                        x1=obs,
                        y1=ylim,
                        line=dict(color=colors[int(pred_part_nodes[node][i])], width=1.5),
                    )
            else:
                print("Note: clusters not shown due to too high number of atoms")

            fig.update_layout(title=f"Node {node}", width=figsize[0], height=figsize[1])
            fig.update_xaxes(title_text="x")
            fig.update_yaxes(title_text="Density")
            fig.show()
        return None

    # ---------- Grid mode ----------
    # Group nodes by level
    levels = sorted({nodes[n]["lvl"] for n in nodes})
    nodes_by_level = {lvl: [n for n in nodes if nodes[n]["lvl"] == lvl] for lvl in levels}
    rows = len(levels)
    cols = max(len(lst) for lst in nodes_by_level.values()) if rows > 0 else 1

    # Subplot titles aligned by row/col
    titles: list[str] = []
    for lvl in levels:
        row_nodes = nodes_by_level[lvl]
        for c in range(cols):
            titles.append(f"Node {row_nodes[c]}" if c < len(row_nodes) else "")

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, vertical_spacing=0.12)

    # One legend only (first subplot)
    first_legend_drawn = False

    for r, lvl in enumerate(levels, start=1):
        row_nodes = nodes_by_level[lvl]
        for c, node in enumerate(row_nodes, start=1):
            # True density (optional)
            if f_true is not None and node in f_true:
                fig.add_trace(
                    go.Scatter(
                        x=f_true[node]["x"],
                        y=f_true[node]["y"],
                        mode="lines",
                        name="True distribution",
                        line=dict(color="blue"),
                        showlegend=not first_legend_drawn,
                    ),
                    row=r,
                    col=c,
                )

            # Posterior density
            fig.add_trace(
                go.Scatter(
                    x=f_post[node]["x"],
                    y=f_post[node]["y"],
                    mode="lines",
                    name="Posterior",
                    line=dict(color="red"),
                    showlegend=not first_legend_drawn,
                ),
                row=r,
                col=c,
            )
            first_legend_drawn = True

            ymax_true = (
                float(f_true[node]["y"].max()) if (f_true is not None and node in f_true) else 0.0
            )
            ymax_post = float(f_post[node]["y"].max())
            ylim = -0.1 * max(ymax_true, ymax_post, 0.0)

            # Force y-range to include the negative tick baseline
            fig.add_trace(
                go.Scatter(
                    x=[f_post[node]["x"].iloc[0]],
                    y=[ylim],
                    mode="markers",
                    marker_opacity=0,
                    showlegend=False,
                ),
                row=r,
                col=c,
            )

            # Observation ticks colored by predicted cluster
            if len(np.unique(pred_part_nodes[node])) < 24:
                colors = px.colors.qualitative.Light24
                for i, obs in enumerate(x[node]):
                    fig.add_shape(
                        type="line",
                        x0=obs,
                        y0=0,
                        x1=obs,
                        y1=ylim,
                        line=dict(color=colors[int(pred_part_nodes[node][i])], width=1.5),
                        row=r,
                        col=c,
                    )
            else:
                # Keep UX silent in grid mode; avoid spamming prints
                pass

            # Axes labels
            fig.update_xaxes(title_text="x", row=r, col=c)
            fig.update_yaxes(title_text="Density", row=r, col=c)

    # Size/layout
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        legend_title_text="",
        margin=dict(l=40, r=10, t=60, b=40),
    )
    fig.show()
    return fig


def _resolve_grid_x(
    grid_x: np.ndarray | None,
    f_true: dict[str, pd.DataFrame] | None,
    x: dict[str, list[float]],
    res: dict[str, Any],
    num: int = 500,
) -> np.ndarray:
    """
    Decide the evaluation grid:
      1) if `grid_x` is provided - use it;
      2) else if `f_true` is provided - use its (first) 'x' column;
      3) else - build linspace over data range expanded by sigma_x.
    """
    if grid_x is not None:
        return np.asarray(grid_x)

    if f_true:
        for df in f_true.values():
            if "x" in df:
                grid_x = df["x"].to_numpy()
                return grid_x

    history = res["history"]
    burn_in = res["params"]["burn_in"]
    sigma_x = float(np.mean(history["sigma_x"][burn_in:]))

    x_all = list(itertools.chain(*x.values()))
    x_min = np.min(x_all) - sigma_x
    x_max = np.max(x_all) + sigma_x
    grid_x = np.linspace(x_min, x_max, num=num)
    return grid_x


def plot_with_computed_post1D(
    nodes: Nodes,
    x: XData,
    res: dict[str, Any],
    f_true: dict[str, pd.DataFrame] | None = None,
    grid_x: np.ndarray | None = None,
    *,
    grid_mode: bool = False,
    figsize: tuple[int, int] = (900, 400),
) -> dict[NodeId, pd.DataFrame]:
    """
    Convenience wrapper: compute `f_post` on `grid_x` and plot (optionally with `f_true`).

    Returns
    -------
    dict[node -> DataFrame{'x','y'}]
        The computed posterior predictive curves.
    """

    grid_x = _resolve_grid_x(grid_x, f_true, x, res, num=500)
    f_post: dict[NodeId, pd.DataFrame] = compute_posterior_predictive(nodes, x, res, grid_x)
    plot_posterior1D(nodes, x, res, f_post, f_true=f_true, grid_mode=grid_mode, figsize=figsize)
    return f_post


def plot_clusters_2D(
    nodes: dict[str, dict[str, Any]],
    x: dict[str, list[np.ndarray]],
    res: dict[str, Any],
    *,
    grid_mode: bool = False,
    figsize: tuple[int, int] = (900, 600),
) -> go.Figure | None:
    """
    Scatter plots of 2D observations colored by predicted cluster labels.

    If `grid_mode=True`, arrange nodes by level (row = level, columns = nodes at that level)
    in a single Plotly figure. Otherwise, produce one interactive figure per node.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Single figure in grid mode, else None (figures are shown individually).
    """
    # Predicted cluster labels per observation
    pred_part, pred_part_nodes, _ = predicted_clusters_2D(nodes, x, res)

    # Consistent color mapping across all nodes
    unique_labels = np.unique(pred_part)
    colors = px.colors.qualitative.Light24
    color_map = {int(lbl): colors[i % len(colors)] for i, lbl in enumerate(unique_labels)}

    # Global axis limits with a small margin
    all_pts = np.vstack([np.asarray(pt) for arr in x.values() for pt in arr])  # (N,2)
    (x_min, y_min), (x_max, y_max) = all_pts.min(axis=0), all_pts.max(axis=0)
    dx, dy = max(x_max - x_min, 1e-9), max(y_max - y_min, 1e-9)
    x_min, x_max = x_min - 0.1 * dx, x_max + 0.1 * dx
    y_min, y_max = y_min - 0.1 * dy, y_max + 0.1 * dy

    if not grid_mode:
        # One figure per node (keep Plotly Express for quick legends)
        category_orders = {"cluster": [str(int(c)) for c in unique_labels]}
        discrete_map = {str(k): v for k, v in color_map.items()}

        for node in nodes:
            df = pd.DataFrame(x[node], columns=[0, 1]).copy()
            df["cluster"] = [str(int(c)) for c in pred_part_nodes[node]]
            fig = px.scatter(
                df,
                x=0,
                y=1,
                color="cluster",
                color_discrete_map=discrete_map,
                category_orders=category_orders,
                title=f"Node {node}",
            )
            fig.update_traces(marker=dict(size=9, line=dict(width=0)))
            fig.update_layout(
                xaxis_range=[x_min, x_max],
                yaxis_range=[y_min, y_max],
                width=figsize[0],
                height=figsize[1],
                legend_title_text="Cluster",
            )
            fig.show()
        return None

    # -------- Grid mode: one figure arranged by levels --------
    levels = sorted({nodes[n]["lvl"] for n in nodes})
    nodes_by_level = {lvl: [n for n in nodes if nodes[n]["lvl"] == lvl] for lvl in levels}
    rows = len(levels)
    cols = max(len(v) for v in nodes_by_level.values()) if rows > 0 else 1

    titles: list[str] = []
    for lvl in levels:
        row_nodes = nodes_by_level[lvl]
        for c in range(cols):
            titles.append(f"Node {row_nodes[c]}" if c < len(row_nodes) else "")

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, vertical_spacing=0.10)

    # Add one trace per cluster per node (for a clean legend with consistent colors)
    shown_for_label: set[int] = set()
    for r, lvl in enumerate(levels, start=1):
        for c, node in enumerate(nodes_by_level[lvl], start=1):
            pts = np.vstack([np.asarray(p) for p in x[node]]) if len(x[node]) else np.empty((0, 2))
            labels_node = np.asarray(pred_part_nodes[node], dtype=int)

            for lbl in unique_labels:
                mask = labels_node == lbl
                if not np.any(mask):
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=pts[mask, 0],
                        y=pts[mask, 1],
                        mode="markers",
                        marker=dict(size=8, color=color_map[int(lbl)]),
                        name=f"Cluster {int(lbl)}",
                        showlegend=(lbl not in shown_for_label),
                    ),
                    row=r,
                    col=c,
                )
                shown_for_label.add(int(lbl))

            # Axes + ranges per cell
            fig.update_xaxes(title_text="x₁", range=[x_min, x_max], row=r, col=c)
            fig.update_yaxes(title_text="x₂", range=[y_min, y_max], row=r, col=c)

    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        legend_title_text="Cluster",
        margin=dict(l=40, r=10, t=60, b=40),
    )
    fig.show()
    return fig

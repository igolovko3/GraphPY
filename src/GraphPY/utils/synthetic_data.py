from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import beta, dirichlet, gamma, multivariate_normal

from GraphPY.types import Nodes


def generate_data_multivariate_gaussian(
    nodes: Nodes,
    *,
    n_dim: int = 2,
    L: int = 100,
    sigma: float = 0.0,
    a_0: float = 1,
    b_0: float = 1,
    n: dict[str, int] | int = 100,
    mu_phi: np.ndarray | None = None,
    Sigma_phi: np.ndarray | None = None,
    Sigma_x: np.ndarray | None = None,
    alphas: dict | None = None,
    parent_weights: dict | None = None,
) -> dict[str, Any]:
    assert sigma >= 0 and sigma < 1

    if isinstance(n, int):
        n = {node: n for node in nodes}

    if mu_phi is None:
        mu_phi = np.zeros(n_dim)
    if Sigma_phi is None:
        Sigma_phi = 10 * np.identity(n_dim)
    if Sigma_x is None:
        Sigma_x = 0.5 * (np.identity(2))

    if alphas is None:
        alphas = {}
        for node in nodes:
            alphas[node] = gamma.rvs(a_0, scale=1 / b_0)

    if parent_weights is None:
        parent_weights = {}
        for node in nodes:
            n_par = len(nodes[node]["par"])
            if n_par > 0:
                parent_weights[node] = dirichlet.rvs(np.ones(n_par))[0]

    parent_indicators = {}
    weights = {}
    atoms = {}
    z = {}
    x = {}

    def _remap_labels_global(z_labels):  # type: ignore
        """Remap all values across all arrays to consecutive integers starting from 1"""
        all_values = set()
        for arr in z_labels.values():
            all_values.update(arr)
        value_to_new = {old_val: new_val for new_val, old_val in enumerate(sorted(all_values), 1)}
        remapped = {}
        for key, arr in z_labels.items():
            remapped[key] = np.array([value_to_new[val] for val in arr])

        return remapped, value_to_new

    for node in nodes:
        v_node = np.zeros(L)
        for i in range(L):
            v_node[i] = beta.rvs(1 - sigma, alphas[node] + (i + 1) * sigma)
        resid = np.ones(L)
        resid[1:] = np.cumprod(1 - v_node)[:-1]
        sb_node = v_node * resid / (1 - np.prod(1 - v_node))
        weights[node] = sb_node

        if nodes[node]["lvl"] == 0:
            atoms[node] = multivariate_normal.rvs(mean=mu_phi, cov=Sigma_phi, size=L)
        else:
            samples_parents = {}
            parents = nodes[node]["par"]
            for p in parents:
                ind_samples_parents = np.random.choice(a=L, p=weights[p], size=L)
                samples_parents[p] = atoms[p][ind_samples_parents]
            indicator_parents = np.random.choice(a=parents, p=parent_weights[node], size=L)
            parent_indicators[node] = indicator_parents
            atoms[node] = np.sum(
                np.array(
                    [
                        np.where((indicator_parents == p)[:, np.newaxis], samples_parents[p], 0)
                        for p in parents
                    ]
                ),
                axis=0,
            )

        z_indices = np.random.choice(a=L, p=weights[node], size=n[node])
        z[node] = atoms[node][z_indices]
        x[node] = np.array(
            [multivariate_normal.rvs(mean=z[node][i], cov=Sigma_x) for i in range(n[node])]
        )

    z_labels = {node: [hash(tuple(z[node][i])) for i in range(n[node])] for node in nodes}

    z_labels, mapping_dict = _remap_labels_global(z_labels)

    data: dict = {
        "x": x,
        "z": z,
        "z_labels": z_labels,
        "alphas": alphas,
        "parent_weights": parent_weights,
        "parent_indicators": parent_indicators,
        "weights": weights,
        "atoms": atoms,
    }

    # (x, z, z_labels, alphas, parent_weights, parent_indicators, weights, atoms)

    return data


def plot_stickbreaking_weights(nodes: Nodes, data: dict[str, Any]) -> None:
    x, z, z_labels, alphas, parent_weights, parent_indicators, weights, atoms = data.values()

    fig = make_subplots(
        rows=len(nodes),
        cols=2,
        subplot_titles=[],
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )

    titles = []
    for node in nodes:
        titles.append(f"{node} - Weights")
        titles.append(f"{node} - Cumulative")

    fig = make_subplots(
        rows=len(nodes),
        cols=2,
        subplot_titles=titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )

    for idx, node in enumerate(nodes.keys(), 1):
        sb_node = weights[node]

        # Bar plot in left column
        bar_fig = px.bar(sb_node)
        for trace in bar_fig.data:
            fig.add_trace(trace, row=idx, col=1)

        line_fig = px.line(np.cumsum(sb_node))
        for trace in line_fig.data:
            fig.add_trace(trace, row=idx, col=2)

    fig.update_layout(
        height=300 * len(nodes),
        width=1000,
        showlegend=False,
        title_text="Node Weights Analysis",
    )

    fig.show()


def plot_data_2D(
    nodes: dict[str, dict[str, Any]],
    data: dict[str, Any],
    *,
    grid_mode: bool = True,
    figsize: tuple[int, int] = (900, 600),
) -> go.Figure | None:
    """
    Scatter plots of 2D observations colored by TRUE cluster labels (z_labels).

    If `grid_mode=True`, arrange nodes on a grid by level
    (row = level, columns = nodes at that level)
    and return a single Figure. Otherwise, show one figure per node and return None.
    """

    x, z, z_labels, alphas, parent_weights, parent_indicators, weights, atoms = data.values()

    unique_labels = sorted({lbl for labels in z_labels.values() for lbl in labels})
    colors = px.colors.qualitative.Light24
    color_map_int = {lbl: colors[i % len(colors)] for i, lbl in enumerate(unique_labels)}
    # px needs string keys for color_discrete_map (only used in non-grid mode)
    color_map_str = {str(k): v for k, v in color_map_int.items()}
    category_orders = {"cluster": [str(lbl) for lbl in unique_labels]}

    # Axis limits with padding
    all_pts = np.vstack([np.asarray(pt) for arr in x.values() for pt in arr])  # (N,2)
    (xmin, ymin), (xmax, ymax) = all_pts.min(axis=0), all_pts.max(axis=0)
    dx, dy = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    xmin, xmax = xmin - 0.1 * dx, xmax + 0.1 * dx
    ymin, ymax = ymin - 0.1 * dy, ymax + 0.1 * dy

    if not grid_mode:
        # One figure per node (simple, per-figure legend)
        for node in nodes:
            df = pd.DataFrame(x[node], columns=[0, 1]).copy()
            df["cluster"] = [str(c) for c in z_labels[node]]
            fig = px.scatter(
                df,
                x=0,
                y=1,
                color="cluster",
                color_discrete_map=color_map_str,
                category_orders=category_orders,
                title=f"Node {node}",
            )
            fig.update_traces(marker=dict(size=9, line=dict(width=0)))
            fig.update_layout(
                xaxis_range=[xmin, xmax],
                yaxis_range=[ymin, ymax],
                width=figsize[0],
                height=figsize[1],
                legend_title_text="True cluster",
            )
            fig.show()
        return None

    # Grid mode: single figure, nodes arranged by level
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

    # Add traces: one per label per node; show legend only once per label
    shown_for_label: set[int] = set()
    for r, lvl in enumerate(levels, start=1):
        for c, node in enumerate(nodes_by_level[lvl], start=1):
            pts = (
                np.vstack([np.asarray(p) for p in x[node]])
                if len(x[node]) > 0
                else np.empty((0, 2))
            )
            labels_node = np.array(z_labels[node], dtype=int)

            for lbl in unique_labels:
                mask = labels_node == lbl
                if not np.any(mask):
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=pts[mask, 0],
                        y=pts[mask, 1],
                        mode="markers",
                        marker=dict(size=8, color=color_map_int[lbl]),
                        name=f"Cluster {lbl}",
                        showlegend=(lbl not in shown_for_label),
                        legendgroup="trueclusters",
                    ),
                    row=r,
                    col=c,
                )
                shown_for_label.add(lbl)

            fig.update_xaxes(title_text="x₁", range=[xmin, xmax], row=r, col=c)
            fig.update_yaxes(title_text="x₂", range=[ymin, ymax], row=r, col=c)

    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        legend_title_text="True cluster",
        margin=dict(l=40, r=10, t=60, b=40),
    )
    fig.show()
    return fig

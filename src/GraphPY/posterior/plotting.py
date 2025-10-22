import itertools
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import invgamma, norm

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


def plot_param_posterior_1D(
    res: dict[str, Any],
    true_values: dict[str, float] | None = None,
    *,
    histnorm: str | None = "probability density",
    figsize: tuple[int, int] = (900, 600),
) -> dict[str, go.Figure]:
    """
    Posterior histograms for generic model parameters (if they are 1-D).

    Returns
    -------
    dict
        {parameter_name: plotly.graph_objects.Figure}
    """

    true_values = true_values or {}
    figs: dict[str, go.Figure] = {}

    burn_in = res["params"]["burn_in"]
    param_history = res["model"].param_history

    for param, samples in param_history.items():

        fig = px.histogram(
            x=samples[burn_in:],
            histnorm=histnorm,
            labels={"x": param, "y": "Density"},
            title=param,
        )
        fig.update_layout(width=figsize[0], height=figsize[1])

        if param in true_values:
            tv = true_values[param]
            fig.add_vline(
                x=tv,
                line_width=3,
                line_dash="dash",
                line_color="red",
            )
            # annotation
            fig.add_annotation(
                x=tv,
                xshift=10,
                y=0.95,
                yref="paper",
                text="True value",
                showarrow=False,
                textangle=-90,
                font=dict(color="black", size=14),
            )

        figs[param] = fig
        fig.show()

    return figs


def plot_param_posterior_Gaussian1D(
    res: dict[str, Any],
    true_values: dict[str, float] | None = None,
    *,
    histnorm: str | None = "probability density",
    figsize: tuple[int, int] = (900, 600),
) -> go.Figure:
    """
    2×2 grid of posteriors for Gaussian-1D hyperparameters with prior overlays.
    Panels: (1,1) μ_φ; (1,2) σ_φ²; (2,1) σ_x²; (2,2) empty.

    Notes
    -----
    - Samples for σ_φ and σ_x are squared before plotting.
    - `true_values` may contain keys 'mu_phi', 'sigma_phi', 'sigma_phi^2', 'sigma_x', 'sigma_x^2'.
      If a std key is provided ('sigma_phi' or 'sigma_x'),
      its value is squared for the vertical line.
    - Prior pdfs:
        μ_φ ~ Normal(μ_base, scale = sqrt(β_base / ((α_base + 1) * sqrt(λ_base))))
        σ_φ² ~ InvGamma(a = α_base, scale = β_base)
        σ_x² ~ InvGamma(a = α_kernel, scale = β_kernel)
    """
    tv = true_values or {}
    burn_in = int(res.get("params", {}).get("burn_in", 0))
    model = res["model"]  # expects attributes shown in the docstring

    # Pull chains (if absent, treat as empty)
    # Expect: res["model"].param_history = {"mu_phi": [...], "sigma_phi": [...], "sigma_x": [...]}
    ph = getattr(model, "param_history", {})
    mu_chain = np.asarray(ph.get("mu_phi", []), dtype=float)
    sphi_chain = np.asarray(ph.get("sigma_phi", []), dtype=float)  # std; will square
    sx_chain = np.asarray(ph.get("sigma_x", []), dtype=float)  # std; will square

    if mu_chain.size:
        mu_chain = mu_chain[burn_in:]
    if sphi_chain.size:
        sphi_chain = sphi_chain[burn_in:] ** 2  # variance
    if sx_chain.size:
        sx_chain = sx_chain[burn_in:] ** 2  # variance

    # Prior params
    # μ prior
    mu_base = float(model._mu_base)
    alpha_base = float(model._alpha_base)
    beta_base = float(model._beta_base)
    lambda_base = float(model._lambda_base)
    mu_scale = float(np.sqrt(beta_base / ((alpha_base + 1.0) * np.sqrt(lambda_base))))

    # σ_φ² prior
    a_base = alpha_base
    b_base = beta_base

    # σ_x² prior
    a_k = float(model._alpha_kernel)
    b_k = float(model._beta_kernel)

    # Helper: true-value getter with auto-square for std keys
    def _true_val(name_pdf: str) -> float | None:
        # name_pdf ∈ {"mu_phi", "sigma_phi^2", "sigma_x^2"}
        if name_pdf == "mu_phi":
            return tv.get("mu_phi")
        if name_pdf == "sigma_phi^2":
            if "sigma_phi^2" in tv:
                return tv["sigma_phi^2"]
            if "sigma_phi" in tv:
                return float(tv["sigma_phi"] ** 2)
        if name_pdf == "sigma_x^2":
            if "sigma_x^2" in tv:
                return tv["sigma_x^2"]
            if "sigma_x" in tv:
                return float(tv["sigma_x"] ** 2)
        return None

    # Build subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            r"$\Large \color{black}{\mu_\phi  \text{ (Mean of the atoms) }}$",
            r"$\Large \color{black}{\sigma_\phi^2 \text{ (Variance of the atoms) }}$",
            r"$\Large \color{black}{\sigma_x^2 \text{ (Variance of the kernel) }}$",
            "",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    # ---------- Panel (1,1): mu_phi ----------
    if mu_chain.size:
        fig.add_trace(
            go.Histogram(
                x=mu_chain,
                histnorm=histnorm,
                name="Posterior",
                marker_color="#6baed6",
                opacity=0.55,
            ),
            row=1,
            col=1,
        )
        x_min, x_max = mu_chain.min(), mu_chain.max()
        pad = 0.15 * (x_max - x_min + 1e-9)
    else:
        # Prior-based range if no samples
        x_min, x_max = mu_base - 4 * mu_scale, mu_base + 4 * mu_scale
        pad = 0.0
    xs = np.linspace(x_min - pad, x_max + pad, 400)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=norm.pdf(xs, loc=mu_base, scale=mu_scale),
            mode="lines",
            name="Prior",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )
    tv_mu = _true_val("mu_phi")
    if tv_mu is not None:
        counts, _ = np.histogram(mu_chain, bins=40, density=True)
        y_max = counts.max()
        fig.add_vline(x=tv_mu, line_width=2, line_dash="dash", line_color="black", row=1, col=1)
        fig.add_annotation(
            x=tv_mu,
            xshift=10,
            y=0.95 * y_max,
            text="True value",
            showarrow=False,
            textangle=-90,
            font=dict(color="black", size=14),
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text=r"$\mu_\phi$", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)

    # ---------- Panel (1,2): sigma_phi^2 ----------
    if sphi_chain.size:
        fig.add_trace(
            go.Histogram(
                x=sphi_chain,
                histnorm=histnorm,
                name="posterior",
                marker_color="#6baed6",
                opacity=0.55,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        v_min, v_max = sphi_chain.min(), sphi_chain.max()
        pad_v = 0.15 * (v_max - v_min + 1e-12)
    else:
        v_min = float(invgamma.ppf(1e-3, a=a_base, scale=b_base))
        v_max = float(invgamma.ppf(1 - 1e-3, a=a_base, scale=b_base))
        pad_v = 0.0
    xs_v = np.linspace(max(1e-12, v_min - pad_v), v_max + pad_v, 400)
    fig.add_trace(
        go.Scatter(
            x=xs_v,
            y=invgamma.pdf(xs_v, a=a_base, scale=b_base),
            mode="lines",
            name="prior",
            line=dict(color="blue", width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    tv_sphi2 = _true_val("sigma_phi^2")
    if tv_sphi2 is not None:
        counts, _ = np.histogram(sphi_chain, bins=40, density=True)
        y_max = counts.max()
        fig.add_vline(x=tv_sphi2, line_width=2, line_dash="dash", line_color="black", row=1, col=2)
        fig.add_annotation(
            x=tv_sphi2,
            xshift=10,
            y=0.95 * y_max,
            text="True value",
            showarrow=False,
            textangle=-90,
            font=dict(color="black", size=14),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text=r"$\sigma_\phi^2$", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=2)

    # ---------- Panel (2,1): sigma_x^2 ----------
    if sx_chain.size:
        fig.add_trace(
            go.Histogram(
                x=sx_chain,
                histnorm=histnorm,
                name="posterior",
                marker_color="#6baed6",
                opacity=0.55,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        k_min, k_max = sx_chain.min(), sx_chain.max()
        pad_k = 0.15 * (k_max - k_min + 1e-12)
    else:
        k_min = float(invgamma.ppf(1e-3, a=a_k, scale=b_k))
        k_max = float(invgamma.ppf(1 - 1e-3, a=a_k, scale=b_k))
        pad_k = 0.0
    xs_k = np.linspace(max(1e-12, k_min - pad_k), k_max + pad_k, 400)
    fig.add_trace(
        go.Scatter(
            x=xs_k,
            y=invgamma.pdf(xs_k, a=a_k, scale=b_k),
            mode="lines",
            name="prior",
            line=dict(color="blue", width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    tv_sx2 = _true_val("sigma_x^2")
    if tv_sx2 is not None:
        counts, _ = np.histogram(sx_chain, bins=40, density=True)
        y_max = counts.max()
        fig.add_vline(x=tv_sx2, line_width=2, line_dash="dash", line_color="black", row=2, col=1)
        fig.add_annotation(
            x=tv_sx2,
            xshift=10,
            y=0.95 * y_max,
            text="True value",
            showarrow=False,
            textangle=-90,
            font=dict(color="black", size=14),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text=r"$\sigma_x^2$", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)

    # ---------- Empty panel (2,2) ----------
    fig.update_xaxes(visible=False, row=2, col=2)
    fig.update_yaxes(visible=False, row=2, col=2)

    # Layout
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        legend_title_text="",
        bargap=0.05,
        margin=dict(l=50, r=20, t=60, b=40),
        title_text="",
    )

    fig.show()
    return fig


def plot_param_posterior_Gaussian2D(
    res: dict[str, Any],
    true_values: dict[str, Any] | None = None,
    *,
    histnorm: str | None = "probability density",
    figsize_mu: tuple[int, int] = (900, 300),
    figsize_cov: tuple[int, int] = (900, 600),
) -> dict[str, go.Figure]:
    """
    Three separate figures for Gaussian-2D hyperparameters with prior overlays.

    Returns
    -------
    dict:
      {
        "mu_phi":   1×2 figure with μ_1, μ_2 (posterior + prior),
        "Sigma_phi": 2×2 figure with [σ^2_11, σ^2_22, σ^2_12, empty],
        "Sigma_x":   2×2 figure with [σ^2_11, σ^2_22, σ^2_12, empty],
      }

    Notes
    -----
    - true_values structure mirrors parameters:
        true_values["mu_phi"] -> shape (2,)
        true_values["Sigma_phi"] -> shape (2,2)
        true_values["Sigma_x"] -> shape (2,2)
    """

    tv = true_values or {}
    burn_in = int(res.get("params", {}).get("burn_in", 0))
    model = res["model"]
    ph = getattr(model, "param_history", {})

    # ------- Extract chains -------
    # mu_phi: (T,2) or (2,T)
    mu_chain = np.asarray(ph.get("mu_phi", []), dtype=float)
    if mu_chain.size:
        mu_chain = mu_chain[burn_in:]
        if mu_chain.ndim == 1:
            if mu_chain.size % 2 != 0:
                raise ValueError("mu_phi 1D chain size not divisible by 2; expected 2D components.")
            mu_chain = mu_chain.reshape(-1, 2)
        elif mu_chain.ndim == 2 and mu_chain.shape[0] == 2:
            mu_chain = mu_chain.T  # (T,2)
        elif not (mu_chain.ndim == 2 and mu_chain.shape[1] == 2):
            raise ValueError("mu_phi chain must be (T,2) or (2,T).")

    # Sigma_phi: (T,2,2)
    Sphi_chain = np.asarray(ph.get("Sigma_phi", []), dtype=float)
    if Sphi_chain.size:
        Sphi_chain = Sphi_chain[burn_in:]
        if not (Sphi_chain.ndim == 3 and Sphi_chain.shape[-2:] == (2, 2)):
            raise ValueError("Sigma_phi chain must be (T,2,2).")

    # Sigma_x: (T,2,2)
    Sx_chain = np.asarray(ph.get("Sigma_x", []), dtype=float)
    if Sx_chain.size:
        Sx_chain = Sx_chain[burn_in:]
        if not (Sx_chain.ndim == 3 and Sx_chain.shape[-2:] == (2, 2)):
            raise ValueError("Sigma_x chain must be (T,2,2).")

    # ------- True values (arrays/matrices) -------
    tv_mu = np.asarray(tv.get("mu_phi", []), dtype=float) if "mu_phi" in tv else None
    tv_Sphi = np.asarray(tv.get("Sigma_phi", []), dtype=float) if "Sigma_phi" in tv else None
    tv_Sx = np.asarray(tv.get("Sigma_x", []), dtype=float) if "Sigma_x" in tv else None

    # ------- Prior parameters -------
    # μ prior covariance for marginals
    mu_base = np.asarray(model._mu_base, dtype=float).reshape(2)
    psi_base = np.asarray(model._psi_base, dtype=float).reshape(2, 2)
    nu_base = float(model._nu_base)
    lam_base = float(model._lambda_base)
    n_dim = int(getattr(model, "_n_dim", 2))
    mu_cov = psi_base / ((nu_base + n_dim + 1.0) * lam_base)
    mu_var = np.diag(mu_cov)  # variances for marginals

    # Σ_φ diag priors
    a_base = (nu_base - 1.0) / 2.0
    b_base_11 = psi_base[0, 0] / 2.0
    b_base_22 = psi_base[1, 1] / 2.0

    # Σ_x diag priors
    psi_k = np.asarray(model._psi_kernel, dtype=float).reshape(2, 2)
    nu_k = float(model._nu_kernel)
    a_k = (nu_k - 1.0) / 2.0
    b_k_11 = psi_k[0, 0] / 2.0
    b_k_22 = psi_k[1, 1] / 2.0

    # ------- Helpers -------
    POSTERIOR_COLOR = "#6baed6"
    PRIOR_COLOR = "blue"

    def _add_hist(fig: go.Figure, x, row, col, showlegend=False):  # type: ignore
        if x is None or (isinstance(x, np.ndarray) and x.size == 0):
            return None, None
        vals = np.asarray(x, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None, None
        fig.add_trace(
            go.Histogram(
                x=vals,
                histnorm=histnorm,
                name="Posterior",
                marker_color=POSTERIOR_COLOR,
                opacity=0.55,
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )
        counts, _ = np.histogram(vals, bins=40, density=True)
        y_max = float(counts.max()) if counts.size else 1.0
        x_min, x_max = float(np.min(vals)), float(np.max(vals))
        pad = 0.15 * (x_max - x_min + 1e-12)
        return (x_min - pad, x_max + pad), y_max

    def _add_prior_normal(fig, mean, var, row, col, showlegend=False, x_range=None):  # type: ignore
        if var <= 0 or not np.isfinite(var):
            return
        sd = np.sqrt(var)
        if x_range is None:
            xs = np.linspace(mean - 4 * sd, mean + 4 * sd, 400)
        else:
            xs = np.linspace(x_range[0], x_range[1], 400)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=norm.pdf(xs, loc=mean, scale=sd),
                mode="lines",
                name="Prior",
                line=dict(color=PRIOR_COLOR, width=2),
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )

    def _add_prior_invgamma(fig, a, scale, row, col, showlegend=False, x_range=None):  # type: ignore
        if a <= 0 or scale <= 0 or not np.isfinite(a) or not np.isfinite(scale):
            return
        if x_range is None:
            x_lo = float(invgamma.ppf(1e-3, a=a, scale=scale))
            x_hi = float(invgamma.ppf(1 - 1e-3, a=a, scale=scale))
        else:
            x_lo, x_hi = x_range
        xs = np.linspace(max(1e-12, x_lo), x_hi, 400)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=invgamma.pdf(xs, a=a, scale=scale),
                mode="lines",
                name="Prior",
                line=dict(color=PRIOR_COLOR, width=2),
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )

    def _add_tv_line(fig, x0, y_max, row, col):  # type: ignore
        if x0 is None or not np.isfinite(x0):
            return
        fig.add_vline(x=x0, line_width=2, line_dash="dash", line_color="black", row=row, col=col)
        fig.add_annotation(
            x=x0,
            xshift=10,
            y=0.95 * y_max,
            text="True value",
            showarrow=False,
            textangle=-90,
            font=dict(color="black", size=14),
            row=row,
            col=col,
        )

    # ============================================
    # Figure 1: mu_phi (1×2)
    # ============================================
    fig_mu = make_subplots(
        rows=1,
        cols=2,
        # subplot_titles=(r"$\Large \color{black}{\mu_1}$", r"$\Large \color{black}{\mu_2}$"),
        horizontal_spacing=0.12,
    )

    mu1 = mu_chain[:, 0] if mu_chain.size else np.array([])
    mu2 = mu_chain[:, 1] if mu_chain.size else np.array([])

    # μ1
    xr1, y1max = _add_hist(fig_mu, mu1, row=1, col=1, showlegend=True)
    _add_prior_normal(fig_mu, mu_base[0], mu_var[0], row=1, col=1, showlegend=True, x_range=xr1)
    _add_tv_line(
        fig_mu,
        (tv_mu[0] if isinstance(tv_mu, np.ndarray) and tv_mu.size >= 2 else None),
        (y1max or 1.0),
        row=1,
        col=1,
    )
    fig_mu.update_xaxes(title_text=r"$\Large \color{black}{\mu_1}$", row=1, col=1)
    fig_mu.update_yaxes(title_text="Density", row=1, col=1)

    # μ2
    xr2, y2max = _add_hist(fig_mu, mu2, row=1, col=2, showlegend=False)
    _add_prior_normal(fig_mu, mu_base[1], mu_var[1], row=1, col=2, showlegend=False, x_range=xr2)
    _add_tv_line(
        fig_mu,
        (tv_mu[1] if isinstance(tv_mu, np.ndarray) and tv_mu.size >= 2 else None),
        (y2max or 1.0),
        row=1,
        col=2,
    )
    fig_mu.update_xaxes(title_text=r"$\Large \color{black}{\mu_2}$", row=1, col=2)
    fig_mu.update_yaxes(title_text="Density", row=1, col=2)

    fig_mu.update_layout(
        width=figsize_mu[0],
        height=figsize_mu[1],
        title_text="Mean of the atoms",
        title_font=dict(color="black", size=20),
        legend_title_text="",
        bargap=0.05,
        margin=dict(l=60, r=30, t=60, b=50),
    )
    # ============================================
    # Figure 2: Sigma_phi (2×2)
    # ============================================
    fig_sphi = make_subplots(
        rows=2,
        cols=2,
        # subplot_titles=(
        #     r"$\Large \color{black}{\sigma^2_{11}}$",
        #     r"$\Large \color{black}{\sigma^2_{22}}$",
        #     r"$\Large \color{black}{\sigma^2_{12}}$",
        #     "",
        # ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )

    sphi_11 = Sphi_chain[:, 0, 0] if Sphi_chain.size else np.array([])
    sphi_22 = Sphi_chain[:, 1, 1] if Sphi_chain.size else np.array([])
    sphi_12 = (
        (0.5 * (Sphi_chain[:, 0, 1] + Sphi_chain[:, 1, 0])) if Sphi_chain.size else np.array([])
    )

    # diag 11
    xr, ymax = _add_hist(fig_sphi, sphi_11, row=1, col=1, showlegend=True)
    _add_prior_invgamma(fig_sphi, a_base, b_base_11, row=1, col=1, showlegend=True, x_range=xr)
    _add_tv_line(
        fig_sphi,
        (tv_Sphi[0, 0] if isinstance(tv_Sphi, np.ndarray) and tv_Sphi.shape == (2, 2) else None),
        (ymax or 1.0),
        row=1,
        col=1,
    )
    fig_sphi.update_xaxes(title_text=r"$\Large \color{black}{\sigma^2_{11}}$", row=1, col=1)
    fig_sphi.update_yaxes(title_text="Density", row=1, col=1)

    # diag 22
    xr, ymax = _add_hist(fig_sphi, sphi_22, row=1, col=2, showlegend=False)
    _add_prior_invgamma(fig_sphi, a_base, b_base_22, row=1, col=2, showlegend=False, x_range=xr)
    _add_tv_line(
        fig_sphi,
        (tv_Sphi[1, 1] if isinstance(tv_Sphi, np.ndarray) and tv_Sphi.shape == (2, 2) else None),
        (ymax or 1.0),
        row=1,
        col=2,
    )
    fig_sphi.update_xaxes(title_text=r"$\Large \color{black}{\sigma^2_{22}}$", row=1, col=2)
    fig_sphi.update_yaxes(title_text="Density", row=1, col=2)

    # off-diag 12 (no prior curve)
    xr, ymax = _add_hist(fig_sphi, sphi_12, row=2, col=1, showlegend=False)
    _add_tv_line(
        fig_sphi,
        (tv_Sphi[0, 1] if isinstance(tv_Sphi, np.ndarray) and tv_Sphi.shape == (2, 2) else None),
        (ymax or 1.0),
        row=2,
        col=1,
    )
    fig_sphi.update_xaxes(title_text=r"$\Large \color{black}{\sigma^2_{12}}$", row=2, col=1)
    fig_sphi.update_yaxes(title_text="Density", row=2, col=1)

    # empty (2,2)
    fig_sphi.update_xaxes(visible=False, row=2, col=2)
    fig_sphi.update_yaxes(visible=False, row=2, col=2)

    fig_sphi.update_layout(
        width=figsize_cov[0],
        height=figsize_cov[1],
        title_text="Covariance matrix of the atoms",
        title_font=dict(color="black", size=20),
        legend_title_text="",
        bargap=0.05,
        margin=dict(l=60, r=30, t=60, b=50),
    )

    # ============================================
    # Figure 3: Sigma_x (2×2)
    # ============================================
    fig_sx = make_subplots(
        rows=2,
        cols=2,
        # subplot_titles=(
        #     r"$\Large \color{black}{\sigma^2_{11}}$",
        #     r"$\Large \color{black}{\sigma^2_{22}}$",
        #     r"$\Large \color{black}{\sigma^2_{12}}$",
        #     "",
        # ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )

    sx_11 = Sx_chain[:, 0, 0] if Sx_chain.size else np.array([])
    sx_22 = Sx_chain[:, 1, 1] if Sx_chain.size else np.array([])
    sx_12 = (0.5 * (Sx_chain[:, 0, 1] + Sx_chain[:, 1, 0])) if Sx_chain.size else np.array([])

    # diag 11
    xr, ymax = _add_hist(fig_sx, sx_11, row=1, col=1, showlegend=True)
    _add_prior_invgamma(fig_sx, a_k, b_k_11, row=1, col=1, showlegend=True, x_range=xr)
    _add_tv_line(
        fig_sx,
        (tv_Sx[0, 0] if isinstance(tv_Sx, np.ndarray) and tv_Sx.shape == (2, 2) else None),
        (ymax or 1.0),
        row=1,
        col=1,
    )
    fig_sx.update_xaxes(title_text=r"$\Large \color{black}{\sigma^2_{11}}$", row=1, col=1)
    fig_sx.update_yaxes(title_text="Density", row=1, col=1)

    # diag 22
    xr, ymax = _add_hist(fig_sx, sx_22, row=1, col=2, showlegend=False)
    _add_prior_invgamma(fig_sx, a_k, b_k_22, row=1, col=2, showlegend=False, x_range=xr)
    _add_tv_line(
        fig_sx,
        (tv_Sx[1, 1] if isinstance(tv_Sx, np.ndarray) and tv_Sx.shape == (2, 2) else None),
        (ymax or 1.0),
        row=1,
        col=2,
    )
    fig_sx.update_xaxes(title_text=r"$\Large \color{black}{\sigma^2_{22}}$", row=1, col=2)
    fig_sx.update_yaxes(title_text="Density", row=1, col=2)

    # off-diag 12 (no prior curve)
    xr, ymax = _add_hist(fig_sx, sx_12, row=2, col=1, showlegend=False)
    _add_tv_line(
        fig_sx,
        (tv_Sx[0, 1] if isinstance(tv_Sx, np.ndarray) and tv_Sx.shape == (2, 2) else None),
        (ymax or 1.0),
        row=2,
        col=1,
    )
    fig_sx.update_xaxes(title_text=r"$\Large \color{black}{\sigma^2_{12}}$", row=2, col=1)
    fig_sx.update_yaxes(title_text="Density", row=2, col=1)

    # empty (2,2)
    fig_sx.update_xaxes(visible=False, row=2, col=2)
    fig_sx.update_yaxes(visible=False, row=2, col=2)

    fig_sx.update_layout(
        width=figsize_cov[0],
        height=figsize_cov[1],
        title_text="Covariance matrix of the kernel",
        title_font=dict(color="black", size=20),
        legend_title_text="",
        bargap=0.05,
        margin=dict(l=60, r=30, t=60, b=50),
    )

    fig_mu.show()
    fig_sphi.show()
    fig_sx.show()

    return {"mu_phi": fig_mu, "Sigma_phi": fig_sphi, "Sigma_x": fig_sx}

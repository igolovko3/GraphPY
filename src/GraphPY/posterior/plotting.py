import itertools
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from GraphPY.clustering.predict import predicted_clusters
from GraphPY.posterior.posterior import compute_posterior_predictive
from GraphPY.types import NodeId, Nodes, XData


def plot_posterior(
    nodes: Nodes,
    x: XData,
    res: dict[str, Any],
    f_post: dict[NodeId, pd.DataFrame],
    f_true: dict[NodeId, pd.DataFrame] | None = None,
) -> None:
    """
    Plot per-node densities and observations colored by predicted clusters.

    If `f_true` is None, only the posterior curve is shown.
    """
    pred_part, pred_part_nodes, _ = predicted_clusters(nodes, x, res)

    for node in nodes:
        if f_true is not None and node in f_true:
            fig = px.line(f_true[node], x="x", y="y", title=f"Node {node}")
            fig.update_traces(name="True distribution", showlegend=True)
            fig.add_scatter(x=f_post[node]["x"], y=f_post[node]["y"], name="Posterior")
            ymax_true = float(f_true[node]["y"].max())
        else:
            fig = px.line(f_post[node], x="x", y="y", title=f"Node {node}")
            fig.update_traces(name="Posterior", showlegend=True)
            ymax_true = 0.0

        ymax_post = float(f_post[node]["y"].max())
        ylim = -0.1 * max(ymax_true, ymax_post, 0.0)

        if len(np.unique(pred_part)) < 24:
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
            for _i, obs in enumerate(x[node]):
                fig.add_shape(type="line", x0=obs, y0=0, x1=obs, y1=ylim, line=dict(width=1.5))

        fig.show()


def _resolve_grid_x(
    grid_x: np.ndarray | None,
    f_true: dict[float, pd.DataFrame] | None,
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


def plot_with_computed_post(
    nodes: Nodes,
    x: XData,
    res: dict[str, Any],
    f_true: dict[float, pd.DataFrame] | None = None,
    grid_x: np.ndarray | None = None,
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
    plot_posterior(nodes, x, res, f_post, f_true=f_true)
    return f_post

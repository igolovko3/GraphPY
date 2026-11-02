"""
Posterior predictive densities of a new observation at each node.

- `posterior_predictive_density`: full posterior estimate. For every stored post-burn-in draw, the
  exact predictive density given that draw's state (seating, atoms, concentrations, discount,
  parent weights, kernel/base hyperparameters), averaged over draws (Rao-Blackwellised estimate).
- `compute_posterior_predictive`: plug-in estimate from a single representative state built from
  posterior point estimates (cheaper, no posterior averaging).

Both use the model's predictive at node j: a new observation joins table t of j's restaurant with
weight (n_t - sigma) and the kernel of t's atom, or opens a new table with weight
(alpha_j + sigma K_j), which draws from the parents' predictives (mixed with the parent weights)
and from the base measure G0 at a root.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

from GraphPY.clustering.predict import predicted_clusters, predicted_clusters_2D
from GraphPY.sampler.engine import CRFState
from GraphPY.sampler.kernels import ObservationModel
from GraphPY.types import Atoms, NodeId, Nodes, XData


@contextmanager
def _model_params(model: ObservationModel, params: dict[str, Any]) -> Iterator[None]:
    # temporarily set the model's hyperparameters (names as in model.param_history)
    saved = {k: getattr(model, k) for k in params}
    try:
        for k, v in params.items():
            setattr(model, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(model, k, v)


def _weights_list(nodes: Nodes, parents_weights: dict[NodeId, Any]) -> list[np.ndarray]:
    return [
        np.asarray(parents_weights[v], dtype=float) if v in parents_weights else np.ones(1)
        for v in nodes
    ]


def _frame(grid: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    # 1D grid -> columns x, y; grid of points (M, d) -> columns x0..x{d-1}, y
    grid = np.asarray(grid, dtype=float)
    if grid.ndim == 1:
        return pd.DataFrame({"x": grid, "y": y})
    cols = {f"x{i}": grid[:, i] for i in range(grid.shape[1])}
    return pd.DataFrame({**cols, "y": y})


def _partition(nodes: Nodes, res: dict[str, Any], i: int) -> np.ndarray:
    # atom label of every observation at iteration i (order: nodes, then observations)
    t2a = res["tables_to_atoms"]
    groups = res["history"]["groups"][i]
    return np.array([t2a[path[0]] for v in nodes for path in groups[v]])


def _variation_of_information(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(joint, (ia, ib), 1.0)
    p = joint / n
    pa, pb = p.sum(axis=1), p.sum(axis=0)
    nz = p > 0
    mutual = np.sum(p[nz] * np.log(p[nz] / np.outer(pa, pb)[nz]))
    entropy = -np.sum(pa * np.log(pa)) - np.sum(pb * np.log(pb))
    return float(entropy - 2 * mutual)


def posterior_predictive_density(
    nodes: Nodes,
    x: XData,
    res: dict[str, Any],
    grid: np.ndarray,
    *,
    thin: int = 1,
    return_draws: bool = False,
) -> dict[NodeId, pd.DataFrame] | tuple[dict[NodeId, pd.DataFrame], dict[NodeId, np.ndarray]]:
    """
    Posterior predictive density of a new observation at each node, averaged over the stored
    post-burn-in draws (every `thin`-th): for each draw, the exact predictive given the draw's
    full state; the estimate is the mean over draws.

    grid: (M,) points for a 1D model, or (M, d) points for a d-dimensional one.

    Returns dict[node -> DataFrame{'x','y'}] (columns x0..x{d-1} for d > 1), and with
    `return_draws=True` also dict[node -> (n_draws, M) array] of per-draw densities, e.g. for
    pointwise credible bands.
    """
    history, model = res["history"], res["model"]
    burn_in = res["params"]["burn_in"]
    iters = range(burn_in, len(history["groups"]), thin)
    node_list = list(nodes)

    draws = np.zeros((len(iters), len(node_list), len(np.asarray(grid))))
    for d, i in enumerate(iters):
        params = {k: v[i] for k, v in model.param_history.items()}
        with _model_params(model, params):
            st = CRFState.from_paths(
                nodes,
                x,
                model,
                history["groups"][i],
                history["atoms"][i],
                res["tables_to_atoms"],
                track_likelihoods=False,
            )
            alpha = np.array([history["concentration"][i][v] for v in node_list], dtype=float)
            draws[d] = st.predictive_density(
                grid,
                alpha,
                float(history["sigma"][i]),
                _weights_list(nodes, history["parents_weights"][i]),
            )

    f_post = {v: _frame(grid, draws[:, j].mean(axis=0)) for j, v in enumerate(node_list)}
    if return_draws:
        return f_post, {v: draws[:, j] for j, v in enumerate(node_list)}
    return f_post


def compute_posterior_predictive(
    nodes: Nodes, x: XData, res: dict[str, Any], grid_x: np.ndarray
) -> dict[NodeId, pd.DataFrame]:
    """
    Plug-in posterior predictive density of a new observation at each node on `grid_x`.

    Evaluates the model's predictive at a single representative state:
    - seating (tables of every restaurant, their customer counts and parent tables): the stored
      post-burn-in draw whose partition is closest, in variation of information, to the minVI
      point-estimate partition;
    - atoms: each cluster of that draw gets the point estimate of the minVI cluster that most of
      its observations belong to (median atom in 1D, mean atom in d > 1);
    - concentrations, discount, parent weights and kernel/base hyperparameters: posterior means.

    For a posterior average over draws, use `posterior_predictive_density`.

    Returns dict[node -> DataFrame{'x','y'}] (columns x0..x{d-1} for a grid of d-dim points).
    """
    history, model = res["history"], res["model"]
    burn_in = res["params"]["burn_in"]
    iters = range(burn_in, len(history["groups"]))
    node_list = list(nodes)

    one_dim = np.asarray([xi for v in nodes for xi in x[v]], dtype=float).ndim == 1
    pred_part, _, atoms_cl = (predicted_clusters if one_dim else predicted_clusters_2D)(
        nodes, x, res
    )
    pred_part = np.asarray(pred_part)

    # representative seating: the draw closest to the point-estimate partition
    i_star = min(
        iters, key=lambda i: _variation_of_information(_partition(nodes, res, i), pred_part)
    )
    part = _partition(nodes, res, i_star)
    atoms: Atoms = {}
    for label in np.unique(part):
        members, counts = np.unique(pred_part[part == label], return_counts=True)
        atoms[int(label)] = atoms_cl[members[np.argmax(counts)]]

    alpha = np.array(
        [np.mean([a[v] for a in history["concentration"][burn_in:]]) for v in node_list]
    )
    sigma = float(np.mean(history["sigma"][burn_in:]))
    weights = {
        v: np.mean([w[v] for w in history["parents_weights"][burn_in:]], axis=0)
        for v in history["parents_weights"][burn_in]
    }

    with _model_params(model, model.posterior_parameters(res=res)):
        st = CRFState.from_paths(
            nodes,
            x,
            model,
            history["groups"][i_star],
            atoms,
            res["tables_to_atoms"],
            track_likelihoods=False,
        )
        dens = st.predictive_density(grid_x, alpha, sigma, _weights_list(nodes, weights))

    return {v: _frame(grid_x, dens[j]) for j, v in enumerate(node_list)}

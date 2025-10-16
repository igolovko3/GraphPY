from typing import Any

import numpy as np
import pandas as pd
import scipy as sc

from gpy.clustering.predict import predicted_clusters
from gpy.types import NodeId, Nodes, XData


def compute_posterior_predictive(
        nodes: Nodes,
        x: XData,
        res: dict[str, Any],
        grid_x: np.ndarray
) -> dict[NodeId, pd.DataFrame]:
    """
    Compute posterior predictive f_post for each node on `grid_x`.

    Returns
    -------
    dict[node -> DataFrame{'x','y'}]
    """
    history = res['history']
    burn_in = res['params']['burn_in']
    mu_phi = res['params']['mu_phi']
    sigma_phi = res['params']['sigma_phi']
    sigma = res['params']['sigma']
    sigma_x = float(np.mean(history['sigma_x'][burn_in:]))

    pred_part, pred_part_nodes, atoms_cl = predicted_clusters(nodes, x, res)

    f_post: dict[str, pd.DataFrame] = {}
    for node in nodes:
        clusters_node = dict(zip(*np.unique(pred_part_nodes[node], return_counts=True)), strict=True)
        alpha_node = float(np.median([state[node] for state in history['concentration'][burn_in:]]))
        n_i = int(len(pred_part_nodes[node]))
        k_i = int(len(clusters_node))

        y_vals = [
            sum(
                sc.stats.norm.pdf(val, loc=atoms_cl[cl], scale=sigma_x) * (n_cl - sigma) / (n_i + alpha_node)
                for cl, n_cl in clusters_node.items()
            ) + (alpha_node + sigma * k_i) / (n_i + alpha_node) * sc.stats.norm.pdf(
                val, loc=mu_phi, scale=np.sqrt(sigma_phi ** 2 + sigma_x ** 2)
            )
            for val in grid_x
        ]
        f_post[node] = pd.DataFrame({'x': grid_x, 'y': y_vals})

    return f_post

from typing import Any

import numpy as np
import pandas as pd

from GraphPY.clustering.predict import predicted_clusters
from GraphPY.types import NodeId, Nodes, XData


def compute_posterior_predictive(
    nodes: Nodes, x: XData, res: dict[str, Any], grid_x: np.ndarray
) -> dict[NodeId, pd.DataFrame]:
    """
    Compute posterior predictive f_post for each node on `grid_x`.

    Returns
    -------
    dict[node -> DataFrame{'x','y'}]
    """
    history = res["history"]
    burn_in = res["params"]["burn_in"]
    sigma = res["params"]["sigma"]

    model = res["params"]["model"]
    model_params = model.posterior_parameters(res=res)

    pred_part, pred_part_nodes, atoms_cl = predicted_clusters(nodes, x, res)

    f_post: dict[str, pd.DataFrame] = {}
    for node in nodes:
        clusters_node = dict(
            zip(*np.unique(pred_part_nodes[node], return_counts=True), strict=True)
        )
        alpha_node = float(np.median([state[node] for state in history["concentration"][burn_in:]]))
        n_i = int(len(pred_part_nodes[node]))
        k_i = int(len(clusters_node))

        y_vals = [
            sum(
                model.like_existing(x=val, atom=atoms_cl[cl], **model_params)
                * (n_cl - sigma)
                / (n_i + alpha_node)
                for cl, n_cl in clusters_node.items()
            )
            + (alpha_node + sigma * k_i)
            / (n_i + alpha_node)
            * model.like_new_root(x=val, **model_params)
            for val in grid_x
        ]
        f_post[node] = pd.DataFrame({"x": grid_x, "y": y_vals})

    return f_post

import itertools
from typing import Any

import numpy as np

from gpy.clustering.psm import comp_psm, minVI
from gpy.types import NodeId, Nodes, XData


def predicted_clusters(
        nodes: Nodes,
        x: XData,
        res: dict[str, Any]
) -> tuple[np.ndarray, dict[NodeId, np.ndarray], dict[int, float]]:
    """s
    Get a representative partition via PSM + minVI and map it per node.

    Returns
    -------
    pred_part : (N,) array
    pred_part_nodes : dict[node -> (n_i,) array]
    atoms_cl : dict[label -> median atom]
    """
    history = res['history']
    tables_to_atoms = res['tables_to_atoms']
    burn_in = res['params']['burn_in']

    partitions = [
        [g_obs[0][1] for g_obs in itertools.chain(*[g[node] for node in nodes])]
        for g in history['groups'][burn_in:]
    ]

    psm = comp_psm(np.array(partitions))
    pred_part = minVI(psm)['cl']

    atoms = []
    for node in nodes:
        atoms.append(np.array(list(itertools.chain(
            [
                [
                    history['atoms'][i][tables_to_atoms[history['groups'][i][node][j][0]]]
                    for j in range(len(x[node]))
                ]
                for i in range(burn_in, len(history['groups']))
            ]
        ))).T)
    atoms = np.concatenate(atoms)
    clusters = np.unique(pred_part)
    atoms_cl = np.array([np.median(atoms[pred_part == cluster]) for cluster in clusters])
    atoms_cl = dict(zip(clusters, atoms_cl, strict=True))

    pred_part_nodes = dict(zip(
        nodes.keys(),
        np.split(pred_part, np.cumsum([len(x[node]) for node in nodes])[:-1]),
        strict=True
    ))
    return pred_part, pred_part_nodes, atoms_cl

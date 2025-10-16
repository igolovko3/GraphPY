from typing import TypedDict

import numpy as np

NodeId = str
TableLabel = int
TableKey = tuple[NodeId, TableLabel]  # (node, table_id) at a given level
Path = tuple[TableKey, ...]  # path from root to current level


class NodeInfo(TypedDict):
    lvl: int
    par: list[NodeId]
    desc: list[NodeId]


Nodes = dict[NodeId, NodeInfo]
Groups = dict[NodeId, list[Path]]
XData = dict[NodeId, list[float | np.ndarray]]
Atoms = dict[TableLabel, float | np.ndarray]
TablesToAtoms = dict[TableKey, TableLabel]
TablesToPath = dict[TableKey, Path]
Alpha = dict[NodeId, float]
ParentsWeights = dict[NodeId, list[float]]

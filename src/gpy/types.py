from typing import Dict, List, Tuple, TypedDict, Union
import numpy as np

NodeId = str
TableLabel = int
TableKey = Tuple[NodeId, TableLabel]  # (node, table_id) at a given level
Path = Tuple[TableKey, ...]  # path from root to current level


class NodeInfo(TypedDict):
    lvl: int
    par: List[NodeId]
    desc: List[NodeId]


Nodes = Dict[NodeId, NodeInfo]
Groups = Dict[NodeId, List[Path]]
XData = Dict[NodeId, List[Union[float, np.ndarray]]]
Atoms = Dict[TableLabel, Union[float, np.ndarray]]
TablesToAtoms = Dict[TableKey, TableLabel]
TablesToPath = Dict[TableKey, Path]
Alpha = Dict[NodeId, float]
ParentsWeights = Dict[NodeId, List[float]]

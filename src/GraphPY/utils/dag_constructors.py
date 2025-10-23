from GraphPY.types import Nodes


def fork(n_nodes: int = 4) -> Nodes:
    """
    Initialize a fork DAG with a given number of child nodes (HDP case)
    :param n_nodes: number of leaves in the graph
    """
    leaves = [str(node) for node in range(1, n_nodes + 1)]
    fork_dag: Nodes = {"0": {"lvl": [0], "desc": [node for node in leaves], "par": []}}
    fork_dag.update({node: {"lvl": [1], "desc": [], "par": ["0"]} for node in leaves})
    return fork_dag


def small_dag() -> Nodes:
    """
    Initialize a small DAG with 4 nodes
    Edges 1->2, 1->3, 2->4, 3->4
    """
    dag: Nodes = {
        "1": {
            "lvl": 0,
            "desc": ["2", "3", "4"],
            "par": [],
        },
        "2": {
            "lvl": 1,
            "desc": ["4"],
            "par": ["1"],
        },
        "3": {
            "lvl": 1,
            "desc": ["4"],
            "par": ["1"],
        },
        "4": {
            "lvl": 2,
            "desc": [],
            "par": ["2", "3"],
        },
    }

    return dag


def big_dag() -> Nodes:
    """
    Initialize the example DAG from the paper
    8 nodes, edges are:
    1->2, 1->3, 1->4
    2->5, 2->6
    3->6, 3->7,
    4->5, 4->7,
    5->8, 6->8, 7->8
    """
    dag: Nodes = {
        "1": {
            "lvl": 0,
            "desc": ["2", "3", "4", "5", "6", "7", "8"],
            "par": [],
        },
        "2": {
            "lvl": 1,
            "desc": ["5", "6", "8"],
            "par": ["1"],
        },
        "3": {
            "lvl": 1,
            "desc": ["6", "7", "8"],
            "par": ["1"],
        },
        "4": {
            "lvl": 1,
            "desc": ["5", "7", "8"],
            "par": ["1"],
        },
        "5": {
            "lvl": 2,
            "desc": ["8"],
            "par": ["2", "4"],
        },
        "6": {
            "lvl": 2,
            "desc": ["8"],
            "par": ["2", "3"],
        },
        "7": {
            "lvl": 2,
            "desc": ["8"],
            "par": ["3", "4"],
        },
        "8": {
            "lvl": 3,
            "desc": [],
            "par": ["5", "6", "7"],
        },
    }

    return dag

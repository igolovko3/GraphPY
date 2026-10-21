import numpy as np

import GraphPY.sampler.engine as engine
from GraphPY.sampler.engine import update_parent_weights
from GraphPY.types import Nodes

NODES: Nodes = {
    "A": {"lvl": 0, "par": [], "desc": ["C", "D"]},
    "B": {"lvl": 0, "par": [], "desc": ["C", "D"]},
    "C": {"lvl": 1, "par": ["A", "B"], "desc": ["D"]},
    "D": {"lvl": 2, "par": ["C"], "desc": []},
}


def dirichlet_params(monkeypatch, groups):
    seen = {}

    class FakeDirichlet:
        @staticmethod
        def rvs(weights):
            seen["weights"] = np.asarray(weights)
            return [np.asarray(weights) / np.sum(weights)]

    monkeypatch.setattr(engine, "dirichlet", FakeDirichlet)
    update_parent_weights({"C": [0.5, 0.5]}, groups, {}, NODES)
    return seen["weights"]


def test_counts_tables_opened_by_descendants(monkeypatch):
    # C1 (at A) holds C's own observations; C2 (at B) holds only D's table D1
    groups = {
        "A": [(("A", 1),)],
        "B": [(("B", 1),)],
        "C": [(("A", 1), ("C", 1)), (("A", 1), ("C", 1))],
        "D": [
            (("B", 1), ("C", 2), ("D", 1)),
            (("B", 1), ("C", 2), ("D", 1)),
            (("A", 1), ("C", 1), ("D", 2)),
        ],
    }
    # Dir(1 + #C tables at A, 1 + #C tables at B), each table counted once
    np.testing.assert_array_equal(dirichlet_params(monkeypatch, groups), [2, 2])


def test_node_without_own_observations(monkeypatch):
    groups = {
        "A": [(("A", 1),)],
        "B": [(("B", 1),)],
        "C": [],
        "D": [(("A", 1), ("C", 1), ("D", 1)), (("B", 1), ("C", 2), ("D", 2))],
    }
    np.testing.assert_array_equal(dirichlet_params(monkeypatch, groups), [2, 2])

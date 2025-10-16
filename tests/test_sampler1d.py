import numpy as np

from gpy.sampler.samplers import GPYSampler1D
from gpy.types import Nodes


def small_tree_and_data():
    # simple 2-level tree: root "r" with child "c"
    nodes: Nodes = {
        "r": {"lvl": 0, "par": [], "desc": ["c"]},
        "c": {"lvl": 1, "par": ["r"], "desc": []},
    }
    x = {
        "r": list(np.random.normal(0.0, 1.0, size=5)),
        "c": list(np.random.normal(2.0, 1.0, size=7)),
    }
    return nodes, x


def test_sampler_smoke():
    nodes, x = small_tree_and_data()
    res = GPYSampler1D(
        nodes,
        x,
        n_iter=30,
        burn_in=10,
        sigma=0.0,  # small, fast
        mu_phi=0.0,
        sigma_phi=5.0,
        a_0=1.0,
        b_0=1.0,
        group_init_mode="diffuse",
        progress=False,
    )
    # basic structure checks
    assert set(res.keys()) == {"history", "tables_to_atoms", "params"}
    hist = res["history"]
    assert "groups" in hist and len(hist["groups"]) == 30
    assert "atoms" in hist and len(hist["atoms"]) == 30
    assert "concentration" in hist and len(hist["concentration"]) == 30
    assert "sigma_x" in hist and len(hist["sigma_x"]) == 30  # 1D model exposes sigma_x
    # groups shape sanity per node
    g_last = hist["groups"][-1]
    assert set(g_last.keys()) == set(nodes.keys())
    assert len(g_last["r"]) == len(x["r"])
    assert len(g_last["c"]) == len(x["c"])

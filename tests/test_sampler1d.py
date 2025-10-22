import numpy as np

from GraphPY.sampler.samplers import GPYSamplerGaussian1D
from GraphPY.types import Nodes


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
    res = GPYSamplerGaussian1D(
        nodes,
        x,
        n_iter=30,
        burn_in=10,
        sigma=0.0,
        mu_base=0.0,
        lambda_base=1.0,
        alpha_base=1.0,
        beta_base=1.0,
        alpha_kernel=1.0,
        beta_kernel=1.0,
        random_init=False,
        a_0=1.0,
        b_0=1.0,
        group_init_mode="diffuse",
        progress=False,
    )
    # basic structure checks
    assert set(res.keys()) == {"history", "tables_to_atoms", "model", "params"}
    hist = res["history"]
    model = res["model"]
    assert "groups" in hist and len(hist["groups"]) == 31
    assert "atoms" in hist and len(hist["atoms"]) == 31
    assert "concentration" in hist and len(hist["concentration"]) == 31
    assert "sigma_x" in model.param_history and len(model.param_history["sigma_x"]) == 31
    assert "sigma_phi" in model.param_history and len(model.param_history["sigma_x"]) == 31
    g_last = hist["groups"][-1]
    assert set(g_last.keys()) == set(nodes.keys())
    assert len(g_last["r"]) == len(x["r"])
    assert len(g_last["c"]) == len(x["c"])

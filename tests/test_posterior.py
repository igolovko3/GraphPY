import numpy as np
import pandas as pd

from GraphPY.posterior.posterior import compute_posterior_predictive
from GraphPY.sampler.samplers import GPYSamplerGaussian1D
from GraphPY.types import Nodes


def tiny_run():
    nodes: Nodes = {
        "r": {"lvl": 0, "par": [], "desc": ["a", "b"]},
        "a": {"lvl": 1, "par": ["r"], "desc": []},
        "b": {"lvl": 1, "par": ["r"], "desc": []},
    }
    x = {
        "r": list(np.random.normal(0.0, 1.0, size=4)),
        "a": list(np.random.normal(-2.0, 1.0, size=5)),
        "b": list(np.random.normal(2.0, 1.0, size=6)),
    }
    res = GPYSamplerGaussian1D(nodes, x, n_iter=30, burn_in=10, progress=False)
    return nodes, x, res


def test_posterior_predictive_basic():
    nodes, x, res = tiny_run()
    data_all = np.array([v for arr in x.values() for v in arr], dtype=float)
    sigma_x = float(
        np.mean(res["model"].param_history["sigma_x"][res["params"]["burn_in"] :] or [1.0])
    )
    grid = np.linspace(data_all.min() - 5 * sigma_x, data_all.max() + 5 * sigma_x, 800)
    print(res["model"].param_history)
    f_post = compute_posterior_predictive(nodes, x, res, grid)
    # structure checks
    assert set(f_post.keys()) == set(nodes.keys())
    for _node, df in f_post.items():
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["x", "y"]
        assert len(df) == len(grid)
        # sanity: densities non-negative & finite
        assert (df["y"].values >= 0).all()
        assert np.isfinite(df["y"].values).all()


if __name__ == "__main__":
    tiny_run()

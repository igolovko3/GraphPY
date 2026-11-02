import numpy as np
import pandas as pd
import pytest

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


def test_resolve_grid_x_without_grid_or_truth():
    # regression: sigma_x was read from res["history"], which has no such key
    from GraphPY.posterior.plotting import _resolve_grid_x
    from GraphPY.sampler.samplers import GPYSamplerGaussian1D

    nodes = {"r": {"lvl": 0, "par": [], "desc": ["c"]}, "c": {"lvl": 1, "par": ["r"], "desc": []}}
    x = {"r": [0.0, 1.0, -1.0], "c": [3.0, 2.5]}
    res = GPYSamplerGaussian1D(nodes, x, n_iter=10, burn_in=2, progress=False)
    grid = _resolve_grid_x(None, None, x, res, num=50)
    assert len(grid) == 50
    assert grid.min() < -1.0 and grid.max() > 3.0


# ---------------- predictive densities respecting the hierarchy ----------------

MULTI = {
    "A": {"lvl": 0, "par": [], "desc": ["C", "D"]},
    "B": {"lvl": 0, "par": [], "desc": ["C", "D"]},
    "C": {"lvl": 1, "par": ["A", "B"], "desc": ["D"]},
    "D": {"lvl": 2, "par": ["C"], "desc": []},
}


def test_state_predictive_matches_reference_likelihood_observation():
    from GraphPY.sampler import reference
    from GraphPY.sampler.engine import CRFState
    from GraphPY.sampler.kernels import Gaussian1DModel

    rng = np.random.RandomState(0)
    np.random.seed(0)
    x = {v: list(rng.normal(0, 2, size=4)) for v in MULTI}
    groups = reference.initialize_groups(MULTI, x, "random")
    atoms = {hash(g[0]): float(rng.normal(0, 2)) for grp in groups.values() for g in grp}
    t2a = {g[lvl]: hash(g[0]) for grp in groups.values() for g in grp for lvl in range(len(g))}
    model = Gaussian1DModel(random_init=True)
    alpha = {"A": 1.0, "B": 0.7, "C": 1.3, "D": 0.5}
    pw = {"C": [0.3, 0.7]}
    grid = np.array([-3.0, 0.1, 2.5])

    st = CRFState.from_paths(MULTI, x, model, groups, atoms, track_likelihoods=False)
    dens = st.predictive_density(
        grid,
        np.array(list(alpha.values())),
        0.25,
        [np.ones(1), np.ones(1), np.array(pw["C"]), np.ones(1)],
    )
    for j, v in enumerate(MULTI):
        for m, val in enumerate(grid):
            expected = reference.likelihood_observation(
                v,
                val,
                nodes=MULTI,
                groups=groups,
                alpha=alpha,
                sigma=0.25,
                atoms=atoms,
                tables_to_atoms=t2a,
                parents_weights=pw,
                model=model,
            )[0]
            assert dens[j, m] == pytest.approx(expected, rel=1e-10)


def multi_run(dim):
    from GraphPY.sampler.engine import run_sampler
    from GraphPY.sampler.kernels import Gaussian1DModel, MultivariateGaussianModel

    rng = np.random.RandomState(1)
    np.random.seed(1)
    centers = {"A": -3.0, "B": 3.0, "C": 0.0, "D": 3.0}
    if dim == 1:
        x = {v: list(c + rng.normal(0, 1, size=8)) for v, c in centers.items()}
        model = Gaussian1DModel()
    else:
        x = {v: list(np.array([c, -c]) + rng.normal(0, 1, size=(8, 2))) for v, c in centers.items()}
        model = MultivariateGaussianModel(n_dim=2)
    res = run_sampler(nodes=MULTI, x=x, model=model, n_iter=30, burn_in=20, progress=False)
    return x, res


def test_full_and_plugin_densities_integrate_to_one_1d():
    from GraphPY.posterior.posterior import posterior_predictive_density

    x, res = multi_run(1)
    grid = np.linspace(-40, 40, 4001)
    full, draws = posterior_predictive_density(MULTI, x, res, grid, thin=2, return_draws=True)
    plug = compute_posterior_predictive(MULTI, x, res, grid)
    for v in MULTI:
        assert draws[v].shape == (len(range(20, 31, 2)), len(grid))
        np.testing.assert_allclose(full[v]["y"], draws[v].mean(axis=0))
        for f in (full, plug):
            assert list(f[v].columns) == ["x", "y"]
            assert np.sum(f[v]["y"]) * (grid[1] - grid[0]) == pytest.approx(1.0, abs=1e-3)


def test_full_and_plugin_densities_integrate_to_one_2d():
    from GraphPY.posterior.posterior import posterior_predictive_density

    x, res = multi_run(2)
    axis = np.linspace(-30, 30, 241)
    g0, g1 = np.meshgrid(axis, axis, indexing="ij")
    grid = np.column_stack([g0.ravel(), g1.ravel()])
    cell = (axis[1] - axis[0]) ** 2
    full = posterior_predictive_density(MULTI, x, res, grid, thin=5)
    plug = compute_posterior_predictive(MULTI, x, res, grid)
    for v in MULTI:
        for f in (full, plug):
            assert list(f[v].columns) == ["x0", "x1", "y"]
            assert f[v]["y"].sum() * cell == pytest.approx(1.0, abs=1e-2)

import pytest
from scipy.stats import norm

from GraphPY.sampler.engine import likelihood_observation
from GraphPY.sampler.kernels import Gaussian1DModel
from GraphPY.types import Nodes


def two_level_state():
    # root "r" with child "c"; root has tables 1, 2; child table ("c", 1) sits at root table 2
    nodes: Nodes = {
        "r": {"lvl": 0, "par": [], "desc": ["c"]},
        "c": {"lvl": 1, "par": ["r"], "desc": []},
    }
    groups = {
        "r": [(("r", 1),), (("r", 1),), (("r", 2),)],
        "c": [(("r", 2), ("c", 1))],
    }
    atoms = {10: -1.0, 20: 2.0}
    tables_to_atoms = {("r", 1): 10, ("r", 2): 20, ("c", 1): 20}
    state = dict(
        nodes=nodes,
        groups=groups,
        alpha={"r": 1.5, "c": 0.7},
        sigma=0.2,
        atoms=atoms,
        tables_to_atoms=tables_to_atoms,
        parents_weights={},
        model=Gaussian1DModel(random_init=False),
    )
    return state


def test_root_predictive_includes_new_table_likelihood():
    state = two_level_state()
    model = state["model"]
    x_obs = 0.3

    total, _, like_new = likelihood_observation("r", x_obs, **state)

    # root restaurant: table counts {1: 2, 2: 2} (table 2 includes the child's table), k = 2
    f = lambda m: norm.pdf(x_obs, loc=m, scale=model.sigma_x)  # noqa: E731
    root_prior = model.like_new_root(x_obs)
    expected = ((2 - 0.2) * f(-1.0) + (2 - 0.2) * f(2.0) + (1.5 + 0.2 * 2) * root_prior) / (1.5 + 4)

    assert like_new == pytest.approx(root_prior)
    assert total == pytest.approx(expected)


def test_child_predictive_uses_root_predictive_for_new_table():
    state = two_level_state()
    model = state["model"]
    x_obs = 0.3

    root_total, _, _ = likelihood_observation("r", x_obs, **state)
    total, _, like_new = likelihood_observation("c", x_obs, **state)

    # child restaurant: one table with one customer, k = 1
    f2 = norm.pdf(x_obs, loc=2.0, scale=model.sigma_x)
    expected = ((1 - 0.2) * f2 + (0.7 + 0.2) * root_total) / (0.7 + 1)

    assert like_new == pytest.approx(root_total)
    assert total == pytest.approx(expected)

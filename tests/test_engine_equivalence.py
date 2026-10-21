"""The fast engine must sample from the same conditionals as the reference implementation."""

import copy

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from GraphPY.sampler import engine, reference
from GraphPY.sampler.engine import CRFState
from GraphPY.sampler.kernels import Gaussian1DModel, MultivariateGaussianModel
from GraphPY.types import Nodes

NODES: Nodes = {
    "A": {"lvl": 0, "par": [], "desc": ["C", "D"]},
    "B": {"lvl": 0, "par": [], "desc": ["C", "D"]},
    "C": {"lvl": 1, "par": ["A", "B"], "desc": ["D"]},
    "D": {"lvl": 2, "par": ["C"], "desc": []},
}


def random_state(seed, n_obs=4):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    x = {v: list(rng.normal(0, 2, size=n_obs)) for v in NODES}
    groups = reference.initialize_groups(NODES, x, "random")
    atoms = {hash(g[0]): float(rng.normal(0, 2)) for grp in groups.values() for g in grp}
    model = Gaussian1DModel(random_init=False)
    model.sigma_x = 1.5  # wide kernel: no seating option is negligible
    return x, groups, atoms, model


def check_invariants(st: CRFState):
    # counts kept incrementally must match a recount from the exported paths
    groups = st.groups()
    customers, children = {}, {}
    for grp in groups.values():
        for path in grp:
            customers[path[-1]] = customers.get(path[-1], 0) + 1
            for a, b in zip(path[:-1], path[1:], strict=False):
                children.setdefault(a, set()).add(b)
    live = [t for j in range(len(st.node_ids)) for t in st.node_tables[j]]
    assert sorted(st.t_key[t] for t in live) == sorted(set(customers) | set(children))
    for t in live:
        key = st.t_key[t]
        assert st.t_count[t] == customers.get(key, 0) + len(children.get(key, ()))
        assert st.t_path[t][-1] == key
        if st.t_parent[t] != -1:
            assert st.t_path[st.t_parent[t]] == st.t_path[t][:-1]
            assert st.t_cluster[st.t_parent[t]] == st.t_cluster[t]
    C, T = np.zeros_like(st.C), np.zeros_like(st.T)
    for t in live:
        C[st.t_node[t], st.t_cluster[t]] += st.t_count[t]
        T[st.t_node[t], st.t_cluster[t]] += 1
    np.testing.assert_allclose(C, st.C)
    np.testing.assert_allclose(T, st.T)
    np.testing.assert_allclose(st.n_node, C.sum(1))
    np.testing.assert_allclose(st.k_node, T.sum(1))
    for k in st.live_slots():
        np.testing.assert_allclose(
            st.LL[:, k], st.model.log_like_matrix(st.X, st.atoms[k : k + 1])[:, 0]
        )


# ---------------- state bookkeeping ----------------


def test_round_trip_paths():
    x, groups, atoms, model = random_state(0)
    st = CRFState.from_paths(NODES, x, model, groups, atoms)
    assert st.groups() == groups
    assert st.atoms_dict() == atoms
    check_invariants(st)


def test_invariants_after_many_reseats():
    x, groups, atoms, model = random_state(1)
    st = CRFState.from_paths(NODES, x, model, groups, atoms)
    alpha = np.array([1.0, 0.7, 1.3, 0.5])
    weights = [np.ones(1), np.ones(1), np.array([0.3, 0.7]), np.ones(1)]
    lnr = model.log_like_new_root_all(st.X)
    for _ in range(30):
        for g in range(st.N):
            st.reseat(g, alpha, 0.25, weights, float(lnr[g]))
        check_invariants(st)


# ---------------- seating conditional ----------------


def reference_outcomes(node, i, x, groups, atoms, alpha, sigma, pw, model):
    """Exact distribution over the new path of obs i at node (reference formulas)."""
    groups = copy.deepcopy(groups)
    groups[node].pop(i)
    t2a, t2p = {}, {}
    for grp in groups.values():
        for g in grp:
            for lvl in range(len(g)):
                t2a[g[lvl]], t2p[g[lvl]] = hash(g[0]), g[: lvl + 1]
    x_obs = x[node][i]
    ctx = dict(
        nodes=NODES,
        groups=groups,
        alpha=alpha,
        sigma=sigma,
        atoms=atoms,
        tables_to_atoms=t2a,
        parents_weights=pw,
        model=model,
    )
    out = {}

    def walk(j, prob, suffix):
        tables, counts = np.unique(reference.get_samples(j, NODES, groups), return_counts=True)
        p_old = (counts - sigma) * np.array(
            [model.like_existing(x_obs, atoms[t2a[(j, int(t))]]) for t in tables]
        )
        total, like_par, like_new = reference.likelihood_observation(j, x_obs, **ctx)
        p_new = (alpha[j] + sigma * len(tables)) * like_new
        z = p_old.sum() + p_new
        for t, p in zip(tables, p_old, strict=True):
            path = t2p[(j, int(t))] + suffix
            out[path] = out.get(path, 0) + prob * p / z
        par = NODES[j]["par"]
        new_prob = prob * p_new / z
        if not par:
            path = ((j, "new"),) + suffix
            out[path] = out.get(path, 0) + new_prob
        elif len(par) == 1:
            walk(par[0], new_prob, ((j, "new"),) + suffix)
        else:
            w = np.array(pw[j]) * np.array(like_par)
            for p, wp in zip(par, w / w.sum(), strict=True):
                walk(p, new_prob * wp, ((j, "new"),) + suffix)

    walk(node, 1.0, ())
    return out, set(t2p)


@pytest.mark.parametrize("node,i", [("A", 0), ("C", 1), ("D", 2)])
def test_reseat_matches_reference_conditional(node, i):
    x, groups, atoms, model = random_state(4)
    alpha = {"A": 1.0, "B": 0.7, "C": 1.3, "D": 0.5}
    sigma, pw = 0.25, {"C": [0.3, 0.7]}
    expected, old_keys = reference_outcomes(node, i, x, groups, atoms, alpha, sigma, pw, model)

    st0 = CRFState.from_paths(NODES, x, model, groups, atoms)
    g = sum(len(x[v]) for v in list(NODES)[: list(NODES).index(node)]) + i
    alpha_arr = np.array([alpha[v] for v in NODES])
    weights = [np.ones(1), np.ones(1), np.array(pw["C"]), np.ones(1)]
    lnr = float(model.log_like_new_root_all(st0.X)[g])

    np.random.seed(0)
    n_draws = 6000
    hits: dict = {}
    for _ in range(n_draws):
        st = copy.deepcopy(st0)
        st.reseat(g, alpha_arr, sigma, weights, lnr)
        path = tuple(k if k in old_keys else (k[0], "new") for k in st.t_path[st.obs_table[g]])
        hits[path] = hits.get(path, 0) + 1

    assert set(hits) <= set(expected)
    assert sum(expected.values()) == pytest.approx(1.0)
    for path, p in expected.items():
        freq = hits.get(path, 0) / n_draws
        se = np.sqrt(p * (1 - p) / n_draws)
        assert abs(freq - p) < 4.5 * se + 1e-9, (path, freq, p)


# ---------------- hyperparameter conditionals ----------------


@pytest.mark.parametrize("counts", [[3, 1, 1], [7], [2, 2, 5, 1, 1, 9], []])
def test_eppf_terms_match_reference(counts):
    counts = np.array(counts, dtype=float)
    for alpha in [0.3, 2.0]:
        for sigma in [0.0, 0.35]:
            assert engine._log_eppf_alpha(alpha, sigma, counts) == pytest.approx(
                reference.concentration_ll(alpha, sigma, counts.astype(int))
            )
        grid = engine.DISCOUNT_GRID
        np.testing.assert_allclose(
            engine._log_eppf_sigma(alpha, grid, counts),
            [reference.discount_ll(alpha, s, counts.astype(int)) for s in grid],
        )


def test_parent_counts_match_reference():
    x, groups, atoms, model = random_state(5, n_obs=6)
    st = CRFState.from_paths(NODES, x, model, groups, atoms)
    j = st.idx["C"]
    tables = {g[:2] for v in ["C", "D"] for g in groups[v]}  # every table at C, by its path prefix
    expected = [sum(t[0][0] == p for t in tables) for p in NODES["C"]["par"]]
    np.testing.assert_array_equal(st.parent_table_counts(j), expected)
    for v in NODES:
        _, counts = np.unique(reference.get_samples(v, NODES, groups), return_counts=True)
        assert sorted(st.table_counts(st.idx[v])) == sorted(counts)


# ---------------- vectorised kernels ----------------


def test_vectorised_kernels_1d():
    model = Gaussian1DModel(random_init=True)
    X, A = np.array([0.3, -1.0, 2.5]), np.array([0.0, 1.0])
    LL = model.log_like_matrix(X, A)
    for n in range(3):
        for k in range(2):
            assert LL[n, k] == pytest.approx(np.log(model.like_existing(X[n], A[k])))
        assert model.log_like_new_root_all(X)[n] == pytest.approx(np.log(model.like_new_root(X[n])))


def test_vectorised_kernels_multivariate():
    model = MultivariateGaussianModel(n_dim=2, random_init=True)
    X = np.array([[0.3, 1.0], [-1.0, 0.2], [2.5, -0.4]])
    A = np.array([[0.0, 0.0], [1.0, -1.0]])
    LL = model.log_like_matrix(X, A)
    for n in range(3):
        for k in range(2):
            assert LL[n, k] == pytest.approx(
                multivariate_normal.logpdf(X[n], mean=A[k], cov=model.Sigma_x)
            )
        assert model.log_like_new_root_all(X)[n] == pytest.approx(np.log(model.like_new_root(X[n])))
    model.Sigma_x = 2.0 * np.eye(2)  # cache must follow a replaced Sigma_x
    assert model.log_like_matrix(X, A)[0, 0] == pytest.approx(
        multivariate_normal.logpdf(X[0], mean=A[0], cov=model.Sigma_x)
    )


def test_sample_atoms_posterior_moments(monkeypatch):
    # replace the draws by their mean / sd to compare with the closed-form posterior
    m1 = Gaussian1DModel(random_init=True)
    monkeypatch.setattr(np.random, "normal", lambda mean, std: np.stack([mean, std]))
    n_k, s_k = np.array([1.0, 4.0]), np.array([0.5, -2.0])
    mean, std = m1.sample_atoms(n_k, s_k)
    prec = 1 / m1.sigma_phi**2 + n_k / m1.sigma_x**2
    np.testing.assert_allclose(std, 1 / np.sqrt(prec))
    np.testing.assert_allclose(mean, (s_k / m1.sigma_x**2 + m1.mu_phi / m1.sigma_phi**2) / prec)

    m2 = MultivariateGaussianModel(n_dim=2, random_init=True)
    monkeypatch.setattr(np.random, "multivariate_normal", lambda mean, cov: mean)
    n_k, s_k = np.array([3.0]), np.array([[1.0, -2.0]])
    post_cov = np.linalg.inv(np.linalg.inv(m2.Sigma_phi) + 3 * np.linalg.inv(m2.Sigma_x))
    post_mean = post_cov @ (
        np.linalg.inv(m2.Sigma_x) @ s_k[0] + np.linalg.inv(m2.Sigma_phi) @ m2.mu_phi
    )
    np.testing.assert_allclose(m2.sample_atoms(n_k, s_k)[0], post_mean)


# ---------------- public API ----------------


def test_run_sampler_output_format():
    np.random.seed(2)
    x = {v: list(np.random.normal(0, 2, size=5)) for v in NODES}
    res = engine.run_sampler(
        nodes=NODES, x=x, model=Gaussian1DModel(), n_iter=20, burn_in=5, progress=False
    )
    hist, t2a = res["history"], res["tables_to_atoms"]
    assert set(res) == {"history", "tables_to_atoms", "model", "params"}
    assert all(len(hist[k]) == 21 for k in hist)
    for groups, atoms in zip(hist["groups"], hist["atoms"], strict=True):
        for v in NODES:
            assert len(groups[v]) == 5
            for path in groups[v]:
                assert len(path) == NODES[v]["lvl"] + 1 and path[-1][0] == v
                assert t2a[path[0]] in atoms  # the access pattern of clustering/predict.py


def test_gibbs_sweep_updates_reference_state_in_place():
    x, groups, atoms, model = random_state(6)
    t2a = {g[lvl]: hash(g[0]) for grp in groups.values() for g in grp for lvl in range(len(g))}
    t2p = {}
    alpha, pw = {v: 1.0 for v in NODES}, {"C": [0.5, 0.5]}
    for _ in range(5):
        engine.gibbs_sweep(
            nodes=NODES,
            x=x,
            model=model,
            groups=groups,
            atoms=atoms,
            tables_to_atoms=t2a,
            tables_to_path=t2p,
            alpha=alpha,
            sigma=0.2,
            parents_weights=pw,
            a_0=1.0,
            b_0=1.0,
        )
        for grp in groups.values():
            for path in grp:
                assert t2a[path[0]] in atoms
                assert all(t2p[path[lvl]] == path[: lvl + 1] for lvl in range(len(path)))
        x_check = {v: [atoms[t2a[p[0]]] for p in groups[v]] for v in NODES}
        assert all(np.all(np.isfinite(vals)) for vals in x_check.values())

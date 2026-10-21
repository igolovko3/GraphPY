"""
Time per sweep of the fast engine vs the reference implementation, from the same state.

Data: 2D Gaussian mixture simulated from the model on the paper's 8-node DAG (`big_dag`).
Both samplers are timed from a state reached after warm-up sweeps (steady state), so early
sweeps from a diffuse start (one cluster per observation) don't dominate the comparison.
Run from the repo root: python benchmarks/bench_engine.py [--quick]
"""

from __future__ import annotations

import copy
import sys
import time

import numpy as np
from scipy.stats import dirichlet

from GraphPY.sampler import engine, reference
from GraphPY.sampler.kernels import MultivariateGaussianModel
from GraphPY.utils.dag_constructors import big_dag
from GraphPY.utils.synthetic_data import generate_data_multivariate_gaussian


def data(n_per_node: int, seed: int = 11):
    nodes = big_dag()
    np.random.seed(seed)
    d = generate_data_multivariate_gaussian(
        nodes, L=300, sigma=0.25, n=n_per_node, alphas={v: 1.0 for v in nodes}, Sigma_x=np.eye(2)
    )
    return nodes, {v: list(d["x"][v]) for v in nodes}


def warm_state(nodes, x, n_warm: int = 20):
    np.random.seed(1)
    model = MultivariateGaussianModel(n_dim=2)
    groups = engine.initialize_groups(nodes, x, "concentrated")
    zero = np.zeros(2)
    atoms = {hash(g[0]): zero for grp in groups.values() for g in grp}
    st = engine.CRFState.from_paths(nodes, x, model, groups, atoms)
    alpha = {v: 1.0 for v in nodes}
    pw = {
        v: list(dirichlet.rvs(np.ones(len(nodes[v]["par"])))[0])
        for v in nodes
        if len(nodes[v]["par"]) > 1
    }
    sigma = 0.0
    for _ in range(n_warm):
        sigma = engine._sweep(
            st, alpha=alpha, sigma=sigma, parents_weights=pw, a_0=1.0, b_0=1.0, freeze_sigma=False
        )
    groups = st.groups()
    t2p = {
        p[: lvl + 1][-1]: p[: lvl + 1] for g in groups.values() for p in g for lvl in range(len(p))
    }
    return dict(
        model=model,
        groups=groups,
        atoms=st.atoms_dict(),
        tables_to_atoms=dict(st.tables_to_atoms),
        tables_to_path=t2p,
        alpha=alpha,
        sigma=sigma,
        parents_weights=pw,
        n_clusters=len(st.live_slots()),
    )


def time_reference(nodes, x, state, n_sweeps: int) -> float:
    s = copy.deepcopy(state)
    s.pop("n_clusters")
    t0 = time.perf_counter()
    for _ in range(n_sweeps):
        s["sigma"] = reference.gibbs_sweep(nodes=nodes, x=x, a_0=1.0, b_0=1.0, **s)
    return (time.perf_counter() - t0) / n_sweeps


def time_engine(nodes, x, state, n_sweeps: int) -> float:
    s = copy.deepcopy(state)
    st = engine.CRFState.from_paths(
        nodes, x, s["model"], s["groups"], s["atoms"], s["tables_to_atoms"]
    )
    sigma = s["sigma"]
    t0 = time.perf_counter()
    for _ in range(n_sweeps):
        sigma = engine._sweep(
            st,
            alpha=s["alpha"],
            sigma=sigma,
            parents_weights=s["parents_weights"],
            a_0=1.0,
            b_0=1.0,
            freeze_sigma=False,
        )
    return (time.perf_counter() - t0) / n_sweeps


def main(quick: bool) -> None:
    sizes = [25, 100] if quick else [25, 100, 400, 1600]
    print(f"{'N':>6} {'clusters':>9} {'reference':>12} {'engine':>10} {'speed-up':>9}")
    for n in sizes:
        nodes, x = data(n)
        n_obs = sum(len(v) for v in x.values())
        state = warm_state(nodes, x)
        t_new = time_engine(nodes, x, state, 20)
        if n_obs <= 800:
            t_ref = time_reference(nodes, x, state, 3)
            ref, ratio = f"{1e3 * t_ref:9.0f} ms", f"{t_ref / t_new:8.0f}x"
        else:
            ref, ratio = f"{'-':>12}", f"{'-':>9}"
        print(f"{n_obs:>6} {state['n_clusters']:>9} {ref} {1e3 * t_new:7.1f} ms {ratio}")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)

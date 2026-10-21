from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from scipy.stats import dirichlet, norm
from tqdm import tqdm

from GraphPY.sampler.kernels import ObservationModel
from GraphPY.types import (
    Alpha,
    Atoms,
    Groups,
    NodeId,
    Nodes,
    ParentsWeights,
    Path,
    TableLabel,
    TablesToAtoms,
    TablesToPath,
    XData,
)

# ---------------- helpers ----------------


def initialize_groups(nodes: Nodes, x: XData, group_init_mode: str = "random") -> Groups:
    groups_: Groups = {}
    n_tables_in_node: dict[NodeId, int] = {}

    def random_path_to(node: NodeId) -> Path:
        # path of a random customer at `node`; a node without observations gets a fresh table
        if groups_[node]:
            return groups_[node][np.random.choice(len(groups_[node]))]
        n_tables_in_node[node] = n_tables_in_node.get(node, 0) + 1
        table = (node, n_tables_in_node[node])
        if nodes[node]["lvl"] == 0:
            return (table,)
        return random_path_to(str(np.random.choice(nodes[node]["par"]))) + (table,)

    for node, info in nodes.items():
        lvl = info["lvl"]
        n_obs = len(x[node])
        groups_[node] = []
        par = info["par"]
        path_to_root: list[NodeId] = [node]

        if group_init_mode == "random":
            if lvl == 0:
                groups_[node] = [((node, i + 1),) for i in range(n_obs)]
            else:
                for i in range(n_obs):
                    parent = str(np.random.choice(par))
                    groups_[node].append(random_path_to(parent) + ((node, i),))

        else:
            for _ in range(lvl):
                path_to_root.append(nodes[path_to_root[-1]]["par"][0])
            path_to_root = path_to_root[::-1]
            for _ in range(n_obs):
                for node_in_path in path_to_root:
                    n_tables_in_node[node_in_path] = n_tables_in_node.get(node_in_path, 0) + int(
                        group_init_mode == "diffuse"
                    )
                groups_[node].append(
                    tuple(
                        (node_in_path, n_tables_in_node[node_in_path])
                        for node_in_path in path_to_root
                    )
                )

    return groups_


def get_tables_from_children(node: NodeId, nodes: Nodes, groups: Groups) -> list[Path]:
    desc = nodes[node]["desc"]
    lvl = nodes[node]["lvl"]
    tables_from_children = set(
        itertools.chain(*[[g[: lvl + 2] for g in groups[i] if g[lvl][0] == node] for i in desc])
    )
    return list(tables_from_children)


def get_samples(node: NodeId, nodes: Nodes, groups: Groups) -> list[TableLabel]:
    lvl = nodes[node]["lvl"]
    tables_from_children = get_tables_from_children(node, nodes, groups)
    return [g[lvl][1] for g in groups[node] + tables_from_children]


# ---------------- likelihood / sampling that use the model ----------------


def likelihood_observation(
    node: NodeId,
    x_obs: Any,
    *,
    nodes: Nodes,
    groups: Groups,
    alpha: Alpha,
    sigma: float,
    atoms: Atoms,
    tables_to_atoms: TablesToAtoms,
    parents_weights: ParentsWeights,
    model: ObservationModel,
) -> tuple[float, list[float] | None, float]:
    lvl = nodes[node]["lvl"]
    alpha_node = alpha[node]

    samples = get_samples(node, nodes, groups)
    tables, counts = np.unique(samples, return_counts=True)
    n_samples, n_tables = int(sum(counts)), int(len(tables))

    # CRP probabilities (without likelihoods)
    p_old = (counts - sigma) / (alpha_node + n_samples)
    p_new = (alpha_node + sigma * n_tables) / (alpha_node + n_samples)

    # multiply by existing-table likelihood terms
    for t in range(n_tables):
        at_label = tables_to_atoms[(node, int(tables[t]))]
        p_old[t] *= float(model.like_existing(np.asarray([x_obs]).ravel(), atoms[at_label]))

    likelihood_parents = None
    if lvl == 0:
        likelihood_new = float(model.like_new_root(np.asarray([x_obs]).ravel()))
    else:
        parents = nodes[node]["par"]
        if len(parents) == 1:
            likelihood_new = likelihood_observation(
                parents[0],
                x_obs,
                nodes=nodes,
                groups=groups,
                alpha=alpha,
                sigma=sigma,
                atoms=atoms,
                tables_to_atoms=tables_to_atoms,
                parents_weights=parents_weights,
                model=model,
            )[0]
        else:
            weights = parents_weights[node]
            likelihood_parents = [
                likelihood_observation(
                    p,
                    x_obs,
                    nodes=nodes,
                    groups=groups,
                    alpha=alpha,
                    sigma=sigma,
                    atoms=atoms,
                    tables_to_atoms=tables_to_atoms,
                    parents_weights=parents_weights,
                    model=model,
                )[0]
                for p in parents
            ]
            likelihood_new = float(np.sum(np.array(weights) * np.array(likelihood_parents)))
    p_new *= likelihood_new

    return float(np.sum(p_old) + p_new), likelihood_parents, likelihood_new


def sample_from_node(
    node: NodeId,
    *,
    x_obs: Any,
    cache: list[Path] | None,
    nodes: Nodes,
    groups: Groups,
    alpha: Alpha,
    sigma: float,
    atoms: Atoms,
    tables_to_atoms: TablesToAtoms,
    parents_weights: ParentsWeights,
    tables_to_path: TablesToPath,
    model: ObservationModel,
) -> Path:
    if cache is None:
        cache = []

    lvl = nodes[node]["lvl"]
    alpha_node = alpha[node]

    samples = get_samples(node, nodes, groups)
    tables, counts = np.unique(samples, return_counts=True)
    n_samples, n_tables = int(sum(counts)), int(len(tables))

    p_old = (counts - sigma) / (alpha_node + n_samples)
    p_new = (alpha_node + sigma * n_tables) / (alpha_node + n_samples)

    for t in range(n_tables):
        at_label = tables_to_atoms[(node, int(tables[t]))]
        p_old[t] *= float(model.like_existing(np.asarray([x_obs]).ravel(), atoms[at_label]))

    likelihood_node, likelihood_parents, likelihood_new = likelihood_observation(
        node,
        x_obs,
        nodes=nodes,
        groups=groups,
        alpha=alpha,
        sigma=sigma,
        atoms=atoms,
        tables_to_atoms=tables_to_atoms,
        parents_weights=parents_weights,
        model=model,
    )
    p_new *= float(likelihood_new)

    p = np.array(list(p_old) + [p_new], dtype=float)
    p = p / p.sum()

    tl = list(map(int, list(tables)))
    tl.append(max(tl, default=0) + 1)
    table_sampled = int(np.random.choice(a=tl, size=1, p=p)[0])

    if table_sampled < max(tl):
        cache.append(tables_to_path[(node, table_sampled)])
    else:
        cache.append(tuple([(node, table_sampled)]))
        if lvl == 0:
            atoms[hash((node, table_sampled))] = model.sample_root_atom(np.asarray(x_obs))
        else:
            parents = nodes[node]["par"]
            if len(parents) == 1:
                parent = parents[0]
            else:
                assert likelihood_parents is not None
                p_par = np.array(parents_weights[node]) * np.array(likelihood_parents)
                p_par = p_par / p_par.sum()
                parent = str(np.random.choice(a=parents, p=p_par))
            sample_from_node(
                parent,
                x_obs=x_obs,
                cache=cache,
                nodes=nodes,
                groups=groups,
                alpha=alpha,
                sigma=sigma,
                atoms=atoms,
                tables_to_atoms=tables_to_atoms,
                parents_weights=parents_weights,
                tables_to_path=tables_to_path,
                model=model,
            )

    return tuple(itertools.chain(*cache[::-1]))


def update_parent_weights(
    parents_weights: ParentsWeights, groups: Groups, alpha: Alpha, nodes: Nodes
) -> None:
    # Note: dependence on alpha is deprecated
    for node in list(parents_weights.keys()):
        lvl = nodes[node]["lvl"]
        # every table at the node, including tables opened only by descendants' tables
        tables = {
            g[: lvl + 1]
            for v in [node, *nodes[node]["desc"]]
            for g in groups[v]
            if len(g) > lvl and g[lvl][0] == node
        }
        tables_by_parents = [t[lvl - 1][0] for t in tables]
        counts = dict(zip(*np.unique(tables_by_parents, return_counts=True), strict=True))
        # TODO: re-write parameter
        weights = np.array([1 + counts.get(parent, 0) for parent in nodes[node]["par"]])
        parents_weights[node] = list(dirichlet.rvs(weights)[0])


def concentration_ll(alpha: float, sigma: float, counts: Any) -> float:
    n = int(np.sum(counts))
    j = int(len(counts))
    return float(
        np.sum([np.log(alpha + i * sigma) for i in range(1, j)])
        - np.sum([np.log(alpha + i) for i in range(1, n)])
    )


def discount_ll(alpha: float, sigma: float, counts: Any) -> float:
    j = int(len(counts))
    return float(
        np.sum([np.log(alpha + i * sigma) for i in range(1, j)])
        + np.sum([np.sum([np.log(1 - sigma + k) for k in range(n_i - 1)]) for n_i in counts])
    )


def update_concentration(
    nodes: Nodes,
    groups: Groups,
    alpha: Alpha,
    a_0: float,
    b_0: float,
    sigma: float,
) -> None:
    for node in nodes:
        samples = get_samples(node, nodes, groups)
        # customers per table in this node's restaurant (own obs + descendants' tables)
        counts = np.unique(samples, return_counts=True)[1]
        alpha_curr = alpha[node]

        # proposal: standard normal proposal
        alpha_prop = float(norm.rvs(loc=alpha_curr))

        if alpha_prop > 0:
            mh_ratio = (
                (alpha_prop / alpha_curr) ** (a_0 - 1)
                * np.exp(-b_0 * (alpha_prop - alpha_curr))
                * np.exp(
                    concentration_ll(alpha_prop, sigma, counts)
                    - concentration_ll(alpha_curr, sigma, counts)
                )
                # * p_rev
                # / p_prop
            )
            if np.random.uniform() < min(1.0, float(mh_ratio)):
                alpha[node] = float(alpha_prop)


def update_discount(nodes: Nodes, groups: Groups, alpha: Alpha) -> float:
    # PY discount must lie in [0, 1): drop sigma = 1 from the grid
    grid = np.linspace(0, 1, 21)[:-1]
    log_p = np.zeros(len(grid))

    for node in nodes:

        alpha_node = alpha[node]
        samples = get_samples(node, nodes, groups)
        # customers per table in this node's restaurant (own obs + descendants' tables)
        counts = np.unique(samples, return_counts=True)[1]

        for k in range(len(grid)):
            sigma_prop = grid[k]
            log_p[k] += discount_ll(alpha_node, sigma_prop, counts)

    log_p -= np.max(log_p)
    p = np.exp(log_p)
    p /= np.sum(p)

    return float(np.random.choice(grid, p=p))


# ---------------- main engine ----------------


def gibbs_sweep(
    *,
    nodes: Nodes,
    x: XData,
    model: ObservationModel,
    groups: Groups,
    atoms: Atoms,
    tables_to_atoms: TablesToAtoms,
    tables_to_path: TablesToPath,
    alpha: Alpha,
    sigma: float,
    parents_weights: ParentsWeights,
    a_0: float,
    b_0: float,
    freeze_sigma: bool = False,
) -> float:
    """
    One sweep of the sampler; updates the state in place and returns the (new) discount.
    """
    # table assignments
    for node in nodes:
        for i in range(len(x[node])):
            # remove obs i from every restaurant on its path, so empty tables vanish
            groups[node].pop(i)
            new_path = sample_from_node(
                node,
                x_obs=x[node][i],
                cache=None,
                nodes=nodes,
                groups=groups,
                alpha=alpha,
                sigma=sigma,
                atoms=atoms,
                tables_to_atoms=tables_to_atoms,
                parents_weights=parents_weights,
                tables_to_path=tables_to_path,
                model=model,
            )
            # update maps
            for lvl in range(len(new_path)):
                tables_to_atoms[new_path[lvl]] = hash(new_path[0])
                tables_to_path[new_path[lvl]] = new_path[: lvl + 1]
            groups[node].insert(i, new_path)

    # model-specific parameter updates
    model.update_atoms(groups, x, tables_to_atoms, atoms)
    model.update_base_distribution(groups, x, atoms, tables_to_atoms)
    model.update_kernel(groups, x, atoms, tables_to_atoms)

    update_parent_weights(parents_weights, groups, alpha, nodes)
    update_concentration(nodes, groups, alpha, a_0, b_0, sigma)
    if not freeze_sigma:
        sigma = update_discount(nodes, groups, alpha)
    return sigma


def run_sampler(
    *,
    nodes: Nodes,
    x: XData,
    model: ObservationModel,
    n_iter: int = 1000,
    burn_in: int | None = None,
    sigma: float = 0.0,
    a_0: float = 1.0,
    b_0: float = 1.0,
    group_init_mode: str = "diffuse",
    progress: bool = True,
    freeze_sigma: bool = False,
) -> dict[str, Any]:
    """
    Dimension/distribution-agnostic MCMC. All likelihood-related calculations delegated to `model`.
    """
    # downstream code pairs groups (nodes order) with x positionally
    assert list(nodes) == list(x), "nodes and x must have the same keys in the same order"
    assert group_init_mode in ["diffuse", "concentrated", "random"]

    if burn_in is None:
        burn_in = int(0.3 * n_iter)
    assert burn_in < n_iter

    # init
    groups = initialize_groups(nodes, x, group_init_mode)
    alpha: Alpha = {node: float(np.random.gamma(shape=a_0, scale=1 / b_0)) for node in nodes}
    parents_weights: ParentsWeights = {
        node: list(dirichlet.rvs(np.ones(len(nodes[node]["par"])))[0])
        for node in nodes
        if len(nodes[node]["par"]) > 1
    }

    # atoms & maps from initial groups
    atom_labels = {hash(g[0]) for t in groups.values() for g in t}
    zero = 0 * np.array([xi for arr in x.values() for xi in arr], dtype=float)[0]
    atoms: Atoms = {lab: zero for lab in atom_labels}
    tables_to_atoms: TablesToAtoms = {}
    tables_to_path: TablesToPath = {}
    for group in groups.values():
        for obs in group:
            for lvl in range(len(obs)):
                tables_to_atoms[obs[lvl]] = hash(obs[0])
                tables_to_path[obs[lvl]] = obs[: lvl + 1]

    history: dict[str, list[Any]] = {
        "groups": [{k: list(v) for k, v in groups.items()}],
        "atoms": [dict(atoms)],
        "parents_weights": [dict(parents_weights)],
        "concentration": [dict(alpha)],
        "sigma": [sigma],
    }

    it_range = tqdm(range(n_iter), desc="Processing", unit="iter") if progress else range(n_iter)
    for _ in it_range:
        sigma = gibbs_sweep(
            nodes=nodes,
            x=x,
            model=model,
            groups=groups,
            atoms=atoms,
            tables_to_atoms=tables_to_atoms,
            tables_to_path=tables_to_path,
            alpha=alpha,
            sigma=sigma,
            parents_weights=parents_weights,
            a_0=a_0,
            b_0=b_0,
            freeze_sigma=freeze_sigma,
        )

        # record
        history["groups"].append({k: list(v) for k, v in groups.items()})
        history["atoms"].append(dict(atoms))
        history["parents_weights"].append(dict(parents_weights))
        history["concentration"].append(dict(alpha))
        history["sigma"].append(sigma)

    return {
        "history": history,
        "tables_to_atoms": tables_to_atoms,
        "model": model,
        "params": {
            "burn_in": burn_in,
            "a_0": a_0,
            "b_0": b_0,
        },
    }

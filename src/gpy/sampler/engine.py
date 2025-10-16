from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from scipy.special import poch
from scipy.stats import dirichlet
from tqdm import tqdm

from gpy.sampler.kernels import ObservationModel
from gpy.types import (
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


def initialize_groups(nodes: Nodes, x: XData, group_init_mode: str = "diffuse") -> Groups:
    groups_: Groups = {}
    n_tables_in_node: dict[NodeId, int] = {}
    for node, info in nodes.items():
        lvl = info["lvl"]
        n_obs = len(x[node])
        groups_[node] = []
        path_to_root: list[NodeId] = [node]
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
                    (node_in_path, n_tables_in_node[node_in_path]) for node_in_path in path_to_root
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


def get_samples(
    node: NodeId, nodes: Nodes, groups: Groups, not_i: int | None = None
) -> list[TableLabel]:
    lvl = nodes[node]["lvl"]
    tables_from_children = get_tables_from_children(node, nodes, groups)
    if not_i is None:
        return [g[lvl][1] for g in groups[node] + tables_from_children]
    return [
        g[lvl][1] for g in groups[node][:not_i] + groups[node][not_i + 1 :] + tables_from_children
    ]


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
) -> tuple[float, list[float] | None]:
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
        p_new *= float(model.like_new_root(np.asarray([x_obs]).ravel()))
    else:
        parents = nodes[node]["par"]
        if len(parents) == 1:
            p_new *= likelihood_observation(
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
            p_new *= float(np.sum(np.array(weights) * np.array(likelihood_parents)))

    return float(np.sum(p_old) + p_new), likelihood_parents


def sample_from_node(
    node: NodeId,
    *,
    i: int | None,
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
    x: XData,
    model: ObservationModel,
) -> Path:
    assert not (i is None and x_obs is None)
    if x_obs is None:
        x_obs = x[node][i]
    if cache is None:
        cache = []

    lvl = nodes[node]["lvl"]
    alpha_node = alpha[node]

    samples = get_samples(node, nodes, groups, not_i=i)
    tables, counts = np.unique(samples, return_counts=True)
    n_samples, n_tables = int(sum(counts)), int(len(tables))

    p_old = (counts - sigma) / (alpha_node + n_samples)
    p_new = (alpha_node + sigma * n_tables) / (alpha_node + n_samples)

    for t in range(n_tables):
        at_label = tables_to_atoms[(node, int(tables[t]))]
        p_old[t] *= float(model.like_existing(np.asarray([x_obs]).ravel(), atoms[at_label]))

    likelihood_new, likelihood_parents = likelihood_observation(
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
    tl.append(max(tl) + 1)
    table_sampled = int(np.random.choice(a=tl, size=1, p=p)[0])

    if table_sampled < max(tl):
        cache.append(tables_to_path[(node, table_sampled)])
    else:
        cache.append(tuple([(node, table_sampled)]))
        if lvl == 0:
            atoms[table_sampled] = model.sample_root_atom(np.asarray(x_obs))
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
                i=None,
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
                x=x,
                model=model,
            )

    return tuple(itertools.chain(*cache[::-1]))


def update_parent_weights(
    parents_weights: ParentsWeights, groups: Groups, alpha: Alpha, nodes: Nodes
) -> None:
    for node in list(parents_weights.keys()):
        tables_by_parents = [g[-2][0] for g in set(groups[node])]
        counts = dict(zip(*np.unique(tables_by_parents, return_counts=True), strict=True))
        weights = np.array([alpha[parent] + counts.get(parent, 0) for parent in nodes[node]["par"]])
        parents_weights[node] = list(dirichlet.rvs(weights)[0])


def update_concentration(
    nodes: Nodes, groups: Groups, alpha: Alpha, a_0: float, b_0: float, sigma: float
) -> None:
    for node in nodes:
        samples = get_samples(node, nodes, groups)
        n_samples, n_tables = len(samples), int(len(np.unique(samples)))
        alpha_curr = alpha[node]
        # proposal: truncated normal on (0, 2*alpha_curr)
        a_trunc, b_trunc = 0.0, 2.0 * alpha_curr
        loc, scale = alpha_curr, alpha_curr
        a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
        from scipy.stats import truncnorm

        alpha_prop = float(truncnorm.rvs(a, b, loc=loc, scale=scale))
        a_rev, b_rev = (0.0 - alpha_prop) / alpha_prop, (2.0 * alpha_prop - alpha_prop) / alpha_prop
        p_prop = float(truncnorm.pdf(alpha_prop, a=a, b=b, loc=loc, scale=scale))
        p_rev = float(truncnorm.pdf(alpha_curr, a=a_rev, b=b_rev, loc=alpha_prop, scale=alpha_prop))

        mh_ratio = (
            (alpha_prop / alpha_curr) ** a_0
            * np.exp(-b_0 * (alpha_prop - alpha_curr))
            * poch(alpha_curr + 1, n_samples - 1)
            / poch(alpha_prop + 1, n_samples - 1)
            * np.prod(
                [(alpha_prop + sigma * i) / (alpha_curr + sigma * i) for i in range(1, n_tables)]
            )
            * p_rev
            / p_prop
        )
        if np.random.uniform() < min(1.0, float(mh_ratio)):
            alpha[node] = float(alpha_prop)


# ---------------- main engine ----------------


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
) -> dict[str, Any]:
    """
    Dimension-agnostic MCMC. All density/atom updates delegated to `model`.
    """
    assert nodes.keys() == x.keys()
    assert group_init_mode in ["diffuse", "concentrated"]

    n_obs = {node: len(x[node]) for node in nodes}
    if burn_in is None:
        burn_in = int(0.3 * n_iter)

    history: dict[str, list[Any]] = {
        "groups": [],
        "atoms": [],
        "parents_weights": [],
        "concentration": [],
        # model maintains its own kernel params; store useful summaries if needed
        "sigma_x": [],  # present for 1D parity; leave empty for 2D unless you add a getter
    }

    # init
    groups = initialize_groups(nodes, x, group_init_mode)
    alpha: Alpha = {node: float(np.random.gamma(shape=a_0, scale=1 / b_0)) for node in nodes}
    parents_weights: ParentsWeights = {
        node: list(dirichlet.rvs(np.ones(len(nodes[node]["par"])))[0])
        for node in nodes
        if len(nodes[node]["par"]) > 1
    }

    # atoms & maps from initial groups
    atom_labels = {g[0][1] for t in groups.values() for g in t}
    atoms: Atoms = {lab: 0.0 for lab in atom_labels}  # model will overwrite as needed
    tables_to_atoms: TablesToAtoms = {}
    tables_to_path: TablesToPath = {}
    for group in groups.values():
        for obs in group:
            for lvl in range(len(obs)):
                tables_to_atoms[obs[lvl]] = obs[0][1]
                tables_to_path[obs[lvl]] = obs[: lvl + 1]

    it_range = tqdm(range(n_iter), desc="Processing", unit="iter") if progress else range(n_iter)
    for _ in it_range:
        # table assignments
        for node in nodes:
            for i in range(n_obs[node]):
                new_path = sample_from_node(
                    node,
                    i=i,
                    x_obs=None,
                    cache=None,
                    nodes=nodes,
                    groups=groups,
                    alpha=alpha,
                    sigma=sigma,
                    atoms=atoms,
                    tables_to_atoms=tables_to_atoms,
                    parents_weights=parents_weights,
                    tables_to_path=tables_to_path,
                    x=x,
                    model=model,
                )
                # update maps
                for lvl in range(len(new_path)):
                    tables_to_atoms[new_path[lvl]] = new_path[0][1]
                    tables_to_path[new_path[lvl]] = new_path[: lvl + 1]
                groups[node][i] = new_path

        # model-specific parameter updates
        model.update_atoms(groups, x, tables_to_atoms, atoms)
        update_parent_weights(parents_weights, groups, alpha, nodes)
        update_concentration(nodes, groups, alpha, a_0, b_0, sigma)
        model.update_kernel(groups, x, atoms, tables_to_atoms)

        # record
        history["groups"].append({k: list(v) for k, v in groups.items()})
        history["atoms"].append(dict(atoms))
        history["parents_weights"].append(dict(parents_weights))
        history["concentration"].append(dict(alpha))
        # optional: for 1D store sigma_x if present
        if hasattr(model, "sigma_x"):
            history["sigma_x"].append(model.sigma_x)

    return {
        "history": history,
        "tables_to_atoms": tables_to_atoms,
        "params": {
            "burn_in": burn_in,
            # expose model-level priors only in wrappers (1D/2D) to keep engine generic
            "sigma": sigma,
            "a_0": a_0,
            "b_0": b_0,
        },
    }

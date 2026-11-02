"""
Sampler engine.

Same model, conditionals and output format as `reference.py`, organised for speed:

- the restaurant state is kept incrementally (`CRFState`): customers per table, tables per
  (node, cluster), parent and cluster of every table; adding or removing a customer is O(depth)
  instead of rebuilding the counts from every path;
- the likelihood of every observation under every atom is computed once per sweep in one
  vectorised call (`log_like_matrix`), and extended by one column when a new cluster opens;
- for each observation the predictive of every node is computed once (in topological order),
  so the DAG recursion costs one evaluation per node, not one per path to the roots;
- hyperparameter updates use vectorised sufficient statistics.

The seating probabilities are those of `reference.sample_from_node`: an existing table t at node j
gets (n_t - sigma) f(x | atom of t), a new one (alpha_j + sigma K_j) times the parents' predictive.
They are sampled in two stages (cluster, then table within the cluster), which gives the same
distribution over tables. Probabilities are rescaled per observation by the largest likelihood, so
far-away observations don't underflow.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln
from scipy.stats import dirichlet
from tqdm import tqdm

from GraphPY.sampler.kernels import ObservationModel
from GraphPY.sampler.reference import (  # noqa: F401  (re-exported for backward compatibility)
    concentration_ll,
    discount_ll,
    get_samples,
    get_tables_from_children,
    initialize_groups,
    likelihood_observation,
    sample_from_node,
    update_concentration,
    update_discount,
    update_parent_weights,
)
from GraphPY.types import (
    Alpha,
    Atoms,
    Groups,
    NodeId,
    Nodes,
    ParentsWeights,
    Path,
    TableKey,
    TablesToAtoms,
    TablesToPath,
    XData,
)

# PY discount must lie in [0, 1): sigma = 1 is not on the grid. Implicit prior: uniform on the grid
DISCOUNT_GRID = np.linspace(0, 1, 21)[:-1]


def _sample_index(weights: np.ndarray) -> int:
    # index drawn with probability proportional to non-negative weights
    c = np.cumsum(weights)
    return int(np.searchsorted(c, np.random.random() * c[-1], side="right"))


def _topological_order(nodes: Nodes) -> list[NodeId]:
    order: list[NodeId] = []
    seen: set[NodeId] = set()

    def visit(v: NodeId) -> None:
        if v in seen:
            return
        for p in nodes[v]["par"]:
            visit(p)
        seen.add(v)
        order.append(v)

    for v in nodes:
        visit(v)
    return order


class CRFState:
    """
    Chinese restaurant franchise state over a DAG.

    Tables have integer ids; clusters are root tables and live in "slots" (columns of the count
    matrices and of the likelihood matrix). Output keys and paths use the reference format:
    a table key is (node, label) and a path lists the keys from the root down to the table.
    """

    def __init__(self, nodes: Nodes, x: XData, model: ObservationModel):
        self.nodes = nodes
        self.model = model
        self.node_ids = list(nodes)
        self.idx = {v: j for j, v in enumerate(self.node_ids)}
        self.topo = [self.idx[v] for v in _topological_order(nodes)]
        self.parents = [[self.idx[p] for p in nodes[v]["par"]] for v in self.node_ids]
        n_nodes = len(self.node_ids)

        # observations, stacked in node order
        self.X = np.asarray([xi for v in self.node_ids for xi in x[v]], dtype=float)
        self.obs_node = np.concatenate(
            [np.full(len(x[v]), self.idx[v], dtype=int) for v in self.node_ids]
        )
        self.N = len(self.X)
        self.n_obs_node = [len(x[v]) for v in self.node_ids]
        self.obs_table = np.full(self.N, -1, dtype=int)

        # tables
        self.t_node: list[int] = []
        self.t_parent: list[int] = []
        self.t_cluster: list[int] = []
        self.t_count: list[int] = []
        self.t_key: list[TableKey] = []
        self.t_path: list[Path] = []
        self.free_tables: list[int] = []
        self.node_tables: list[set[int]] = [set() for _ in range(n_nodes)]
        self.tables_by_cluster: dict[tuple[int, int], list[int]] = {}
        self.next_label = [1] * n_nodes
        self.n_node = np.zeros(n_nodes)  # customers per restaurant
        self.k_node = np.zeros(n_nodes)  # tables per restaurant

        # clusters (= root tables) in slots
        self.cap = 0
        self.C = np.zeros((n_nodes, 0))  # customers at node j's tables of cluster k
        self.T = np.zeros((n_nodes, 0))  # tables at node j of cluster k
        self.atoms = np.zeros((0,) + self.X.shape[1:])
        self.slot_label: list[int | None] = []
        self.free_slots: list[int] = []
        self.LL = np.zeros((self.N, 0))  # log p(x_n | atom of cluster k)

        self.tables_to_atoms: TablesToAtoms = {}  # every table key ever created -> atom label

    # ---------------- construction / export ----------------

    @classmethod
    def from_paths(
        cls,
        nodes: Nodes,
        x: XData,
        model: ObservationModel,
        groups: Groups,
        atoms: Atoms,
        tables_to_atoms: TablesToAtoms | None = None,
    ) -> CRFState:
        """Build the state from reference-format seating paths and atoms (keyed by label)."""
        st = cls(nodes, x, model)
        table_of: dict[TableKey, int] = {}
        g = 0
        for v in st.node_ids:
            for path in groups[v]:
                parent = -1
                for key in path:
                    if key not in table_of:
                        if parent == -1:
                            label = (
                                tables_to_atoms[key] if tables_to_atoms is not None else hash(key)
                            )
                            k = st._new_cluster(atoms[label], label)
                        else:
                            k = st.t_cluster[parent]
                        t = st._new_table(st.idx[key[0]], parent, k, key=key)
                        if parent != -1:
                            st._add_customer(parent)
                        table_of[key] = t
                    parent = table_of[key]
                st._add_customer(parent)
                st.obs_table[g] = parent
                g += 1
        for key in st.t_key:
            j = st.idx[key[0]]
            st.next_label[j] = max(st.next_label[j], key[1] + 1)
        return st

    def groups(self) -> Groups:
        out: Groups = {}
        g = 0
        for v, n in zip(self.node_ids, self.n_obs_node, strict=True):
            out[v] = [self.t_path[t] for t in self.obs_table[g : g + n]]
            g += n
        return out

    def atoms_dict(self) -> Atoms:
        out: Atoms = {}
        for k, label in enumerate(self.slot_label):
            if label is not None:
                atom = self.atoms[k]
                out[label] = float(atom) if np.ndim(atom) == 0 else np.array(atom)
        return out

    def live_slots(self) -> np.ndarray:
        return np.array([k for k, lab in enumerate(self.slot_label) if lab is not None], dtype=int)

    # ---------------- low-level bookkeeping ----------------

    def _grow(self) -> None:
        new_cap = max(8, 2 * self.cap)
        extra = new_cap - self.cap
        n_nodes = len(self.node_ids)
        self.C = np.hstack([self.C, np.zeros((n_nodes, extra))])
        self.T = np.hstack([self.T, np.zeros((n_nodes, extra))])
        self.atoms = np.concatenate([self.atoms, np.zeros((extra,) + self.atoms.shape[1:])])
        self.LL = np.hstack([self.LL, np.full((self.N, extra), -np.inf)])
        self.slot_label.extend([None] * extra)
        self.free_slots.extend(range(new_cap - 1, self.cap - 1, -1))
        self.cap = new_cap

    def _new_cluster(self, atom: Any, label: int | None = None) -> int:
        if not self.free_slots:
            self._grow()
        k = self.free_slots.pop()
        self.atoms[k] = atom
        self.slot_label[k] = label  # set by _new_table for new root tables when None
        self.LL[:, k] = self.model.log_like_matrix(self.X, self.atoms[k : k + 1])[:, 0]
        return k

    def _new_table(self, j: int, parent: int, k: int, key: TableKey | None = None) -> int:
        if key is None:
            key = (self.node_ids[j], self.next_label[j])
            self.next_label[j] += 1
        path = (key,) if parent == -1 else self.t_path[parent] + (key,)
        if self.free_tables:
            t = self.free_tables.pop()
            self.t_node[t], self.t_parent[t], self.t_cluster[t] = j, parent, k
            self.t_count[t], self.t_key[t], self.t_path[t] = 0, key, path
        else:
            t = len(self.t_node)
            self.t_node.append(j)
            self.t_parent.append(parent)
            self.t_cluster.append(k)
            self.t_count.append(0)
            self.t_key.append(key)
            self.t_path.append(path)
        if parent == -1 and self.slot_label[k] is None:
            self.slot_label[k] = hash(key)
        self.node_tables[j].add(t)
        self.tables_by_cluster.setdefault((j, k), []).append(t)
        self.T[j, k] += 1
        self.k_node[j] += 1
        label = self.slot_label[k]
        assert label is not None
        self.tables_to_atoms[key] = label
        return t

    def _add_customer(self, t: int) -> None:
        j, k = self.t_node[t], self.t_cluster[t]
        self.t_count[t] += 1
        self.C[j, k] += 1
        self.n_node[j] += 1

    def _remove_customer(self, t: int) -> None:
        # one customer leaves table t; empty tables are deleted, recursively up to the root
        while True:
            j, k = self.t_node[t], self.t_cluster[t]
            self.t_count[t] -= 1
            self.C[j, k] -= 1
            self.n_node[j] -= 1
            if self.t_count[t] > 0:
                return
            self.T[j, k] -= 1
            self.k_node[j] -= 1
            self.node_tables[j].discard(t)
            same = self.tables_by_cluster[(j, k)]
            same.remove(t)
            if not same:
                del self.tables_by_cluster[(j, k)]
            parent = self.t_parent[t]
            self.free_tables.append(t)
            if parent == -1:  # a root table: its cluster is gone
                self.slot_label[k] = None
                self.LL[:, k] = -np.inf
                self.free_slots.append(k)
                return
            t = parent

    # ---------------- seating ----------------

    def reseat(
        self,
        g: int,
        alpha: np.ndarray,
        sigma: float,
        weights: list[np.ndarray],
        log_new_root: float,
    ) -> None:
        """Gibbs update of the seating of observation g (removed first, then re-seated)."""
        self._remove_customer(self.obs_table[g])

        ll = self.LL[g]
        m = max(float(np.max(ll)) if self.cap else -np.inf, log_new_root)
        lik = np.exp(ll - m)  # f(x | atom_k), rescaled; 0 for empty slots
        lik_root = np.exp(log_new_root - m)

        W = self.C - sigma * self.T  # sum over a node's tables of cluster k of (n_t - sigma)
        A = W @ lik
        new_w = alpha + sigma * self.k_node  # weight of a new table, per node
        denom = alpha + self.n_node
        pred = np.zeros(len(self.node_ids))
        like_new = np.zeros(len(self.node_ids))
        for j in self.topo:
            par = self.parents[j]
            if not par:
                like_new[j] = lik_root
            elif len(par) == 1:
                like_new[j] = pred[par[0]]
            else:
                like_new[j] = float(weights[j] @ pred[par])
            pred[j] = (A[j] + new_w[j] * like_new[j]) / denom[j]

        # walk up from the observation's node while new tables are opened
        j = int(self.obs_node[g])
        chain: list[int] = []
        while True:
            w = np.append(W[j] * lik, new_w[j] * like_new[j])
            k = _sample_index(w)
            if k < self.cap:
                tables = self.tables_by_cluster[(j, k)]
                if len(tables) == 1:
                    top = tables[0]
                else:
                    counts = np.array([self.t_count[t] for t in tables], dtype=float)
                    top = tables[_sample_index(counts - sigma)]
                break
            chain.append(j)
            par = self.parents[j]
            if not par:
                k = self._new_cluster(self.model.sample_root_atom(self.X[g]))
                top = -1
                break
            j = par[0] if len(par) == 1 else par[_sample_index(weights[j] * pred[par])]

        # open the new tables from the top down; each is one customer of the table above
        for j in reversed(chain):
            t = self._new_table(j, top, k)
            if top != -1:
                self._add_customer(top)
            top = t
        self._add_customer(top)
        self.obs_table[g] = top

    # ---------------- hyperparameter updates ----------------

    def obs_slots(self) -> np.ndarray:
        return np.asarray(np.asarray(self.t_cluster, dtype=int)[self.obs_table])

    def table_counts(self, j: int) -> np.ndarray:
        return np.array([self.t_count[t] for t in self.node_tables[j]], dtype=float)

    def update_atoms(self) -> None:
        live = self.live_slots()
        z = self.obs_slots()
        n_k = np.bincount(z, minlength=self.cap).astype(float)[live]
        sum_k = np.zeros((self.cap,) + self.X.shape[1:])
        np.add.at(sum_k, z, self.X)
        self.atoms[live] = self.model.sample_atoms(n_k, sum_k[live])

    def refresh_likelihoods(self) -> None:
        live = self.live_slots()
        self.LL[:] = -np.inf
        if len(live):
            self.LL[:, live] = self.model.log_like_matrix(self.X, self.atoms[live])

    def parent_table_counts(self, j: int) -> np.ndarray:
        # tables at node j by the parent node they sit in (order of nodes[v]["par"])
        counts = np.zeros(len(self.parents[j]))
        pos = {p: i for i, p in enumerate(self.parents[j])}
        for t in self.node_tables[j]:
            counts[pos[self.t_node[self.t_parent[t]]]] += 1
        return counts


# ---------------- concentration / discount (vectorised versions of reference.py) ----------------


def _log_eppf_alpha(alpha: float, sigma: float, counts: np.ndarray) -> float:
    # terms of the PY EPPF that depend on alpha (same as reference.concentration_ll)
    n, k = float(np.sum(counts)), len(counts)
    if n == 0:  # empty restaurant
        return 0.0
    return float(
        np.sum(np.log(alpha + sigma * np.arange(1, k))) - (gammaln(alpha + n) - gammaln(alpha + 1))
    )


def _log_eppf_sigma(alpha: float, sigma_grid: np.ndarray, counts: np.ndarray) -> np.ndarray:
    # terms of the PY EPPF that depend on sigma, on a grid (same as reference.discount_ll)
    k = len(counts)
    i = np.arange(1, k)
    first = np.sum(np.log(alpha + np.outer(sigma_grid, i)), axis=1)
    second = np.sum(
        gammaln(counts[None, :] - sigma_grid[:, None]) - gammaln(1 - sigma_grid[:, None]), axis=1
    )
    return np.asarray(first + second)


def _update_concentration(st: CRFState, alpha: Alpha, a_0: float, b_0: float, sigma: float) -> None:
    for j, v in enumerate(st.node_ids):
        counts = st.table_counts(j)
        alpha_curr = alpha[v]
        alpha_prop = float(np.random.normal(alpha_curr))  # random-walk proposal, sd 1
        if alpha_prop > 0:
            log_ratio = (
                (a_0 - 1) * np.log(alpha_prop / alpha_curr)
                - b_0 * (alpha_prop - alpha_curr)
                + _log_eppf_alpha(alpha_prop, sigma, counts)
                - _log_eppf_alpha(alpha_curr, sigma, counts)
            )
            if np.log(np.random.uniform()) < min(0.0, log_ratio):
                alpha[v] = alpha_prop


def _update_discount(st: CRFState, alpha: Alpha) -> float:
    log_p = np.zeros(len(DISCOUNT_GRID))
    for j, v in enumerate(st.node_ids):
        log_p += _log_eppf_sigma(alpha[v], DISCOUNT_GRID, st.table_counts(j))
    p = np.exp(log_p - np.max(log_p))
    return float(DISCOUNT_GRID[_sample_index(p)])


def _update_parent_weights(st: CRFState, parents_weights: ParentsWeights) -> None:
    for v in parents_weights:
        j = st.idx[v]
        parents_weights[v] = list(dirichlet.rvs(1 + st.parent_table_counts(j))[0])


def _sweep(
    st: CRFState,
    *,
    alpha: Alpha,
    sigma: float,
    parents_weights: ParentsWeights,
    a_0: float,
    b_0: float,
    freeze_sigma: bool,
) -> float:
    model = st.model
    alpha_arr = np.array([alpha[v] for v in st.node_ids], dtype=float)
    weights = [
        np.asarray(parents_weights[v], dtype=float) if v in parents_weights else np.ones(1)
        for v in st.node_ids
    ]

    # table assignments (hyperparameters are fixed during the seating phase)
    st.refresh_likelihoods()
    log_new_root = model.log_like_new_root_all(st.X)
    for g in range(st.N):
        st.reseat(g, alpha_arr, sigma, weights, float(log_new_root[g]))

    # model-specific parameter updates (same order as reference.gibbs_sweep)
    st.update_atoms()
    model.update_base_from_atoms(st.atoms[st.live_slots()])
    model.update_kernel_from_resid(st.X - st.atoms[st.obs_slots()])

    _update_parent_weights(st, parents_weights)
    _update_concentration(st, alpha, a_0, b_0, sigma)
    if not freeze_sigma:
        sigma = _update_discount(st, alpha)
    return sigma


# ---------------- public API ----------------


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
    One sweep of the sampler on a reference-format state; updates it in place and returns the
    (new) discount. Converts to and from `CRFState`, so `run_sampler` is faster for many sweeps.
    """
    st = CRFState.from_paths(nodes, x, model, groups, atoms, tables_to_atoms)
    sigma = _sweep(
        st,
        alpha=alpha,
        sigma=sigma,
        parents_weights=parents_weights,
        a_0=a_0,
        b_0=b_0,
        freeze_sigma=freeze_sigma,
    )
    new_groups = st.groups()
    for v in nodes:
        groups[v][:] = new_groups[v]
    atoms.clear()
    atoms.update(st.atoms_dict())
    tables_to_atoms.update(st.tables_to_atoms)
    for group in new_groups.values():
        for path in group:
            for lvl in range(len(path)):
                tables_to_path[path[lvl]] = path[: lvl + 1]
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

    # init (same as reference.run_sampler)
    groups = initialize_groups(nodes, x, group_init_mode)
    alpha: Alpha = {node: float(np.random.gamma(shape=a_0, scale=1 / b_0)) for node in nodes}
    parents_weights: ParentsWeights = {
        node: list(dirichlet.rvs(np.ones(len(nodes[node]["par"])))[0])
        for node in nodes
        if len(nodes[node]["par"]) > 1
    }
    zero = 0 * np.array([xi for arr in x.values() for xi in arr], dtype=float)[0]
    atoms: Atoms = {hash(g[0]): zero for t in groups.values() for g in t}
    st = CRFState.from_paths(nodes, x, model, groups, atoms)

    history: dict[str, list[Any]] = {
        "groups": [st.groups()],
        "atoms": [st.atoms_dict()],
        "parents_weights": [dict(parents_weights)],
        "concentration": [dict(alpha)],
        "sigma": [sigma],
    }

    it_range = tqdm(range(n_iter), desc="Processing", unit="iter") if progress else range(n_iter)
    for _ in it_range:
        sigma = _sweep(
            st,
            alpha=alpha,
            sigma=sigma,
            parents_weights=parents_weights,
            a_0=a_0,
            b_0=b_0,
            freeze_sigma=freeze_sigma,
        )
        history["groups"].append(st.groups())
        history["atoms"].append(st.atoms_dict())
        history["parents_weights"].append(dict(parents_weights))
        history["concentration"].append(dict(alpha))
        history["sigma"].append(sigma)

    return {
        "history": history,
        "tables_to_atoms": st.tables_to_atoms,
        "model": model,
        "params": {
            "burn_in": burn_in,
            "a_0": a_0,
            "b_0": b_0,
        },
    }

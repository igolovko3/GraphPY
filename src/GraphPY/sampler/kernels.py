from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.stats import invgamma, invwishart, multivariate_normal, norm

from GraphPY.types import Atoms, Groups, TablesToAtoms, XData

LOG_2PI = float(np.log(2 * np.pi))


def _members_by_atom(
    groups: Groups, x: XData, tables_to_atoms: TablesToAtoms
) -> tuple[list[int], np.ndarray, np.ndarray]:
    # atom labels (in order of first appearance), observations, and atom index of each observation
    all_tables = [obs for group in groups.values() for obs in group]
    labels = list(dict.fromkeys(tables_to_atoms[g[0]] for g in all_tables))
    index = {lab: k for k, lab in enumerate(labels)}
    X = np.asarray([xi for arr in x.values() for xi in arr], dtype=float)
    z = np.array([index[tables_to_atoms[g[0]]] for g in all_tables], dtype=int)
    return labels, X, z


def _sums_by_atom(X: np.ndarray, z: np.ndarray, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    n_k = np.bincount(z, minlength=n_atoms).astype(float)
    sum_k = np.zeros((n_atoms,) + X.shape[1:])
    np.add.at(sum_k, z, X)
    return n_k, sum_k


class ObservationModel(ABC):
    """
    Dimension-specific math for likelihoods, atom updates, and kernel updates.
    Engine calls these; everything else (CRP/paths/alphas) is dimension-agnostic.
    """

    param_history: dict[str, Any]

    # existing-table likelihood p(x | atom)  (vectorized x allowed)
    @abstractmethod
    def like_existing(
        self, x: float | np.ndarray, atom: float | np.ndarray, **kwargs: Any
    ) -> float:
        pass

    # new-table prior predictive at ROOT
    @abstractmethod
    def like_new_root(self, x: float | np.ndarray, **kwargs: Any) -> float:
        pass

    # sample atom for NEW ROOT table given x_obs
    @abstractmethod
    def sample_root_atom(self, x_obs: float | np.ndarray) -> float | np.ndarray:
        pass

    # update all atoms in place given current assignments
    @abstractmethod
    def update_atoms(
        self, groups: Groups, x: XData, tables_to_atoms: TablesToAtoms, atoms: Atoms
    ) -> None:
        pass

    # update kernel hyperparameters in place (e.g., sigma_x / Sigma_x)
    @abstractmethod
    def update_kernel(
        self, groups: Groups, x: XData, atoms: Atoms, tables_to_atoms: TablesToAtoms, **kwargs: Any
    ) -> None:
        pass

    @abstractmethod
    def update_base_distribution(
        self, groups: Groups, x: XData, atoms: Atoms, tables_to_atoms: TablesToAtoms, **kwargs: Any
    ) -> None:
        pass

    # ---- array-based interface (used by the engine) ----
    # X: observations stacked along axis 0; atoms: atom values stacked along axis 0

    # log p(X[n] | atoms[k]) for every pair: shape (N, K)
    @abstractmethod
    def log_like_matrix(self, X: np.ndarray, atoms: np.ndarray) -> np.ndarray:
        pass

    # log of the new-table prior predictive at a ROOT for every observation: shape (N,)
    @abstractmethod
    def log_like_new_root_all(self, X: np.ndarray) -> np.ndarray:
        pass

    # draw atoms from their conjugate posteriors given per-atom counts and sums of observations
    @abstractmethod
    def sample_atoms(self, n_k: np.ndarray, sum_k: np.ndarray) -> np.ndarray:
        pass

    # update base-distribution hyperparameters given the distinct atoms
    @abstractmethod
    def update_base_from_atoms(self, unique_atoms: np.ndarray) -> None:
        pass

    # update kernel hyperparameters given residuals x - atom of every observation
    @abstractmethod
    def update_kernel_from_resid(self, resid: np.ndarray) -> None:
        pass

    def posterior_parameters(self, res: dict[str, Any]) -> dict[str, Any]:

        burn_in = res["params"]["burn_in"]
        model_params: dict[str, Any] = {
            param: np.mean(samples[burn_in:], axis=0)
            for param, samples in self.param_history.items()
        }

        return model_params


# ---------- 1D Gaussian ----------
class Gaussian1DModel(ObservationModel):
    def __init__(
        self,
        mu_base: float = 0.0,
        lambda_base: float = 1.0,
        alpha_base: float = 0.5,
        beta_base: float = 10,
        alpha_kernel: float = 0.5,
        beta_kernel: float = 2.5,
        random_init: bool = False,
    ):
        self._mu_base = float(mu_base)
        self._lambda_base = float(lambda_base)
        self._alpha_base = float(alpha_base)
        self._beta_base = float(beta_base)
        self._alpha_kernel = float(alpha_kernel)
        self._beta_kernel = float(beta_kernel)

        if not random_init:
            self.sigma_phi = np.sqrt(self._beta_base / (self._alpha_base + 1))
            self.mu_phi = self._mu_base
            self.sigma_x = np.sqrt(self._beta_kernel / (self._alpha_kernel + 1))
        else:
            self.sigma_phi = np.sqrt(invgamma.rvs(a=self._alpha_base, scale=self._beta_base))
            self.mu_phi = norm.rvs(
                loc=self._mu_base, scale=self.sigma_phi / np.sqrt(self._lambda_base)
            )
            self.sigma_x = np.sqrt(invgamma.rvs(a=self._alpha_kernel, scale=self._beta_kernel))

        self.param_history = {
            "sigma_phi": [self.sigma_phi],
            "mu_phi": [self.mu_phi],
            "sigma_x": [self.sigma_x],
        }

    def like_existing(
        self,
        x: float | np.ndarray,
        atom: float | np.ndarray,
        *,
        sigma_x: float | None = None,
        **kwargs: Any,
    ) -> float:
        if sigma_x is None:
            sigma_x = self.sigma_x
        like: float = norm.pdf(x, loc=atom, scale=sigma_x)
        return like

    def like_new_root(
        self,
        x: float | np.ndarray,
        *,
        mu_phi: float | None = None,
        sigma_phi: float | None = None,
        sigma_x: float | None = None,
        **kwargs: Any,
    ) -> float:
        if mu_phi is None:
            mu_phi = self.mu_phi
        if sigma_phi is None:
            sigma_phi = self.sigma_phi
        if sigma_x is None:
            sigma_x = self.sigma_x

        like: float = norm.pdf(x, loc=mu_phi, scale=np.sqrt(sigma_phi**2 + sigma_x**2))
        return like

    def sample_root_atom(self, x_obs: float | np.ndarray) -> float:
        # posterior: N( (x/σx² + μ/σφ²) / (1/σx² + 1/σφ²),  1 / (1/σx² + 1/σφ²) )
        std = 1.0 / np.sqrt(1.0 / self.sigma_phi**2 + 1.0 / self.sigma_x**2)
        mean = (x_obs / self.sigma_x**2 + self.mu_phi / self.sigma_phi**2) * std**2
        return float(np.random.normal(mean, std))

    def update_atoms(
        self, groups: Groups, x: XData, tables_to_atoms: TablesToAtoms, atoms: Atoms
    ) -> None:
        labels, X, z = _members_by_atom(groups, x, tables_to_atoms)
        new_atoms = self.sample_atoms(*_sums_by_atom(X, z, len(labels)))
        for lab, atom in zip(labels, new_atoms, strict=True):
            atoms[lab] = float(atom)

    # ---- array-based interface ----
    def log_like_matrix(self, X: np.ndarray, atoms: np.ndarray) -> np.ndarray:
        d = (np.asarray(X, dtype=float)[:, None] - np.asarray(atoms, dtype=float)[None, :]) / (
            self.sigma_x
        )
        return np.asarray(-0.5 * d**2 - np.log(self.sigma_x) - 0.5 * LOG_2PI)

    def log_like_new_root_all(self, X: np.ndarray) -> np.ndarray:
        s = np.sqrt(self.sigma_phi**2 + self.sigma_x**2)
        d = (np.asarray(X, dtype=float) - self.mu_phi) / s
        return np.asarray(-0.5 * d**2 - np.log(s) - 0.5 * LOG_2PI)

    def sample_atoms(self, n_k: np.ndarray, sum_k: np.ndarray) -> np.ndarray:
        std = 1.0 / np.sqrt(1.0 / self.sigma_phi**2 + n_k / self.sigma_x**2)
        mean = (sum_k / self.sigma_x**2 + self.mu_phi / self.sigma_phi**2) * std**2
        return np.asarray(np.random.normal(mean, std))

    def update_kernel(
        self, groups: Groups, x: XData, atoms: Atoms, tables_to_atoms: TablesToAtoms, **kwargs: Any
    ) -> None:
        all_tables = [obs for group in groups.values() for obs in group]
        all_data = np.array([xi for arr in x.values() for xi in arr], dtype=float)
        all_atoms = np.array([atoms[tables_to_atoms[g[0]]] for g in all_tables], dtype=float)
        self.update_kernel_from_resid(all_data - all_atoms)

    def update_kernel_from_resid(self, resid: np.ndarray) -> None:
        alpha_post = self._alpha_kernel + len(resid) / 2
        beta_post = self._beta_kernel + np.sum(resid**2) / 2
        self.sigma_x = np.sqrt(invgamma.rvs(a=alpha_post, scale=beta_post))
        self.param_history["sigma_x"].append(self.sigma_x)

    def update_base_distribution(
        self, groups: Groups, x: XData, atoms: Atoms, tables_to_atoms: TablesToAtoms, **kwargs: Any
    ) -> None:
        unique_tables = set([obs[0] for group in groups.values() for obs in group])
        unique_atoms = np.array([atoms[tables_to_atoms[g]] for g in unique_tables], dtype=float)
        self.update_base_from_atoms(unique_atoms)

    def update_base_from_atoms(self, unique_atoms: np.ndarray) -> None:
        unique_atoms = np.asarray(unique_atoms, dtype=float)
        mean_atoms = np.mean(unique_atoms)
        n_atoms = len(unique_atoms)

        alpha_post = self._alpha_base + n_atoms / 2
        beta_post = (
            self._beta_base
            + np.sum((unique_atoms - mean_atoms) ** 2) / 2
            + (n_atoms * self._lambda_base * (mean_atoms - self._mu_base) ** 2)
            / (2 * (n_atoms + self._lambda_base))
        )
        mu_post = (self._mu_base * self._lambda_base + n_atoms * mean_atoms) / (
            self._lambda_base + n_atoms
        )
        lambda_post = self._lambda_base + n_atoms

        self.sigma_phi = np.sqrt(invgamma.rvs(a=alpha_post, scale=beta_post))
        self.mu_phi = norm.rvs(loc=mu_post, scale=self.sigma_phi / np.sqrt(lambda_post))
        self.param_history["sigma_phi"].append(self.sigma_phi)
        self.param_history["mu_phi"].append(self.mu_phi)


class MultivariateGaussianModel(ObservationModel):
    def __init__(
        self,
        n_dim: int = 2,
        mu_base: np.ndarray | None = None,
        lambda_base: float = 1.0,
        nu_base: float | None = None,
        psi_base: np.ndarray | None = None,
        nu_kernel: float | None = None,
        psi_kernel: np.ndarray | None = None,
        random_init: bool = False,
    ):

        self._n_dim = int(n_dim)

        self._mu_base = np.zeros(self._n_dim) if mu_base is None else mu_base
        self._lambda_base = lambda_base
        self._nu_base = float(self._n_dim) if nu_base is None else nu_base
        self._psi_base = (
            4 * (2 * self._n_dim + 1) * np.eye(self._n_dim) if psi_base is None else psi_base
        )
        self._nu_kernel = float(self._n_dim) if nu_kernel is None else nu_kernel
        self._psi_kernel = (
            (2 * self._n_dim + 1) * np.eye(self._n_dim) if psi_kernel is None else psi_kernel
        )

        if not random_init:
            self.Sigma_phi = self._psi_base / (self._nu_base + self._n_dim + 1)
            self.mu_phi = self._mu_base
            self.Sigma_x = self._psi_kernel / (self._nu_kernel + self._n_dim + 1)
        else:
            self.Sigma_phi = invwishart.rvs(df=self._nu_base, scale=self._psi_base)
            self.mu_phi = multivariate_normal.rvs(
                mean=self._mu_base, cov=self.Sigma_phi / self._lambda_base
            )
            self.Sigma_x = invwishart.rvs(df=self._nu_kernel, scale=self._psi_kernel)

        self._Sigma_phi_inv = np.linalg.inv(self.Sigma_phi)
        self._Sigma_x_inv = np.linalg.inv(self.Sigma_x)

        self.param_history = {
            "Sigma_phi": [np.array(self.Sigma_phi)],
            "mu_phi": [np.array(self.mu_phi)],
            "Sigma_x": [np.array(self.Sigma_x)],
        }

    def like_existing(
        self,
        x: float | np.ndarray,
        atom: float | np.ndarray,
        *,
        Sigma_x: np.ndarray | None = None,
        **kwargs: Any,
    ) -> float:
        if Sigma_x is None:
            Sigma_x = self.Sigma_x
        like: float = multivariate_normal.pdf(x, mean=np.asarray(atom), cov=Sigma_x)
        return like

    def like_new_root(
        self,
        x: float | np.ndarray,
        *,
        mu_phi: np.ndarray | None = None,
        Sigma_phi: np.ndarray | None = None,
        Sigma_x: np.ndarray | None = None,
        **kwargs: Any,
    ) -> float:
        if mu_phi is None:
            mu_phi = self.mu_phi
        if Sigma_phi is None:
            Sigma_phi = self.Sigma_phi
        if Sigma_x is None:
            Sigma_x = self.Sigma_x
        like: float = multivariate_normal.pdf(x, mean=mu_phi, cov=Sigma_phi + Sigma_x)
        return like

    def sample_root_atom(self, x_obs: float | np.ndarray) -> np.ndarray:
        Sigma_post = np.linalg.inv(self._Sigma_phi_inv + self._Sigma_x_inv)
        mu_post = Sigma_post @ (self._Sigma_x_inv @ x_obs + self._Sigma_phi_inv @ self.mu_phi)
        return np.random.multivariate_normal(mu_post, Sigma_post)

    def update_atoms(
        self, groups: Groups, x: XData, tables_to_atoms: TablesToAtoms, atoms: Atoms
    ) -> None:
        labels, X, z = _members_by_atom(groups, x, tables_to_atoms)
        new_atoms = self.sample_atoms(*_sums_by_atom(X, z, len(labels)))
        for lab, atom in zip(labels, new_atoms, strict=True):
            atoms[lab] = atom

    # ---- array-based interface ----
    def _chol_inv_x(self) -> tuple[np.ndarray, float]:
        # inverse Cholesky factor and log-determinant of Sigma_x, recomputed only when replaced
        if getattr(self, "_chol_x_src", None) is not self.Sigma_x:
            self._chol_x_cache = self._chol_inv(self.Sigma_x)
            self._chol_x_src = self.Sigma_x
        return self._chol_x_cache

    @staticmethod
    def _chol_inv(cov: np.ndarray) -> tuple[np.ndarray, float]:
        chol = np.linalg.cholesky(cov)
        return np.linalg.inv(chol), float(2.0 * np.sum(np.log(np.diag(chol))))

    @staticmethod
    def _logpdf_rows(diff: np.ndarray, chol_inv: np.ndarray, log_det: float) -> np.ndarray:
        # log N(diff; 0, Sigma) along the last axis, with Sigma^-1 = chol_inv^T chol_inv
        sol = diff @ chol_inv.T
        d = chol_inv.shape[0]
        return np.asarray(-0.5 * (np.sum(sol**2, axis=-1) + log_det + d * LOG_2PI))

    def log_like_matrix(self, X: np.ndarray, atoms: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(-1, self._n_dim)
        atoms = np.asarray(atoms, dtype=float).reshape(-1, self._n_dim)
        return self._logpdf_rows(X[:, None, :] - atoms[None, :, :], *self._chol_inv_x())

    def log_like_new_root_all(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(-1, self._n_dim)
        return self._logpdf_rows(
            X - np.asarray(self.mu_phi), *self._chol_inv(self.Sigma_phi + self.Sigma_x)
        )

    def sample_atoms(self, n_k: np.ndarray, sum_k: np.ndarray) -> np.ndarray:
        out = np.empty((len(n_k), self._n_dim))
        prior_term = self._Sigma_phi_inv @ self.mu_phi
        for k in range(len(n_k)):
            Sigma_post = np.linalg.inv(self._Sigma_phi_inv + n_k[k] * self._Sigma_x_inv)
            mu_post = Sigma_post @ (self._Sigma_x_inv @ sum_k[k] + prior_term)
            out[k] = np.random.multivariate_normal(mu_post, Sigma_post)
        return out

    def update_kernel(
        self, groups: Groups, x: XData, atoms: Atoms, tables_to_atoms: TablesToAtoms, **kwargs: Any
    ) -> None:
        all_tables = [obs for group in groups.values() for obs in group]
        all_data = np.vstack([np.asarray(xi) for arr in x.values() for xi in arr])  # (N,2)
        all_atoms = np.vstack(
            [np.asarray(atoms[tables_to_atoms[g[0]]]) for g in all_tables]
        )  # (N,2)
        self.update_kernel_from_resid(all_data - all_atoms)

    def update_kernel_from_resid(self, resid: np.ndarray) -> None:
        resid = np.asarray(resid, dtype=float).reshape(-1, self._n_dim)
        nu_post = self._nu_kernel + len(resid)
        psi_post = self._psi_kernel + resid.T @ resid

        self.Sigma_x = invwishart.rvs(df=nu_post, scale=psi_post)
        self._Sigma_x_inv = np.linalg.inv(self.Sigma_x)

        self.param_history["Sigma_x"].append(np.array(self.Sigma_x))

    def update_base_distribution(
        self, groups: Groups, x: XData, atoms: Atoms, tables_to_atoms: TablesToAtoms, **kwargs: Any
    ) -> None:
        unique_tables = set([obs[0] for group in groups.values() for obs in group])
        unique_atoms = np.array([atoms[tables_to_atoms[g]] for g in unique_tables], dtype=float)
        self.update_base_from_atoms(unique_atoms)

    def update_base_from_atoms(self, unique_atoms: np.ndarray) -> None:
        unique_atoms = np.asarray(unique_atoms, dtype=float).reshape(-1, self._n_dim)
        mean_atoms = np.mean(unique_atoms, axis=0)
        n_atoms = unique_atoms.shape[0]

        nu_post = self._nu_base + n_atoms
        psi_post = (
            self._psi_base
            + np.cov(unique_atoms.T, bias=True) * n_atoms
            + self._lambda_base
            * n_atoms
            * np.outer(mean_atoms - self._mu_base, mean_atoms - self._mu_base)
            / (self._lambda_base + n_atoms)
        )
        mu_post = (self._mu_base * self._lambda_base + n_atoms * mean_atoms) / (
            self._lambda_base + n_atoms
        )
        lambda_post = self._lambda_base + n_atoms

        self.Sigma_phi = invwishart.rvs(df=nu_post, scale=psi_post)
        self.mu_phi = multivariate_normal.rvs(mean=mu_post, cov=self.Sigma_phi / lambda_post)
        self._Sigma_phi_inv = np.linalg.inv(self.Sigma_phi)

        self.param_history["Sigma_phi"].append(self.Sigma_phi)
        self.param_history["mu_phi"].append(self.mu_phi)

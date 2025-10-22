from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.stats import invgamma, invwishart, multivariate_normal, norm

from GraphPY.types import Atoms, Groups, TablesToAtoms, XData


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
        all_tables = [obs for group in groups.values() for obs in group]
        all_data = [xi for arr in x.values() for xi in arr]
        curr_atoms = {tables_to_atoms[g[0]] for g in all_tables}
        for at in curr_atoms:
            members = [
                all_data[i] for i in range(len(all_data)) if tables_to_atoms[all_tables[i][0]] == at
            ]
            std = 1.0 / np.sqrt(1.0 / self.sigma_phi**2 + len(members) / self.sigma_x**2)
            mean = (sum(members) / self.sigma_x**2 + self.mu_phi / self.sigma_phi**2) * std**2
            atoms[at] = float(np.random.normal(mean, std))

    def update_kernel(
        self, groups: Groups, x: XData, atoms: Atoms, tables_to_atoms: TablesToAtoms, **kwargs: Any
    ) -> None:
        all_tables = [obs for group in groups.values() for obs in group]
        all_data = np.array([xi for arr in x.values() for xi in arr], dtype=float)
        all_atoms = np.array([atoms[tables_to_atoms[g[0]]] for g in all_tables], dtype=float)

        alpha_post = self._alpha_kernel + len(all_data) / 2
        beta_post = self._beta_kernel + np.sum((all_data - all_atoms) ** 2) / 2
        self.sigma_x = np.sqrt(invgamma.rvs(a=alpha_post, scale=beta_post))
        self.param_history["sigma_x"].append(self.sigma_x)

    def update_base_distribution(
        self, groups: Groups, x: XData, atoms: Atoms, tables_to_atoms: TablesToAtoms, **kwargs: Any
    ) -> None:
        unique_tables = set([obs[0] for group in groups.values() for obs in group])
        unique_atoms = np.array([atoms[tables_to_atoms[g]] for g in unique_tables], dtype=float)
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


# ---------- 2D Gaussian ----------
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
        all_tables = [obs for group in groups.values() for obs in group]
        all_data = [xi for arr in x.values() for xi in arr]  # each xi is (2,)
        curr_atoms = {tables_to_atoms[g[0]] for g in all_tables}
        for at in curr_atoms:
            members = [
                np.asarray(all_data[i])
                for i in range(len(all_data))
                if tables_to_atoms[all_tables[i][0]] == at
            ]
            n = len(members)
            if n == 0:
                continue
            Sigma_post = np.linalg.inv(self._Sigma_phi_inv + n * self._Sigma_x_inv)
            mu_post = Sigma_post @ (
                self._Sigma_x_inv @ np.sum(members, axis=0) + self._Sigma_phi_inv @ self.mu_phi
            )
            atoms[at] = np.random.multivariate_normal(mu_post, Sigma_post)

    def update_kernel(
        self, groups: Groups, x: XData, atoms: Atoms, tables_to_atoms: TablesToAtoms, **kwargs: Any
    ) -> None:
        all_tables = [obs for group in groups.values() for obs in group]
        all_data = np.vstack([np.asarray(xi) for arr in x.values() for xi in arr])  # (N,2)
        all_atoms = np.vstack(
            [np.asarray(atoms[tables_to_atoms[g[0]]]) for g in all_tables]
        )  # (N,2)
        resid = all_data - all_atoms
        n_obs = len(resid)

        nu_post = self._nu_kernel + n_obs
        psi_post = self._psi_kernel + resid.T @ resid

        self.Sigma_x = invwishart.rvs(df=nu_post, scale=psi_post)
        self._Sigma_x_inv = np.linalg.inv(self.Sigma_x)

        self.param_history["Sigma_x"].append(np.array(self.Sigma_x))

    def update_base_distribution(
        self, groups: Groups, x: XData, atoms: Atoms, tables_to_atoms: TablesToAtoms, **kwargs: Any
    ) -> None:
        unique_tables = set([obs[0] for group in groups.values() for obs in group])
        unique_atoms = np.array([atoms[tables_to_atoms[g]] for g in unique_tables], dtype=float)
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
        self.param_history["Sigma_phi"].append(self.Sigma_phi)
        self.param_history["mu_phi"].append(self.mu_phi)

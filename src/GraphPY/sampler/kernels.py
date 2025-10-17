from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.stats import multivariate_normal, norm

from GraphPY.types import Atoms, Groups, TablesToAtoms, XData


class ObservationModel(Protocol):
    """
    Dimension-specific math for likelihoods, atom updates, and kernel updates.
    Engine calls these; everything else (CRP/paths/alphas) is dimension-agnostic.
    """

    # existing-table likelihood p(x | atom)  (vectorized x allowed)
    def like_existing(self, x: float | np.ndarray, atom: float | np.ndarray) -> float:
        pass

    # new-table prior predictive at ROOT
    def like_new_root(self, x: float | np.ndarray) -> float:
        pass

    # sample atom for NEW ROOT table given x_obs
    def sample_root_atom(self, x_obs: float | np.ndarray) -> float | np.ndarray:
        pass

    # update all atoms in place given current assignments
    def update_atoms(
        self, groups: Groups, x: XData, tables_to_atoms: TablesToAtoms, atoms: Atoms
    ) -> None:
        pass

    # update kernel hyperparameters in place (e.g., sigma_x / Sigma_x)
    def update_kernel(
        self, groups: Groups, x: XData, atoms: Atoms, tables_to_atoms: TablesToAtoms
    ) -> None:
        pass


# ---------- 1D Gaussian ----------
class Gaussian1DModel(ObservationModel):
    def __init__(self, mu_phi: float, sigma_phi: float, sigma_x: float = 1.0):
        self.mu_phi = float(mu_phi)
        self.sigma_phi = float(sigma_phi)
        self.sigma_x = float(sigma_x)

    def like_existing(self, x: float | np.ndarray, atom: float | np.ndarray) -> float:
        like: float = norm.pdf(x, loc=atom, scale=self.sigma_x)
        return like

    def like_new_root(self, x: float | np.ndarray) -> float:
        like: float = norm.pdf(
            x, loc=self.mu_phi, scale=np.sqrt(self.sigma_phi**2 + self.sigma_x**2)
        )
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
        self, groups: Groups, x: XData, atoms: Atoms, tables_to_atoms: TablesToAtoms
    ) -> None:
        all_tables = [obs for group in groups.values() for obs in group]
        all_data = np.array([xi for arr in x.values() for xi in arr], dtype=float)
        all_atoms = np.array([atoms[tables_to_atoms[g[0]]] for g in all_tables], dtype=float)
        self.sigma_x = float(np.sqrt(np.mean((all_data - all_atoms) ** 2)))


# ---------- 2D Gaussian ----------
class Gaussian2DModel(ObservationModel):
    def __init__(self, mu_phi: np.ndarray, Sigma_phi: np.ndarray, Sigma_x: np.ndarray):
        self.mu_phi = np.asarray(mu_phi, dtype=float)  # shape (2,)
        self.Sigma_phi = np.asarray(Sigma_phi, dtype=float)  # (2,2)
        self.Sigma_x = np.asarray(Sigma_x, dtype=float)  # (2,2)
        # precompute inverses where helpful
        self._Sigma_phi_inv = np.linalg.inv(self.Sigma_phi)
        self._Sigma_x_inv = np.linalg.inv(self.Sigma_x)

    def like_existing(self, x: float | np.ndarray, atom: float | np.ndarray) -> float:
        like: float = multivariate_normal.pdf(x, mean=np.asarray(atom), cov=self.Sigma_x)
        return like

    def like_new_root(self, x: float | np.ndarray) -> float:
        like: float = multivariate_normal.pdf(
            x, mean=self.mu_phi, cov=self.Sigma_phi + self.Sigma_x
        )
        return like

    def sample_root_atom(self, x_obs: float | np.ndarray) -> np.ndarray:
        # posterior: N( Σ * (Σx^{-1} x + Σφ^{-1} μ), Σ ) with Σ = (Σφ^{-1} + Σx^{-1})^{-1}
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
        self, groups: Groups, x: XData, atoms: Atoms, tables_to_atoms: TablesToAtoms
    ) -> None:
        # simple empirical covariance of residuals
        all_tables = [obs for group in groups.values() for obs in group]
        X = np.vstack([np.asarray(xi) for arr in x.values() for xi in arr])  # (N,2)
        A = np.vstack([np.asarray(atoms[tables_to_atoms[g[0]]]) for g in all_tables])  # (N,2)
        R = X - A
        # unbiased covariance
        self.Sigma_x = np.cov(R.T, bias=False)
        self._Sigma_x_inv = np.linalg.inv(self.Sigma_x)

from __future__ import annotations

from typing import Any

import numpy as np

from GraphPY.sampler.engine import run_sampler
from GraphPY.sampler.kernels import Gaussian1DModel, MultivariateGaussianModel
from GraphPY.types import Nodes, XData


def GPYSamplerGaussian1D(
    nodes: Nodes,
    x: XData,
    n_iter: int = 1000,
    burn_in: int | None = None,
    sigma: float = 0.0,
    mu_base: float = 0.0,
    lambda_base: float = 1.0,
    alpha_base: float = 0.5,
    beta_base: float = 10,
    alpha_kernel: float = 0.5,
    beta_kernel: float = 2.5,
    random_init: bool = False,
    a_0: float = 1.0,
    b_0: float = 1.0,
    group_init_mode: str = "diffuse",
    progress: bool = True,
) -> dict[str, Any]:
    """
    Wrapper for the engine with the Gaussian1D Model
    """
    model = Gaussian1DModel(
        mu_base=mu_base,
        lambda_base=lambda_base,
        alpha_base=alpha_base,
        beta_base=beta_base,
        alpha_kernel=alpha_kernel,
        beta_kernel=beta_kernel,
        random_init=random_init,
    )

    res: dict[str, Any] = run_sampler(
        nodes=nodes,
        x=x,
        model=model,
        n_iter=n_iter,
        burn_in=burn_in,
        sigma=sigma,
        a_0=a_0,
        b_0=b_0,
        group_init_mode=group_init_mode,
        progress=progress,
    )
    return res


def GPYSamplerMultivariateGaussian(
    nodes: Nodes,
    x: XData,
    n_iter: int = 1000,
    burn_in: int | None = None,
    sigma: float = 0.0,
    mu_base: np.ndarray | None = None,
    lambda_base: float = 1.0,
    nu_base: float | None = None,
    psi_base: np.ndarray | None = None,
    nu_kernel: float | None = None,
    psi_kernel: np.ndarray | None = None,
    random_init: bool = False,
    a_0: float = 1.0,
    b_0: float = 1.0,
    group_init_mode: str = "diffuse",
    progress: bool = True,
) -> dict[str, Any]:
    """
    Wrapper for the engine with the MultivariateGaussian Model
    """
    n_dim = np.array([xi for arr in x.values() for xi in arr], dtype=float)[0].shape[0]

    model = MultivariateGaussianModel(
        n_dim=n_dim,
        mu_base=mu_base,
        lambda_base=lambda_base,
        nu_base=nu_base,
        psi_base=psi_base,
        nu_kernel=nu_kernel,
        psi_kernel=psi_kernel,
        random_init=random_init,
    )
    res: dict[str, Any] = run_sampler(
        nodes=nodes,
        x=x,
        model=model,
        n_iter=n_iter,
        burn_in=burn_in,
        sigma=sigma,
        a_0=a_0,
        b_0=b_0,
        group_init_mode=group_init_mode,
        progress=progress,
    )

    return res

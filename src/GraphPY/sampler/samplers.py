from __future__ import annotations

from typing import Any

import numpy as np

from GraphPY.sampler.engine import run_sampler
from GraphPY.sampler.kernels import Gaussian1DModel, Gaussian2DModel
from GraphPY.types import Nodes, XData


def GPYSampler1D(
    nodes: Nodes,
    x: XData,
    n_iter: int = 1000,
    burn_in: int | None = None,
    sigma: float = 0.0,
    mu_phi: float = 0.0,
    sigma_phi: float = 10.0,
    a_0: float = 1.0,
    b_0: float = 1.0,
    group_init_mode: str = "diffuse",
    progress: bool = True,
) -> dict[str, Any]:
    """
    1D wrapper with the same public API as the original GPYSampler from older versions.
    """
    model = Gaussian1DModel(mu_phi=mu_phi, sigma_phi=sigma_phi, sigma_x=1.0)
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
    # add user-facing params for convenience
    res["params"].update({"mu_phi": mu_phi, "sigma_phi": sigma_phi})
    return res


def GPYSampler2D(
    nodes: Nodes,
    x: XData,
    n_iter: int = 1000,
    burn_in: int | None = None,
    sigma: float = 0.0,
    mu_phi: np.ndarray | None = None,
    Sigma_phi: np.ndarray | None = None,
    Sigma_x_init: np.ndarray | None = None,
    a_0: float = 1.0,
    b_0: float = 1.0,
    group_init_mode: str = "diffuse",
    progress: bool = True,
) -> dict[str, Any]:
    """
    2D wrapper. Pass 2D arrays for observations and Gaussian hyperparameters.
    """

    if not mu_phi:
        mu_phi = np.zeros(2)
    if not Sigma_phi:
        Sigma_phi = np.eye(2)
    if not Sigma_x_init:
        Sigma_x_init = np.eye(2)

    model = Gaussian2DModel(
        mu_phi=np.asarray(mu_phi), Sigma_phi=np.asarray(Sigma_phi), Sigma_x=np.asarray(Sigma_x_init)
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
    # expose priors for convenience
    res["params"].update({"mu_phi": mu_phi, "Sigma_phi": Sigma_phi})
    return res

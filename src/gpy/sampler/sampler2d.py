from __future__ import annotations

from typing import Any, Dict
import numpy as np
from gpy.sampler.engine import run_sampler
from gpy.sampler.kernels import Gaussian2DModel
from gpy.types import Nodes, XData

# Note: x for 2D should be Dict[str, List[np.ndarray]] where each obs is shape (2,)

def GPYSampler2D(
    nodes: Nodes,
    x: XData,
    n_iter: int = 1000,
    burn_in: int | None = None,
    sigma: float = 0.0,
    mu_phi: np.ndarray = np.zeros(2),
    Sigma_phi: np.ndarray = np.eye(2),
    Sigma_x_init: np.ndarray = np.eye(2),
    a_0: float = 1.0,
    b_0: float = 1.0,
    group_init_mode: str = "diffuse",
    progress: bool = True,
) -> Dict[str, Any]:
    """
    2D wrapper. Pass 2D arrays for observations and Gaussian hyperparameters.
    """
    model = Gaussian2DModel(mu_phi=np.asarray(mu_phi), Sigma_phi=np.asarray(Sigma_phi), Sigma_x=np.asarray(Sigma_x_init))
    res = run_sampler(
        nodes=nodes, x=x, model=model, n_iter=n_iter, burn_in=burn_in,
        sigma=sigma, a_0=a_0, b_0=b_0, group_init_mode=group_init_mode, progress=progress
    )
    # expose priors for convenience
    res["params"].update({"mu_phi": mu_phi, "Sigma_phi": Sigma_phi})
    return res

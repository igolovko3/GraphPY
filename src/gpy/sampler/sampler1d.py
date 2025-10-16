from __future__ import annotations

from typing import Any, Dict
from gpy.sampler.engine import run_sampler
from gpy.sampler.kernels import Gaussian1DModel
from gpy.types import Nodes, XData


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
) -> Dict[str, Any]:
    """
    1D wrapper with the same public API as the original GPYSampler from older versions.
    """
    model = Gaussian1DModel(mu_phi=mu_phi, sigma_phi=sigma_phi, sigma_x=1.0)
    res = run_sampler(
        nodes=nodes, x=x, model=model, n_iter=n_iter, burn_in=burn_in,
        sigma=sigma, a_0=a_0, b_0=b_0, group_init_mode=group_init_mode, progress=progress
    )
    # add user-facing params for convenience
    res["params"].update({"mu_phi": mu_phi, "sigma_phi": sigma_phi})
    return res

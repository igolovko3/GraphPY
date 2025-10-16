import numpy as np
from gpy.clustering.psm import comp_psm, VI_lb, minVI


def test_psm_and_minvi_shapes():
    # Build synthetic partitions: 12 partitions of 8 items with 3–4 clusters each
    rng = np.random.default_rng(123)
    n_part, n = 12, 8
    cls = np.vstack([
        rng.integers(0, rng.integers(3, 5), size=n) for _ in range(n_part)
    ])
    psm = comp_psm(cls)
    assert psm.shape == (n, n)
    # symmetry + diag ones
    assert np.allclose(psm, psm.T)
    assert np.allclose(np.diag(psm), 1.0)

    # Candidate VI over average-linkage cuts
    out = minVI(psm, max_k=4)
    assert "cl" in out and out["cl"].shape == (n,)
    assert "cls_avg" in out and out["cls_avg"].ndim == 2

    # VI_lb returns one value per candidate
    VI_vals = VI_lb(out["cls_avg"], psm)
    assert VI_vals.shape[0] == out["cls_avg"].shape[0]

from typing import Any

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage


def comp_psm(cls: np.ndarray) -> np.ndarray:
    """
    Posterior Similarity Matrix (PSM).

    PSM[i, j] = fraction of sampled partitions in which items i and j share a label.

    Parameters
    ----------
    cls : (n_partitions, n_objects) array
        Cluster labels per object for each sampled partition.

    Returns
    -------
    (n_objects, n_objects) array
        Symmetric matrix with entries in [0, 1].

    Notes
    -----
    Complexity O(n_partitions * n_objects^2).
    """
    n_obj = cls.shape[1]
    n_part = cls.shape[0]
    psm = np.zeros((n_obj, n_obj))

    for n in range(n_obj):
        for n_ in range(n_obj):
            for part in range(n_part):
                if cls[part, n] == cls[part, n_]:
                    psm[n, n_] += 1

    psm = psm / n_part
    return psm


def VI_lb(cls: np.ndarray, psm: np.ndarray) -> np.ndarray:
    """
    Lower bound to Variation of Information (VI) for candidate partitions given a PSM.

    Parameters
    ----------
    cls : array
        Candidate partitions: (n_candidates, n_objects). If 1D, it is transposed.
    psm : (n, n) array
        Posterior Similarity Matrix for the same n objects.

    Returns
    -------
    (n_candidates,) array
        VI lower-bound values, one per candidate.
    """
    if isinstance(cls, np.ndarray) and cls.ndim == 1:
        cls = np.transpose(cls)

    n = psm.shape[0]

    def VI_lb_compute(c: np.ndarray) -> float:
        f = 0.0
        for i in range(n):
            ind = c == c[i]
            f += (
                np.log2(np.sum(ind))
                + np.log2(np.sum(psm[i, :]))
                - 2 * np.log2(np.sum(ind * psm[i, :]))
            ) / n
        return float(f)

    output = np.apply_along_axis(VI_lb_compute, 1, cls)
    return output


def minVI(
    psm: np.ndarray,
    max_k: int | None = None,
    start_cl: np.ndarray | None = None,
    maxiter: int | None = None,
    suppress_comment: bool = True,
) -> dict[str, Any]:
    """
    Choose an “average” partition by minimizing the VI lower bound over
    average-linkage cuts of 1 - PSM (k = 1..max_k).

    Parameters
    ----------
    psm : (n, n) array
        Posterior Similarity Matrix.
    max_k : int, optional
        Max clusters considered (default: ceil(n/4)).
    start_cl, maxiter, l, suppress_comment : optional
        Kept for API parity; unused.

    Returns
    -------
    dict
        {
            "cl": (n,) labels for the selected partition,
            "value": minimal VI lower bound (float),
            "method": "avg",
            "cls_avg": all average-linkage candidates (k=1..max_k)
        }
    """
    if max_k is None:
        max_k = int(np.ceil(psm.shape[0] / 4))

    hclust_avg = linkage(1 - psm, method="average")
    cls_avg = cut_tree(hclust_avg, range(1, max_k + 1)).T
    VI_avg = VI_lb(cls_avg, psm)
    val_avg = np.min(VI_avg)
    cl_avg = cls_avg[np.argmin(VI_avg), :]

    output: dict[str, Any] = {"cl": cl_avg, "value": val_avg, "method": "avg", "cls_avg": cls_avg}
    return output

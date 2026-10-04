"""Shared helpers for alignment methods (author: 晨星)."""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC

from domainforge.core.errors import AdaptationError, InvalidDataError
from domainforge.core.types import DomainSplit


def make_clf(C: float = 1.0, seed: int = 42) -> SVC:
    """Shared target classifier. RBF-SVM captures the non-linear structure of
    moons/blobs; probability=True so the fusion layer can soft-vote.

    NOTE: n_jobs is intentionally NOT set (受限 Windows 环境并发会崩, SOP 坑).
    """
    return SVC(C=C, kernel="rbf", gamma="scale", probability=True, random_state=seed)


def check_and_stack(split: DomainSplit) -> tuple[np.ndarray, np.ndarray]:
    """Validate split and return (X_all = [Xs; Xt], domain indicator)."""
    if split.X_source.shape[1] != split.X_target.shape[1]:
        raise InvalidDataError("source/target feature dims differ")
    return (
        np.vstack([split.X_source, split.X_target]),
        np.concatenate(
            [np.zeros(split.X_source.shape[0]), np.ones(split.X_target.shape[0])]
        ),
    )


def rbf_kernel_median(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """RBF kernel with median-heuristic bandwidth (deterministic)."""
    Z = X if Y is None else np.vstack([X, Y])
    from sklearn.metrics.pairwise import euclidean_distances

    D = euclidean_distances(Z, squared=True)
    iu = np.triu_indices(Z.shape[0], k=1)
    med = float(np.median(D[iu]))
    med = med if med > 0 else 1.0
    K = euclidean_distances(X, Z)
    return np.exp(-K / med)


def stack_pair(Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    if Xs.shape[1] != Xt.shape[1]:
        raise AdaptationError("dim mismatch in kernel stack")
    return np.vstack([Xs, Xt])

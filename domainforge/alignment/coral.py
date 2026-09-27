"""CORAL: CORrelation ALignment (Sun et al., 2011) - pure numpy (author: 晨星).

Whiten the source with its covariance, re-color it with the target covariance:
    Xs' = (Xs - mu_s) * Cs^{-1/2} * Ct^{1/2} + mu_t
The target itself is left untouched. Invariant checked by tests: the covariance
of the transformed source matches the target covariance.
"""

from __future__ import annotations

import numpy as np

from domainforge.alignment._common import make_clf
from domainforge.core.errors import AdaptationError
from domainforge.core.types import DomainSplit


def _spd_sqrt(M: np.ndarray, sign: int) -> np.ndarray:
    """Symmetric PSD matrix power via eigendecomposition (sign=-1 -> inverse)."""
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-10, None)
    p = w ** (sign / 2.0)
    return (V * p) @ V.T


class CoralAdapter:
    name = "coral"

    def __init__(self, clf_C: float = 1.0, seed: int = 42) -> None:
        self.clf_C = clf_C
        self.seed = seed
        self._mu_s: np.ndarray | None = None
        self._mu_t: np.ndarray | None = None
        self._A: np.ndarray | None = None  # combined linear map
        self._clf = None

    def fit(self, split: DomainSplit) -> CoralAdapter:
        Xs, Xt = split.X_source, split.X_target
        if Xs.shape[1] != Xt.shape[1]:
            raise AdaptationError("dim mismatch")
        self._mu_s = Xs.mean(axis=0)
        self._mu_t = Xt.mean(axis=0)
        Xs0 = Xs - self._mu_s
        Xt0 = Xt - self._mu_t
        Cs = np.cov(Xs0, rowvar=False) + 1e-8 * np.eye(Xs.shape[1])
        Ct = np.cov(Xt0, rowvar=False) + 1e-8 * np.eye(Xt.shape[1])
        self._A = _spd_sqrt(Cs, -1) @ _spd_sqrt(Ct, +1)
        Xs_aligned = Xs0 @ self._A + self._mu_t
        self._clf = make_clf(C=self.clf_C, seed=self.seed)
        self._clf.fit(Xs_aligned, split.y_source)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._A is None:
            raise AdaptationError("coral not fitted")
        return np.asarray(X, dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """CORAL re-colors the SOURCE into the target distribution; target
        points are consumed unchanged (the aligned source lives only inside
        the classifier). Applying the map to target inputs would double-shift
        them - the bug this method once had."""
        if self._clf is None:
            raise AdaptationError("coral not fitted")
        return self._clf.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise AdaptationError("coral not fitted")
        return self._clf.predict_proba(np.asarray(X, dtype=float))

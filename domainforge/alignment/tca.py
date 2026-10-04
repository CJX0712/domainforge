"""TCA: Transfer Component Analysis (Pan et al., 2011) - pure numpy (author: 晨星).

Solve  min_w  MMD^2(P(Zs), P(Zt)) + mu*||W||^2  with centering constraint
W^T K H K W = I, via the generalized eigenproblem
    (K M0 K) w = lam (K H K) w,  take the k smallest eigenvalues.
Embedding of sample i is row i of K W.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel

from domainforge.alignment._common import make_clf, stack_pair
from domainforge.core.errors import AdaptationError
from domainforge.core.types import DomainSplit


def _median_gamma(Xs: np.ndarray, Xt: np.ndarray) -> float:
    from sklearn.metrics.pairwise import euclidean_distances

    Z = stack_pair(Xs, Xt)
    D = euclidean_distances(Z, squared=True)
    iu = np.triu_indices(Z.shape[0], k=1)
    med = float(np.median(D[iu]))
    return 1.0 / med if med > 0 else 1.0


class TcaAdapter:
    name = "tca"

    def __init__(
        self,
        n_components: int = 16,
        mu: float = 1e-2,
        clf_C: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_components = n_components
        self.mu = mu
        self.clf_C = clf_C
        self.seed = seed
        self._W: np.ndarray | None = None
        self._Ksrc: np.ndarray | None = None  # kernel rows of source for transform
        self._Z_all: np.ndarray | None = None
        self._gamma: float = 1.0
        self._mu_s: np.ndarray | None = None
        self._mu_t: np.ndarray | None = None
        self._clf = None

    # ---- core solver -------------------------------------------------
    def _fit_embedding(self, Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
        # per-domain centering BEFORE the kernel: without it a large mean
        # shift makes the cross-domain kernel block vanish (median bandwidth
        # explodes) and the embedding degenerates - SOP-verified pitfall
        self._mu_s = Xs.mean(axis=0)
        self._mu_t = Xt.mean(axis=0)
        Xs0, Xt0 = Xs - self._mu_s, Xt - self._mu_t
        Z = stack_pair(Xs0, Xt0)
        ns, nt = Xs.shape[0], Xt.shape[0]
        self._gamma = _median_gamma(Xs0, Xt0)
        K = rbf_kernel(Z, Z, gamma=self._gamma)
        n = ns + nt
        d = np.concatenate([np.full(ns, 1.0 / ns), np.full(nt, 1.0 / nt)])
        M0 = np.outer(d, d)
        H = np.eye(n) - np.ones((n, n)) / n
        KMK = K @ M0 @ K
        KHK = K @ H @ K
        eps = 1e-8 * np.trace(KHK) / n
        k = int(min(self.n_components, n - 1))
        _w, V = eigh(
            KMK + self.mu * np.eye(n), KHK + eps * np.eye(n), subset_by_index=[0, k - 1]
        )
        self._W = V
        self._Z_all = Z
        emb = K @ V
        return emb

    def _embed(self, X: np.ndarray) -> np.ndarray:
        """Embed TARGET-domain points (contract: predict/transform are called
        on target inputs; target is centered by the target mean)."""
        if self._W is None or self._Z_all is None or self._mu_t is None:
            raise AdaptationError("tca not fitted")
        X0 = np.asarray(X, dtype=float) - self._mu_t
        Kx = rbf_kernel(X0, self._Z_all, gamma=self._gamma)
        return Kx @ self._W

    # ---- Adapter API --------------------------------------------------
    def fit(self, split: DomainSplit) -> TcaAdapter:
        emb = self._fit_embedding(split.X_source, split.X_target)
        ns = split.X_source.shape[0]
        self._clf = make_clf(C=self.clf_C, seed=self.seed)
        self._clf.fit(emb[:ns], split.y_source)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._embed(np.asarray(X, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise AdaptationError("tca not fitted")
        return self._clf.predict(self._embed(np.asarray(X, dtype=float)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise AdaptationError("tca not fitted")
        return self._clf.predict_proba(self._embed(np.asarray(X, dtype=float)))

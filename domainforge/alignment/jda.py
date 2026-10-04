"""JDA: Joint Distribution Adaptation (Long et al., 2013) - pure numpy (author: 晨星).

TCA (marginal MMD) + iterative class-conditional MMD on target pseudo-labels:
    M = M0 + sum_c M_c,  solve (K M K) w = lam (K H K) w, k smallest.
Each round: re-embed -> retrain on source -> refresh pseudo-labels.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel

from domainforge.alignment._common import make_clf, stack_pair
from domainforge.core.errors import AdaptationError
from domainforge.core.types import DomainSplit
from domainforge.data.synthetic import validate_split


class JdaAdapter:
    name = "jda"

    def __init__(
        self,
        n_components: int = 16,
        mu: float = 1e-2,
        n_iters: int = 3,
        clf_C: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_components = n_components
        self.mu = mu
        self.n_iters = max(1, int(n_iters))
        self.clf_C = clf_C
        self.seed = seed
        self._W: np.ndarray | None = None
        self._Z_all: np.ndarray | None = None
        self._gamma: float = 1.0
        self._mu_s: np.ndarray | None = None
        self._mu_t: np.ndarray | None = None
        self._clf = None
        self.n_pseudo_flips_ = 0

    # ------------------------------------------------------------------
    def _mats(
        self, ns: int, nt: int, y_s: np.ndarray, y_t_pseudo: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        n = ns + nt
        d = np.concatenate([np.full(ns, 1.0 / ns), np.full(nt, 1.0 / nt)])
        M = np.outer(d, d)
        if y_t_pseudo is not None:
            classes = np.unique(y_s)
            for c in classes:
                es = (y_s == c).astype(float)
                et = (y_t_pseudo == c).astype(float)
                ns_c = max(es.sum(), 1.0)
                nt_c = max(et.sum(), 1.0)
                # class-conditional MMD block must be placed at the correct
                # domain offsets inside the full (n x n) matrix
                Mc = np.zeros((n, n))
                Mc[:ns, :ns] = np.outer(es, es) / (ns_c * ns_c)
                Mc[ns:, ns:] = np.outer(et, et) / (nt_c * nt_c)
                Mc[:ns, ns:] -= 2.0 * np.outer(es, et) / (ns_c * nt_c)
                Mc[ns:, :ns] -= 2.0 * np.outer(et, es) / (ns_c * nt_c)
                M = M + Mc
        H = np.eye(n) - np.ones((n, n)) / n
        return M, H

    def fit(self, split: DomainSplit) -> JdaAdapter:
        validate_split(split)
        Xs, Xt, ys = split.X_source, split.X_target, split.y_source
        ns, nt = Xs.shape[0], Xt.shape[0]
        # per-domain centering before the kernel (same rationale as TCA)
        self._mu_s = Xs.mean(axis=0)
        self._mu_t = Xt.mean(axis=0)
        Z = stack_pair(Xs - self._mu_s, Xt - self._mu_t)
        from sklearn.metrics.pairwise import euclidean_distances

        D = euclidean_distances(Z, squared=True)
        iu = np.triu_indices(n := ns + nt, k=1)
        med = float(np.median(D[iu]))
        self._gamma = 1.0 / med if med > 0 else 1.0
        K = rbf_kernel(Z, Z, gamma=self._gamma)
        y_t_pseudo: np.ndarray | None = None
        prev: np.ndarray | None = None
        emb = K @ np.eye(K.shape[0])[:, :1]  # placeholder; first embed below
        for _ in range(self.n_iters):
            M, H = self._mats(ns, nt, ys, y_t_pseudo)
            KMK, KHK = K @ M @ K, K @ H @ K
            eps = 1e-8 * np.trace(KHK) / n
            k = int(min(self.n_components, n - 1))
            _w, V = eigh(
                KMK + self.mu * np.eye(n),
                KHK + eps * np.eye(n),
                subset_by_index=[0, k - 1],
            )
            self._W = V
            emb = K @ V
            clf = make_clf(C=self.clf_C, seed=self.seed)
            clf.fit(emb[:ns], ys)
            y_t_pseudo = clf.predict(emb[ns:])
            if prev is not None:
                self.n_pseudo_flips_ = int((prev != y_t_pseudo).sum())
            prev = y_t_pseudo
        self._Z_all = Z
        self._clf = make_clf(C=self.clf_C, seed=self.seed)
        self._clf.fit(emb[:ns], ys)
        return self

    def _embed(self, X: np.ndarray) -> np.ndarray:
        """Embed TARGET-domain points (centered by the target mean)."""
        if self._W is None or self._Z_all is None or self._mu_t is None:
            raise AdaptationError("jda not fitted")
        X0 = np.asarray(X, dtype=float) - self._mu_t
        return rbf_kernel(X0, self._Z_all, gamma=self._gamma) @ self._W

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._embed(np.asarray(X, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise AdaptationError("jda not fitted")
        return self._clf.predict(self._embed(np.asarray(X, dtype=float)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise AdaptationError("jda not fitted")
        return self._clf.predict_proba(self._embed(np.asarray(X, dtype=float)))

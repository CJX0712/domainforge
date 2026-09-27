"""Instance reweighting: KLIEP-lite density-ratio via a domain discriminator
(author: 晨星).

Train a logistic discriminator p(target | x) on standardized features; the
importance weight is w(x) = p / (1 - p), clipped to [clip_lo, clip_hi] and
normalized to mean 1. The source classifier is then fit with
``sample_weight``. This is the cheap discriminative surrogate for KLIEP
(Sugiyama et al., 2008); the full KLIEP objective lives in the optional
``adapt`` backend cross-check row.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from domainforge.alignment._common import make_clf
from domainforge.core.errors import AdaptationError
from domainforge.core.types import DomainSplit


class KliepAdapter:
    name = "kliep"

    def __init__(
        self, disc_C: float = 1.0, clip_hi: float = 8.0, clf_C: float = 1.0, seed: int = 42
    ) -> None:
        self.disc_C = disc_C
        self.clip_hi = clip_hi
        self.clf_C = clf_C
        self.seed = seed
        self._sc: StandardScaler | None = None
        self._disc: LogisticRegression | None = None
        self._clf = None
        self._weights_mean: float = 1.0

    def fit(self, split: DomainSplit) -> KliepAdapter:
        Xs, Xt = split.X_source, split.X_target
        self._sc = StandardScaler().fit(np.vstack([Xs, Xt]))
        Zs, Zt = self._sc.transform(Xs), self._sc.transform(Xt)
        Xd = np.vstack([Zs, Zt])
        yd = np.concatenate([np.zeros(Zs.shape[0]), np.ones(Zt.shape[0])])
        # NOTE: sklearn 1.9 removed multi_class=; lbfgs handles multiclass natively
        self._disc = LogisticRegression(C=self.disc_C, solver="lbfgs", max_iter=1000)
        self._disc.fit(Xd, yd)
        p = np.clip(self._disc.predict_proba(Zs)[:, 1], 1e-3, 1 - 1e-3)
        w = p / (1.0 - p)
        w = np.clip(w, 1e-3, self.clip_hi)
        self.weights_ = w.copy()
        w = w / w.mean()
        self._weights_mean = float(w.mean())
        self._clf = make_clf(C=self.clf_C, seed=self.seed)
        self._clf.fit(Zs, split.y_source, sample_weight=w)
        # the classifier operates in standardized space; predict() standardizes
        self._clf_class_order = getattr(self._clf, "classes_", None)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._sc is None:
            raise AdaptationError("kliep not fitted")
        return self._sc.transform(np.asarray(X, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None or self._sc is None:
            raise AdaptationError("kliep not fitted")
        return self._clf.predict(self._sc.transform(np.asarray(X, dtype=float)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None or self._sc is None:
            raise AdaptationError("kliep not fitted")
        return self._clf.predict_proba(self._sc.transform(np.asarray(X, dtype=float)))

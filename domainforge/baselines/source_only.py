"""Source-only baseline: train on source, predict target unchanged (author: 晨星)."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from domainforge.alignment._common import make_clf
from domainforge.core.errors import AdaptationError
from domainforge.core.types import DomainSplit


class SourceOnlyAdapter:
    name = "source_only"

    def __init__(self, clf_C: float = 1.0, seed: int = 42) -> None:
        self.clf_C = clf_C
        self.seed = seed
        self._sc: StandardScaler | None = None
        self._clf = None

    def fit(self, split: DomainSplit) -> SourceOnlyAdapter:
        self._sc = StandardScaler().fit(split.X_source)
        self._clf = make_clf(C=self.clf_C, seed=self.seed)
        self._clf.fit(self._sc.transform(split.X_source), split.y_source)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._sc is None:
            raise AdaptationError("source_only not fitted")
        return self._sc.transform(np.asarray(X, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None or self._sc is None:
            raise AdaptationError("source_only not fitted")
        return self._clf.predict(self._sc.transform(np.asarray(X, dtype=float)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None or self._sc is None:
            raise AdaptationError("source_only not fitted")
        return self._clf.predict_proba(self._sc.transform(np.asarray(X, dtype=float)))

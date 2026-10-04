"""Optional ``adapt`` library backend (cross-check row, author: 晨星).

``adapt`` pulls TensorFlow; importing it is slow and it may be broken in
restricted environments. Availability is probed once, lazily, and cached
(SOP rule: optional SOTA backend must degrade to skipped, never fake rows).
"""

from __future__ import annotations

import time

import numpy as np

from domainforge.core.errors import BackendUnavailableError
from domainforge.core.types import DomainSplit

_PROBE_CACHE: dict[str, bool] = {}


def available_adapt() -> bool:
    """One-shot lazy probe with caching. Never raises."""
    if "adapt" in _PROBE_CACHE:
        return _PROBE_CACHE["adapt"]
    try:
        from adapt.feature_based import CORAL as _AdaptCoral  # noqa: F401

        _PROBE_CACHE["adapt"] = True
    except Exception:
        _PROBE_CACHE["adapt"] = False
    return _PROBE_CACHE["adapt"]


class AdaptCoralBackend:
    """Thin wrapper around adapt.feature_based.CORAL with an RBF-SVC estimator."""

    name = "adapt_coral"

    def __init__(self, clf_C: float = 1.0, seed: int = 42) -> None:
        self.clf_C = clf_C
        self.seed = seed
        self._model = None

    def fit(self, split: DomainSplit) -> AdaptCoralBackend:
        if not available_adapt():
            raise BackendUnavailableError("adapt/TF not importable")
        from adapt.feature_based import CORAL
        from sklearn.svm import SVC

        est = SVC(
            C=self.clf_C,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=self.seed,
        )
        self._model = CORAL(estimator=est, Xt=split.X_target, verbose=0)
        t0 = time.perf_counter()
        self._model.fit(split.X_source, split.y_source)
        self._fit_seconds = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise BackendUnavailableError("adapt_coral not fitted")
        return np.asarray(self._model.predict(np.asarray(X, dtype=float)))

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise BackendUnavailableError("adapt_coral not fitted")
        return np.asarray(X, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise BackendUnavailableError("adapt_coral not fitted")
        return np.asarray(self._model.predict_proba(np.asarray(X, dtype=float)))

"""Method interfaces (Protocol contracts, author: 晨星).

Semantic contract shared by every adaptation method:
- ``fit(split)`` learns from source (+ tiny labeled target val) only;
  target test labels are NEVER passed to fit.
- ``predict(X)`` returns class predictions on the target domain.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from domainforge.core.types import DomainSplit


@runtime_checkable
class Adapter(Protocol):
    """Every DA method must satisfy this protocol."""

    name: str

    def fit(self, split: DomainSplit) -> Adapter:
        """Fit on source + tiny labeled target val; return self."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for target-domain samples."""

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the aligned/transformed representation of ``X``."""

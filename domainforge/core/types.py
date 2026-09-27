"""Core data types for DomainForge (author: 晨星)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DomainSplit:
    """A source/target domain pair with a tiny labeled target validation set.

    Contract
    --------
    - ``X_source``/``y_source``: fully labeled source domain.
    - ``X_target``/``y_target_private``: target labels are PRIVATE and never
      visible to any adaptation method (used only for final evaluation).
    - ``X_target_val``/``y_target_val``: a tiny labeled target sample
      (``n_target_val``) that methods may use for hyper-parameter selection.
    """

    name: str
    X_source: np.ndarray
    y_source: np.ndarray
    X_target: np.ndarray
    y_target_private: np.ndarray
    X_target_val: np.ndarray
    y_target_val: np.ndarray
    task: str = "classification"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for nm in (
            "X_source",
            "y_source",
            "X_target",
            "y_target_private",
            "X_target_val",
            "y_target_val",
        ):
            arr = np.asarray(getattr(self, nm))
            setattr(self, nm, arr)
        if self.X_source.ndim != 2:
            raise ValueError("X_source must be 2-D")


@dataclass
class MethodResult:
    """Prediction output of one adaptation method on the target test set."""

    method: str
    y_pred: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRow:
    """One row of the cross-method benchmark table."""

    dataset: str
    method: str
    accuracy: float
    macro_f1: float
    mmd2: float  # source-vs-target linear MMD^2 in the model's transformed space
    fit_seconds: float
    status: str = "ok"  # ok | skipped | error
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "method": self.method,
            "accuracy": round(self.accuracy, 6),
            "macro_f1": round(self.macro_f1, 6),
            "mmd2": round(self.mmd2, 6),
            "fit_seconds": round(self.fit_seconds, 4),
            "status": self.status,
            "detail": self.detail,
        }

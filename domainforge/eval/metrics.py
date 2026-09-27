"""Evaluation metrics (author: 晨星).

- accuracy / macro-F1 on the (private) target test labels.
- linear MMD^2 between source and target in the method's transformed space:
  measures how much distribution shift remains AFTER adaptation. Lower is
  better; source_only transforms nothing so its MMD is the raw shift.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def linear_mmd2(Xs: np.ndarray, Xt: np.ndarray) -> float:
    """Unbiased linear MMD^2 = mean(K_ss) + mean(K_tt) - 2*mean(K_st) on
    linear kernel (cheap, deterministic)."""
    ns, nt = Xs.shape[0], Xt.shape[0]
    sub_s = Xs if ns <= 400 else Xs[np.random.default_rng(0).choice(ns, 400, replace=False)]
    sub_t = Xt if nt <= 400 else Xt[np.random.default_rng(1).choice(nt, 400, replace=False)]
    Kss = sub_s @ sub_s.T
    Ktt = sub_t @ sub_t.T
    Kst = sub_s @ sub_t.T
    n1, n2 = sub_s.shape[0], sub_t.shape[0]
    ter_s = (Kss.sum() - np.trace(Kss)) / (n1 * (n1 - 1))
    ter_t = (Ktt.sum() - np.trace(Ktt)) / (n2 * (n2 - 1))
    return float(ter_s + ter_t - 2.0 * Kst.mean())


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Returns (accuracy, macro_f1). Both in [0, 1]."""
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro")),
    )

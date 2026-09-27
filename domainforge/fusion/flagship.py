"""SAFuse: Safeguarded Accuracy-Filtered fusion flagship (author: 晨星).

Innovation (flagship method)
----------------------------
Stage 1 - per-method HPO on the tiny labeled target val sample.
Stage 2 - gain-gated soft voting: each method's vote weight is
          g_m = max(0, val_acc(m) - val_acc(source_only)); methods that do
          not beat source-only get ZERO weight, so weak/adversarial members
          cannot drag the ensemble down.
Stage 3 - safeguard: if the weighted ensemble's val accuracy is non-inferior
          to the best single member (tolerance, default 0.01), ship it;
          otherwise fall back to the best single member. The fallback is a
          first-class outcome, never a failure.

This mirrors stacked generalization but with (a) a zero-below-baseline gate
and (b) an explicit non-inferiority safeguard - on tiny val samples (n=20)
that is the difference between a real gain and val-overfitting.
"""

from __future__ import annotations

import numpy as np

from domainforge.core.config import CONFIG
from domainforge.core.errors import AdaptationError
from domainforge.core.types import DomainSplit
from domainforge.hpo.search import hpo_method
from domainforge.registry import build_method

BASE_MEMBERS = ["coral", "tca", "jda", "kliep"]


class SafuseAdapter:
    name = "safuse"

    def __init__(
        self,
        members: list[str] | None = None,
        do_hpo: bool = True,
        n_trials: int | None = None,
        noninferior_tol: float | None = None,
        survivor_tol: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.members = members if members is not None else list(BASE_MEMBERS)
        self.do_hpo = do_hpo
        self.n_trials = n_trials
        self.noninferior_tol = (
            CONFIG.fuse_noninferior_tol if noninferior_tol is None else noninferior_tol
        )
        # tiny-val noise guard: only members whose val acc is within
        # survivor_tol of the best member AND that beat the anchor may vote
        self.survivor_tol = survivor_tol
        self.seed = seed
        self._fitted: list[tuple[str, object, float]] = []  # (name, model, weight)
        self._classes: np.ndarray | None = None
        self.mode_: str = "pending"
        self.member_val_acc_: dict[str, float] = {}
        self.member_params_: dict[str, dict] = {}

    # ------------------------------------------------------------------
    def _val_acc(self, model, split: DomainSplit) -> float:
        pred = model.predict(split.X_target_val)
        return float((pred == split.y_target_val).mean())

    def _member(self, name: str, split: DomainSplit):
        params: dict = {}
        if self.do_hpo:
            params, _ = hpo_method(name, split, n_trials=self.n_trials)
        self.member_params_[name] = params
        return build_method(name, params=params, seed=self.seed)

    def fit(self, split: DomainSplit) -> SafuseAdapter:
        # anchor: source-only val accuracy (the zero-gain baseline)
        anchor = build_method("source_only", seed=self.seed).fit(split)
        anchor_val = self._val_acc(anchor, split)

        fitted_members: list[tuple[str, object, float]] = []
        for name in self.members:
            try:
                m = self._member(name, split).fit(split)
                v = self._val_acc(m, split)
                self.member_val_acc_[name] = v
                gain = max(0.0, v - anchor_val)
                fitted_members.append((name, m, gain))
            except Exception:
                continue  # a broken member never kills the flagship
        if not fitted_members:
            self._fitted = [("source_only", anchor, 1.0)]
            self.mode_ = "fallback_source_only"
            return self

        best_name, best_m = max(
            fitted_members, key=lambda t: self.member_val_acc_.get(t[0], 0.0)
        )[:2]
        best_val = self.member_val_acc_[best_name]

        # survivor filter: near-best on val AND beats the anchor; cap to the
        # top-2 members - on a 20-sample val every member looks similar, and
        # stacking many doubtful members flips correct votes (SOP-verified)
        survivors = sorted(
            [
                (n, m, g)
                for n, m, g in fitted_members
                if g > 0 and self.member_val_acc_.get(n, 0.0) >= best_val - self.survivor_tol
            ],
            key=lambda t: self.member_val_acc_.get(t[0], 0.0),
            reverse=True,
        )[:2]
        if not survivors:
            # nobody beats the anchor -> fall back to the single best member
            self._fitted = [(best_name, best_m, 1.0)]
            self.mode_ = "fallback_best_single"
            return self

        gains = np.array([g for _, _, g in survivors])
        weights = gains / gains.sum()
        self._fitted = [
            (n, m, float(w)) for (n, m, _), w in zip(survivors, weights, strict=False)
        ]

        # safeguard on val: ensemble must be non-inferior to best single
        ens_val = self._val_acc(self, split)
        if ens_val < best_val - self.noninferior_tol:
            self._fitted = [(best_name, best_m, 1.0)]
            self.mode_ = "fallback_best_single"
        else:
            self.mode_ = "ensemble"
        return self

    # ------------------------------------------------------------------
    def _proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise AdaptationError("safuse not fitted")
        X = np.asarray(X, dtype=float)
        # union of classes across members; scatter each member's prob columns
        # into the union layout (columns are ordered per-member classes_)
        union: np.ndarray | None = None
        parts: list[tuple[np.ndarray, np.ndarray, float]] = []
        for _name, m, w in self._fitted:
            p = np.asarray(m.predict_proba(X))
            cls = np.asarray(m.classes_) if hasattr(m, "classes_") else np.arange(p.shape[1])
            parts.append((p, cls, w))
            union = cls if union is None else np.union1d(union, cls)
        assert union is not None
        P = np.zeros((X.shape[0], union.size))
        for p, cls, w in parts:
            col = np.searchsorted(union, cls)
            P[:, col] += p * w
        # normalize rows (members may miss classes present in the union)
        row = P.sum(axis=1, keepdims=True)
        P = np.divide(P, row, out=np.full_like(P, 1.0 / union.size), where=row > 0)
        return P, union

    def predict(self, X: np.ndarray) -> np.ndarray:
        P, classes = self._proba(np.asarray(X, dtype=float))
        return classes[np.argmax(P, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        P, _ = self._proba(np.asarray(X, dtype=float))
        return P

    def transform(self, X: np.ndarray) -> np.ndarray:
        # identity: fusion operates on predictions, not representations
        return np.asarray(X, dtype=float)

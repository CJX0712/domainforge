"""Synthetic domain-shift dataset generators (author: 晨星).

Difficulty knobs are tuned so that ``source_only`` is meaningfully below the
ceiling (otherwise adaptation has no room to win and HPO collapses to a flat
objective - a pitfall recorded in the delivery SOP).
"""

from __future__ import annotations

import numpy as np

from domainforge.core.errors import InvalidDataError
from domainforge.core.types import DomainSplit


def _val_sample(
    X_t: np.ndarray, y_t: np.ndarray, n_val: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the target pool into a tiny labeled val set + the rest (test)."""
    n = X_t.shape[0]
    n_val = max(2, min(n_val, n // 2))
    idx = rng.permutation(n)
    val_idx, test_idx = idx[:n_val], idx[n_val:]
    return X_t[val_idx], y_t[val_idx], X_t[test_idx], y_t[test_idx]


def make_covariate_shift(
    n_val: int = 20, seed: int = 42, sep: float = 1.1, shift: float = 2.2
) -> DomainSplit:
    """Two-class Gaussians; target = source + per-feature rescale + mean shift.

    Difficulty knobs tuned so source_only lands well below the ceiling
    (validation set is drawn from the FULL domain, matching test).
    """
    rng = np.random.default_rng(seed)
    d = 8
    w = rng.normal(size=d) / np.sqrt(d)
    n_s = 400
    n_t = 400
    Xs = rng.normal(size=(n_s, d)) + 0.25 * rng.normal(size=(n_s, d))
    ys = (Xs @ w + 0.2 * rng.normal(size=n_s) > 0).astype(int)
    Xs = Xs + sep * np.where(ys[:, None] == 1, 1.0, -1.0) * w

    scale = rng.uniform(0.5, 2.0, size=d)  # heterogeneous rescale
    off = shift * rng.uniform(-1.0, 1.0, size=d)  # mean shift
    # guarantee the shift actually destroys the source decision boundary:
    # rescale the offset so its projection on the discriminative direction
    # dominates the class signal (otherwise source_only may stay near-ceiling
    # by seed luck)
    proj = float(off @ w)
    if abs(proj) < 1.5:
        off = off * (min(1.5 / max(abs(proj), 1e-6), 3.0))
    Xt = (rng.normal(size=(n_t, d)) + 0.25 * rng.normal(size=(n_t, d))) * scale + off
    yt = (Xt @ w + 0.2 * rng.normal(size=n_t) > 0).astype(int)
    Xt = Xt + sep * np.where(yt[:, None] == 1, 1.0, -1.0) * w

    Xv, yv, Xte, yte = _val_sample(Xt, yt, n_val, rng)
    return DomainSplit(
        "covariate_shift", Xs, ys, Xte, yte, Xv, yv, meta={"w": w, "shift": shift}
    )


def make_rotated_moons(
    n_val: int = 20, seed: int = 42, angle_deg: float = 55.0, noise: float = 0.22
) -> DomainSplit:
    """Two interleaved moons; target is rotated + noisier (covariate shift)."""
    from sklearn.datasets import make_moons

    rng = np.random.default_rng(seed)
    Xs, ys = make_moons(n_samples=400, noise=noise, random_state=seed)
    Xs = Xs * 2.0 + rng.normal(scale=0.05, size=Xs.shape)  # extra jitter (diversity)
    ang = np.deg2rad(angle_deg)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    Xt, yt = make_moons(n_samples=400, noise=noise + 0.10, random_state=seed + 1)
    Xt = (Xt * 2.0 + rng.normal(scale=0.05, size=Xt.shape)) @ R.T

    Xv, yv, Xte, yte = _val_sample(Xt, yt, n_val, rng)
    return DomainSplit("rotated_moons", Xs, ys, Xte, yte, Xv, yv, meta={"angle_deg": angle_deg})


def make_scaled_blobs(
    n_val: int = 20, seed: int = 42, sep: float = 1.0, smax: float = 3.0
) -> DomainSplit:
    """4-class blobs sharing the SAME cluster centers in both domains; the
    target applies heterogeneous axis rescaling (1..smax), wider spread and a
    mean shift. (Earlier version drew independent centers -> label semantics
    unaligned, an impossible problem rather than domain adaptation.)"""
    rng = np.random.default_rng(seed)
    d = 10
    centers = rng.uniform(-4.0, 4.0, size=(4, d))
    n_s, n_t = 480, 480
    ys = rng.integers(0, 4, size=n_s)
    yt = rng.integers(0, 4, size=n_t)
    Xs = centers[ys] + rng.normal(scale=sep, size=(n_s, d))
    scales = rng.uniform(1.0, smax, size=d)
    off = 0.8 * rng.uniform(-1.0, 1.0, size=d)
    Xt = (centers[yt] + rng.normal(scale=sep + 0.3, size=(n_t, d))) * scales + off

    Xv, yv, Xte, yte = _val_sample(Xt, yt, n_val, rng)
    return DomainSplit("scaled_blobs", Xs, ys, Xte, yte, Xv, yv, meta={"scales": scales})


def make_label_shift_digits(n_val: int = 20, seed: int = 42, tilt: float = 2.5) -> DomainSplit:
    """Digits (classes 0..4, 8x8 -> 32 dims via downsample-free flatten of 8x8).

    Target reweights class proportions with a Dirichlet-style tilt (label
    shift) plus mild isotropic covariate noise.
    """
    from sklearn.datasets import load_digits

    rng = np.random.default_rng(seed)
    data = load_digits()
    keep = data.target <= 4
    X_all = data.data[keep].astype(float) / 16.0
    y_all = data.target[keep]

    # both domains see all 5 classes; only the class PRIORS shift (label shift)
    src_priors = np.full(5, 1.0 / 5)
    src_counts = rng.multinomial(400, src_priors)
    Xs_parts, ys_parts = [], []
    for cls in range(5):
        pool = X_all[y_all == cls]
        take = min(src_counts[cls], pool.shape[0] // 2)
        idx = rng.choice(pool.shape[0], size=take, replace=False)
        Xs_parts.append(pool[idx])
        ys_parts.append(np.full(take, cls))
    Xs = np.vstack(Xs_parts)
    ys = np.concatenate(ys_parts)

    priors = np.array([0.30, 0.25, 0.20, 0.10 + tilt * 0.05, 0.05 + tilt * 0.10])
    priors = priors / priors.sum()
    counts = rng.multinomial(400, priors)
    Xt_parts, yt_parts = [], []
    for cls in range(5):
        pool = X_all[y_all == cls]
        idx = rng.choice(pool.shape[0], size=min(counts[cls], pool.shape[0]), replace=False)
        Xt_parts.append(pool[idx])
        yt_parts.append(np.full(idx.shape[0], cls))
    # covariate nuisance on target: pixel noise + brightness shift
    Xt = np.clip(
        np.vstack(Xt_parts) + rng.normal(scale=0.08, size=(sum(counts), X_all.shape[1])) + 0.03,
        0.0,
        None,
    )
    yt = np.concatenate(yt_parts)

    Xv, yv, Xte, yte = _val_sample(Xt, yt, n_val, rng)
    return DomainSplit(
        "label_shift_digits",
        Xs,
        ys,
        Xte,
        yte,
        Xv,
        yv,
        meta={"priors_target": priors.tolist()},
    )


def make_all(n_target_val: int = 20, seed: int = 42) -> list[DomainSplit]:
    """The four standard benchmark splits (deterministic given seed)."""
    return [
        make_covariate_shift(n_val=n_target_val, seed=seed),
        make_rotated_moons(n_val=n_target_val, seed=seed),
        make_scaled_blobs(n_val=n_target_val, seed=seed),
        make_label_shift_digits(n_val=n_target_val, seed=seed),
    ]


def validate_split(split: DomainSplit) -> None:
    """Raise InvalidDataError when a split violates the contract."""
    if split.X_source.shape[0] == 0 or split.X_target.shape[0] == 0:
        raise InvalidDataError("source or target is empty")
    if split.X_source.shape[1] != split.X_target.shape[1]:
        raise DimensionMismatchLike()
    classes = np.unique(np.concatenate([split.y_source, split.y_target_val]))
    if classes.size < 2:
        raise InvalidDataError("need at least 2 classes")


def DimensionMismatchLike() -> Exception:  # noqa: N802 - small factory
    from domainforge.core.errors import DimensionMismatchError

    return DimensionMismatchError("source/target feature dims differ")

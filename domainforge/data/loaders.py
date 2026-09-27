"""File loaders for DomainForge (author: 晨星).

Supports .npz (keys Xs, ys, Xt, yt) and .csv with columns feature_* + label +
domain (0=source, 1=target). The tiny labeled target val sample is carved out
deterministically with the given seed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from domainforge.core.errors import InvalidDataError
from domainforge.core.types import DomainSplit


def load_npz(path: str | Path, n_target_val: int = 20, seed: int = 42) -> DomainSplit:
    p = Path(path)
    if not p.exists():
        raise InvalidDataError(f"npz not found: {p}")
    with np.load(p) as z:
        Xs, ys = z["Xs"], z["ys"]
        Xt, yt = z["Xt"], z["yt"]
    if Xs.ndim != 2 or Xt.ndim != 2 or Xs.shape[1] != Xt.shape[1]:
        raise InvalidDataError("npz arrays malformed")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(Xt.shape[0])
    n_val = max(2, min(n_target_val, Xt.shape[0] // 2))
    v, t = idx[:n_val], idx[n_val:]
    return DomainSplit(p.stem, Xs, ys, Xt[t], yt[t], Xt[v], yt[v])


def load_csv(path: str | Path, n_target_val: int = 20, seed: int = 42) -> DomainSplit:
    p = Path(path)
    if not p.exists():
        raise InvalidDataError(f"csv not found: {p}")
    rows: list[list[str]] = []
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            rows.append(row)
    try:
        li, di = header.index("label"), header.index("domain")
    except ValueError as exc:
        raise InvalidDataError("csv must contain 'label' and 'domain' columns") from exc
    fi = [i for i in range(len(header)) if header[i].startswith("feature_")]
    if not fi:
        raise InvalidDataError("no feature_* columns")
    arr = np.array([[float(r[i]) for i in fi] for r in rows])
    lab = np.array([int(float(r[li])) for r in rows])
    dom = np.array([int(float(r[di])) for r in rows])
    Xs, ys, Xt, yt = arr[dom == 0], lab[dom == 0], arr[dom == 1], lab[dom == 1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(Xt.shape[0])
    n_val = max(2, min(n_target_val, Xt.shape[0] // 2))
    v, t = idx[:n_val], idx[n_val:]
    return DomainSplit(p.stem, Xs, ys, Xt[t], yt[t], Xt[v], yt[v])

"""Benchmark pipeline (author: 晨星).

Flow: run(split, method_name) -> BenchmarkRow; benchmark(splits, methods) ->
list of rows + aggregate table. Deterministic (fixed seeds), per-dataset rows
are independent. Optional backends degrade to status="skipped" rows, never
fake numbers.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from domainforge.baselines.adapt_backend import available_adapt
from domainforge.core.errors import BenchmarkError, UnknownMethodError
from domainforge.core.types import BenchmarkRow, DomainSplit
from domainforge.data.synthetic import validate_split
from domainforge.eval.metrics import evaluate, linear_mmd2
from domainforge.registry import CORE_METHODS, OPTIONAL_METHODS, build_method


def _resolve_methods(methods: Iterable[str]) -> list[str]:
    names = list(methods)
    if names == ["all"]:
        names = CORE_METHODS + ["safuse"]
    if names == ["core"]:
        names = list(CORE_METHODS)
    out = []
    for n in names:
        if n in OPTIONAL_METHODS and not available_adapt():
            continue  # optional backend unavailable -> row omitted, logged by caller
        out.append(n)
    return out


def run(
    split: DomainSplit, method_name: str, do_hpo_safuse: bool = True, seed: int = 42
) -> BenchmarkRow:
    """Run one method on one split; never raises - errors become status rows."""
    t0 = time.perf_counter()
    try:
        validate_split(split)
        if method_name == "adapt_coral":
            from domainforge.baselines.adapt_backend import AdaptCoralBackend

            model = AdaptCoralBackend(seed=seed).fit(split)
        elif method_name == "safuse":
            from domainforge.fusion.flagship import SafuseAdapter

            model = SafuseAdapter(do_hpo=do_hpo_safuse, seed=seed).fit(split)
        else:
            model = build_method(method_name, seed=seed).fit(split)
        pred = model.predict(split.X_target)
        acc, f1 = evaluate(split.y_target_private, pred)
        try:
            Zs = model.transform(split.X_source)
            Zt = model.transform(split.X_target)
            mmd = linear_mmd2(np.asarray(Zs), np.asarray(Zt))
        except Exception:
            mmd = float("nan")
        fit_s = time.perf_counter() - t0
        detail = ""
        if method_name == "safuse" and hasattr(model, "mode_"):
            detail = str(model.mode_)
        return BenchmarkRow(
            split.name, method_name, acc, f1, mmd, fit_s, status="ok", detail=detail
        )
    except UnknownMethodError:
        raise
    except Exception as exc:
        return BenchmarkRow(
            split.name,
            method_name,
            0.0,
            0.0,
            float("nan"),
            time.perf_counter() - t0,
            status="error",
            detail=f"{type(exc).__name__}: {exc}",
        )


def run_optional_row(
    split: DomainSplit, method_name: str, seed: int = 42
) -> BenchmarkRow:
    """Optional-backend row: unavailable -> skipped (honest placeholder)."""
    if method_name in OPTIONAL_METHODS and not available_adapt():
        return BenchmarkRow(
            split.name,
            method_name,
            0.0,
            0.0,
            float("nan"),
            0.0,
            status="skipped",
            detail="adapt/TF backend unavailable",
        )
    return run(split, method_name, seed=seed)


def benchmark(
    splits: Iterable[DomainSplit], methods: Iterable[str] = ("all",), seed: int = 42
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for split in splits:
        for name in _resolve_methods(methods):
            if name == "adapt_coral":
                rows.append(run_optional_row(split, name, seed=seed))
            else:
                rows.append(run(split, name, seed=seed))
    if not rows:
        raise BenchmarkError("no benchmark rows produced")
    return rows


def aggregate(rows: list[BenchmarkRow]) -> dict[str, dict[str, float]]:
    """Mean accuracy / macro-F1 per method across datasets (ok rows only)."""
    acc: dict[str, list[float]] = {}
    f1: dict[str, list[float]] = {}
    for r in rows:
        if r.status == "ok":
            acc.setdefault(r.method, []).append(r.accuracy)
            f1.setdefault(r.method, []).append(r.macro_f1)
    return {
        m: {"mean_acc": float(np.mean(v)), "mean_f1": float(np.mean(f1[m]))}
        for m, v in acc.items()
    }


def format_table(rows: list[BenchmarkRow]) -> str:
    """Fixed-width table (SOP: never rely on CJK-width auto alignment)."""
    header = ["dataset", "method", "acc", "macro_f1", "mmd2", "sec", "status"]
    lines = [
        " ".join(f"{h:<18}" if i == 0 else f"{h:<12}" for i, h in enumerate(header))
    ]
    for r in rows:
        cells = [
            r.dataset,
            r.method,
            f"{r.accuracy:.4f}",
            f"{r.macro_f1:.4f}",
            f"{r.mmd2:.4f}" if np.isfinite(r.mmd2) else "-",
            f"{r.fit_seconds:.2f}",
            r.status,
        ]
        lines.append(
            " ".join(f"{c:<18}" if i == 0 else f"{c:<12}" for i, c in enumerate(cells))
        )
    return "\n".join(lines)


def save_rows(rows: list[BenchmarkRow], path: str | Path) -> Path:
    p = Path(path)
    payload: dict[str, Any] = {
        "rows": [r.as_dict() for r in rows],
        "aggregate": aggregate(rows),
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p

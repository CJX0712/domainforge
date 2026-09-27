"""End-to-end demo: synthetic domains -> cross-method benchmark -> JSON
(author: 晨星). Run:  python -m domainforge.examples.run_demo
"""

from __future__ import annotations

from pathlib import Path

from domainforge.core.config import CONFIG
from domainforge.data.synthetic import make_all
from domainforge.pipeline.pipeline import aggregate, benchmark, format_table, save_rows


def main() -> None:
    splits = make_all(n_target_val=CONFIG.n_target_val, seed=CONFIG.seed)
    rows = benchmark(splits, ("all",))
    print(format_table(rows))
    agg = aggregate(rows)
    print("\n== aggregate (mean over 4 datasets) ==")
    for m, v in sorted(agg.items(), key=lambda kv: -kv[1]["mean_acc"]):
        print(f"{m:<14} mean_acc={v['mean_acc']:.4f}  mean_f1={v['mean_f1']:.4f}")
    out = save_rows(rows, Path(__file__).resolve().parents[2] / "benchmark.json")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

"""DomainForge CLI (author: 晨星).

python -m domainforge.cli demo                     # full synthetic benchmark
python -m domainforge.cli benchmark --out bench.json
python -m domainforge.cli run --npz data.npz --method tca
"""

from __future__ import annotations

import argparse
import sys

from domainforge import __version__
from domainforge.core.config import CONFIG
from domainforge.data.synthetic import make_all
from domainforge.pipeline.pipeline import aggregate, benchmark, format_table, save_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="domainforge", description="DomainForge: 域自适应系统"
    )
    parser.add_argument(
        "--version", action="version", version=f"domainforge {__version__}"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="run the end-to-end synthetic benchmark")

    p_bench = sub.add_parser("benchmark", help="benchmark and save JSON")
    p_bench.add_argument("--out", default="benchmark.json")
    p_bench.add_argument("--methods", nargs="*", default=["all"])
    p_bench.add_argument("--seed", type=int, default=None)

    p_run = sub.add_parser("run", help="adapt one npz split with one method")
    p_run.add_argument("--npz", required=True)
    p_run.add_argument("--method", default="tca")

    args = parser.parse_args(argv)
    seed = CONFIG.seed if args.cmd in ("demo",) else getattr(args, "seed", None)

    if args.cmd == "demo":
        rows = benchmark(
            make_all(n_target_val=CONFIG.n_target_val, seed=CONFIG.seed), ("all",)
        )
        print(format_table(rows))
        agg = aggregate(rows)
        print("\n== aggregate (mean over 4 datasets) ==")
        for m, v in sorted(agg.items(), key=lambda kv: -kv[1]["mean_acc"]):
            print(f"{m:<14} mean_acc={v['mean_acc']:.4f}  mean_f1={v['mean_f1']:.4f}")
        return 0
    if args.cmd == "benchmark":
        s = seed if seed is not None else CONFIG.seed
        rows = benchmark(
            make_all(n_target_val=CONFIG.n_target_val, seed=s), tuple(args.methods)
        )
        out = save_rows(rows, args.out)
        print(format_table(rows))
        print(f"\nsaved -> {out}")
        return 0
    if args.cmd == "run":
        from domainforge.data.loaders import load_npz
        from domainforge.pipeline.pipeline import run

        split = load_npz(args.npz, n_target_val=CONFIG.n_target_val, seed=CONFIG.seed)
        row = run(split, args.method)
        print(format_table([row]))
        return 0 if row.status == "ok" else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())

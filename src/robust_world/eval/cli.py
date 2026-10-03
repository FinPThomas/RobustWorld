"""robustworld-eval: sample clips, run the baseline, build comparison grids.

    robustworld-eval sample --name sample5 --n 5 --seed 0
    robustworld-eval baseline --sample data/eval/sample5
    robustworld-eval grid --sample data/eval/sample5 --models cosmos_predict2 wan21
    robustworld-eval ball-compare          # V-JEPA 2 vs TAPNext rules on ball-level measures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..paths import REPO_ROOT
from . import ball_compare, baselines, grid, sample

PRED_ROOT = REPO_ROOT / "outputs" / "predictions"
GRID_ROOT = REPO_ROOT / "outputs" / "grids"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="robustworld-eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="draw random included clips into data/eval/<name>/")
    s.add_argument("--name", default="sample5")
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)

    b = sub.add_parser("baseline", help="hold-last-frame predictions (no GPU needed)")
    b.add_argument("--sample", type=Path, default=sample.EVAL_ROOT / "sample5")
    b.add_argument("--pred-root", type=Path, default=PRED_ROOT)

    g = sub.add_parser("grid", help="comparison videos: ground truth + the given models")
    g.add_argument("--sample", type=Path, default=sample.EVAL_ROOT / "sample5")
    g.add_argument("--models", nargs="+", required=True,
                   help="prediction folder names under <pred-root>/<sample>/, in column order")
    g.add_argument("--pred-root", type=Path, default=PRED_ROOT)
    g.add_argument("--out", type=Path, help="default: outputs/grids/<sample>/<models joined by +>")
    g.add_argument("--cell", type=int, default=256, help="pixel size of each panel")

    sub.add_parser("ball-compare", help="ball-level comparison table + plot across models")

    args = p.parse_args(argv)
    if args.cmd == "sample":
        sample.make_sample(args.name, args.n, args.seed)
    elif args.cmd == "baseline":
        baselines.hold_last_frame(args.sample, args.pred_root / args.sample.name / "hold")
    elif args.cmd == "grid":
        out = args.out or GRID_ROOT / args.sample.name / "+".join(args.models)
        grid.build(args.sample, args.pred_root / args.sample.name, args.models, out, args.cell)
    elif args.cmd == "ball-compare":
        ball_compare.compare()
    return 0


if __name__ == "__main__":
    sys.exit(main())

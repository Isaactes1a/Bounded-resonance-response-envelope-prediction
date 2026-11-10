"""Run the current eight-configuration workflow for all three cases.

The detailed experiment implementation lives in ``run_frequency_validation.py``;
this module is only the batch orchestrator.  Keeping the common arguments here
prevents the three cases from silently using different meshes or sweep ranges.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--out", type=Path, default=Path("results") / "unified_square_cases_modal_sweep_cuda")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sweep-points", type=int, default=801)
    parser.add_argument("--sweep-frequency-min", type=float, default=0.0)
    parser.add_argument("--sweep-frequency-max", type=float, default=200.0)
    parser.add_argument("--refine-steps", type=int, default=500)
    parser.add_argument("--pod-refine-steps", type=int, default=500)
    args = parser.parse_args()
    if args.sweep_points < 2:
        parser.error("--sweep-points must be at least two")
    if args.sweep_frequency_max <= args.sweep_frequency_min:
        parser.error("--sweep-frequency-max must exceed --sweep-frequency-min")
    if args.refine_steps < 0 or args.pod_refine_steps < 0:
        parser.error("refinement steps must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    for case in ("flat", "annular", "stiffened"):
        command = [
            sys.executable,
            str(root / "run_frequency_validation.py"),
            "--case",
            case,
            "--epochs",
            str(args.epochs),
            "--n-labeled",
            "48",
            "--n-physics",
            "160",
            "--frequency-min",
            "0",
            "--frequency-max",
            "200",
            "--modal-resonance",
            "--modal-modes",
            "8",
            "--compare-pod",
            "--pod-gradient-balance",
            "--refine-steps",
            str(args.refine_steps),
            "--pod-refine-steps",
            str(args.pod_refine_steps),
            "--square-ndiv",
            "30",
            "--plate-stencil",
            "20",
            "--plate-poly-degree",
            "2",
            "--beam-stencil",
            "7",
            "--hollow-sweep-points",
            "15",
            "--hollow-tangent-points-per-patch",
            "16",
            "--hollow-stencil",
            "20",
            "--hollow-poly-degree",
            "2",
            "--hole-diameter-ratio",
            "0.25",
            "--rbf-condition-limit",
            "1.0e8",
            "--device",
            args.device,
            "--sweep-frequency-min",
            str(args.sweep_frequency_min),
            "--sweep-frequency-max",
            str(args.sweep_frequency_max),
            "--sweep-points",
            str(args.sweep_points),
            "--out",
            str(args.out),
        ]
        print("RUN:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

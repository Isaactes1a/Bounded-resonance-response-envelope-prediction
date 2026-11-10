# -*- coding: utf-8 -*-
"""Run the controlled four-model parametric ablation used in Section 6.4.

Models
------
direct_data      : no modal-rational branch, physics weight = 0
direct_physics   : no modal-rational branch, physics weight > 0
pole_data        : modal-rational branch, physics weight = 0
full             : modal-rational branch, physics weight > 0

All variants use the same seed, parameter counts, frequency grids and network
settings supplied through ``--extra`` so that only the two method components
change.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


VARIANTS = {
    "direct_data": ("--disable-modal-rational", "--physics-weight", "0"),
    "direct_physics": ("--disable-modal-rational",),
    "pole_data": ("--physics-weight", "0"),
    "full": (),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("flat", "hollow", "annular", "stiffened", "all"), default="all")
    parser.add_argument("--out", type=Path, default=Path("results") / "parametric_ablation")
    parser.add_argument("--physics-weight", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=(),
        help="Arguments forwarded to run_parametric_many_query.py after '--extra'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).resolve().with_name("run_parametric_many_query.py")
    cases = ("flat", "annular", "stiffened") if args.case == "all" else (args.case,)
    commands = []
    for case in cases:
        for variant, switches in VARIANTS.items():
            command = [
                sys.executable,
                str(script),
                "--case",
                case,
                "--out",
                str(args.out / variant),
                "--physics-weight",
                str(args.physics_weight),
            ]
            command.extend(switches)
            command.extend(args.extra)
            commands.append(command)

    manifest = args.out / "ablation_commands.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "\n".join(" ".join(shlex.quote(token) for token in command) for command in commands) + "\n",
        encoding="utf-8",
    )
    for command in commands:
        print(" ".join(shlex.quote(token) for token in command))
        if not args.dry_run:
            subprocess.run(command, check=True)
    print(f"Ablation command manifest: {manifest}")


if __name__ == "__main__":
    main()

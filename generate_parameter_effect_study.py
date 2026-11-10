# -*- coding: utf-8 -*-
"""Generate high-fidelity single-parameter FRF sensitivity data for Section 6.2.

The study varies one of ``s_E``, ``s_rho``, ``s_p`` and ``eta`` at a time while
the other parameters remain at the center of the configured training bounds.
It uses the direct constrained RBF-FD family rather than the surrogate so the
figure documents the physical effect that motivates the modal-rational branch.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rbf_fd_full_order.visualization import apply_journal_style, save_figure, style_axis
from rbf_fd_full_order.visualization.result_support import load_parametric_case

CASES = ("flat", "annular", "stiffened")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results") / "parametric_many_query")
    parser.add_argument("--out", type=Path, default=Path("results") / "parameter_effects")
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sweep-points", type=int, default=801)
    parser.add_argument(
        "--levels",
        type=float,
        nargs="*",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
        help="Relative locations between each configured lower and upper bound.",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_case(case: str, args: argparse.Namespace) -> None:
    source = args.results_root / case
    output = args.out / case
    output.mkdir(parents=True, exist_ok=True)
    metadata, system, operators, family, result, observation_index = load_parametric_case(source, device=args.device)
    bounds = family.bounds
    fmin, fmax = [float(value) for value in metadata["frequency_interval_hz"]]
    frequency = np.linspace(fmin, fmax, int(args.sweep_points))
    center = np.asarray(bounds.center, dtype=float)
    levels = np.asarray(args.levels, dtype=float)
    if np.any((levels < 0.0) | (levels > 1.0)):
        raise ValueError("--levels must lie in [0, 1].")

    rows = []
    curves = {}
    for parameter_index, name in enumerate(bounds.names):
        values = bounds.lower[parameter_index] + levels * (
            bounds.upper[parameter_index] - bounds.lower[parameter_index]
        )
        param_curves = []
        for level, value in zip(levels, values):
            mu = center.copy()
            mu[parameter_index] = value
            response = family.solve_grid(frequency, mu)
            curve = np.abs(response[:, observation_index]) * 1.0e3
            param_curves.append(curve)
            peak_index = int(np.argmax(curve))
            for i, f in enumerate(frequency):
                rows.append(
                    {
                        "case": case,
                        "parameter": name,
                        "relative_level": float(level),
                        "parameter_value": float(value),
                        "frequency_hz": float(f),
                        "response_mm": float(curve[i]),
                        "peak_frequency_hz": float(frequency[peak_index]),
                        "peak_amplitude_mm": float(curve[peak_index]),
                    }
                )
        curves[name] = (values, np.asarray(param_curves))
    _write_csv(output / "parameter_effect_frf.csv", rows)

    apply_journal_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.42, 5.1), constrained_layout=True)
    palette = ("#0070C0", "#808080", "#000000", "#808080", "#C00000")
    styles = ("--", ":", "-", "-.", "--")
    for ax, (name, (values, param_curves)) in zip(axes.ravel(), curves.items()):
        for idx, (value, curve) in enumerate(zip(values, param_curves)):
            ax.plot(
                frequency,
                curve,
                color=palette[min(idx, len(palette) - 1)],
                linestyle=styles[min(idx, len(styles) - 1)],
                lw=1.0 if idx != 2 else 1.35,
                label=f"{value:.4g}",
            )
        ax.set_title(name.replace("_scale", " scale").replace("loss_factor", r"$\eta$"))
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel(r"$|w|$ (mm)")
        style_axis(ax)
        ax.legend(title="value", fontsize=8, title_fontsize=8, ncol=2)
    save_figure(fig, output / "parameter_effect_frf.png", tight=False)
    plt.close(fig)

    summary = {
        "case": case,
        "purpose": "High-fidelity physical interpretation of each uncertain parameter",
        "parameter_names": list(bounds.names),
        "levels": levels.tolist(),
        "frequency_points": int(frequency.size),
        "observation_constrained_index": int(observation_index),
    }
    with (output / "parameter_effect_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    cases = CASES if args.case == "all" else (args.case,)
    for case in cases:
        run_case(case, args)
    print(f"Wrote parameter-effect studies to {args.out}")


if __name__ == "__main__":
    main()

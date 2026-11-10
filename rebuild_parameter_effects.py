# -*- coding: utf-8 -*-
"""Thread-parallel single-parameter FRF sensitivity study.

Same output as ``generate_parameter_effect_study.py`` — the CSV consumed by
Figure 5 (``parameter_effect_frf.csv``) plus the four-panel overview — but the
``n_parameters x n_levels`` direct sweeps are evaluated on a thread pool instead
of one after another.  The scipy sparse solve releases the GIL, so the sweeps
overlap.

The frequency band is read from the case metadata, so the study follows a model
whose band has been extended.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from rbf_fd_full_order.visualization.result_support import load_parametric_case  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path,
                        default=Path("results/parametric_many_query_stiffened_5000_best/stiffened"))
    parser.add_argument("--out", type=Path, default=Path("results/parameter_effects"))
    parser.add_argument("--sweep-points", type=int, default=601)
    parser.add_argument("--levels", type=float, nargs="*",
                        default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    metadata, _, _, family, result, observation_index = load_parametric_case(
        args.case_dir, device=args.device)
    case = str(metadata.get("case", args.case_dir.name))
    bounds = family.bounds
    fmin, fmax = (float(value) for value in metadata["frequency_interval_hz"])
    frequency = np.linspace(fmin, fmax, args.sweep_points)
    centre = np.asarray(bounds.center, dtype=float)
    levels = np.asarray(args.levels, dtype=float)

    jobs = []
    for index, name in enumerate(bounds.names):
        values = bounds.lower[index] + levels * (bounds.upper[index] - bounds.lower[index])
        for level, value in zip(levels, values):
            mu = centre.copy()
            mu[index] = value
            jobs.append((name, float(level), float(value), mu))

    print(f"case={case}  band={fmin:.0f}-{fmax:.0f} Hz  points={args.sweep_points}")
    print(f"{len(jobs)} single-parameter sweeps, {args.workers} threads", flush=True)

    def solve(job):
        response = family.solve_grid(frequency, job[3])
        return np.abs(response[:, observation_index]) * 1.0e3

    curves = [None] * len(jobs)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(solve, job): index for index, job in enumerate(jobs)}
        for future in as_completed(futures):
            curves[futures[future]] = future.result()
            done += 1
            print(f"  sweep {done}/{len(jobs)}", flush=True)

    rows = []
    for (name, level, value, _), curve in zip(jobs, curves):
        peak = int(np.argmax(curve))
        for i, f in enumerate(frequency):
            rows.append({
                "case": case,
                "parameter": name,
                "relative_level": level,
                "parameter_value": value,
                "frequency_hz": float(f),
                "response_mm": float(curve[i]),
                "peak_frequency_hz": float(frequency[peak]),
                "peak_amplitude_mm": float(curve[peak]),
            })

    output = args.out / case
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "parameter_effect_frf.csv", rows)
    summary = {
        "case": case,
        "purpose": "High-fidelity physical interpretation of each uncertain parameter",
        "parameter_names": list(bounds.names),
        "levels": levels.tolist(),
        "frequency_points": int(frequency.size),
        "frequency_band_hz": [fmin, fmax],
        "observation_constrained_index": int(observation_index),
        "parallelism": f"thread pool, {args.workers} workers",
    }
    (output / "parameter_effect_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"written {output / 'parameter_effect_frf.csv'}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()

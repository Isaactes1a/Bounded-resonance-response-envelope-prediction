# -*- coding: utf-8 -*-
"""Thread-parallel rebuild of the 40 x 40 calibration archive.

Identical in logic to ``recalibrate_stiffened_curve_coverage.py``, but the
high-fidelity observation curves are evaluated with a thread pool instead of a
spawn process pool.  The scipy sparse solve releases the GIL, so threads give
real parallelism, and threads need no named pipes (which the execution sandbox
does not grant to child processes).

Sampling, seeds, calibration logic and every written artefact match the
original script, so downstream consumers see the same file layout:

    <out>/ellipsoidal_calibration_40x40.npz
    <out>/ellipsoidal_calibration_40x40_summary.json
    <out>/ellipsoidal_calibration_reliability.csv
    <out>/ellipsoidal_calibration_example.csv
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

from rbf_fd_full_order import (  # noqa: E402
    CurveConformalCalibrator,
    EllipsoidalConvexSet,
    correlation_matrix,
    empirical_curve_coverage,
    predict_parametric_observation_members,
)
from rbf_fd_full_order.visualization.result_support import load_parametric_case  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir", type=Path,
        default=Path("results/parametric_many_query_stiffened_5000_best/stiffened"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/ellipsoidal_uq/stiffened"))
    parser.add_argument("--n-calibration", type=int, default=40)
    parser.add_argument("--n-test", type=int, default=40)
    parser.add_argument("--sweep-points", type=int, default=601)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--conformal-scale-floor", type=float, default=0.02)
    parser.add_argument("--archive-name", default="ellipsoidal_calibration_40x40.npz")
    args = parser.parse_args()
    if min(args.n_calibration, args.n_test, args.sweep_points, args.workers) < 1:
        parser.error("sample counts, sweep points and workers must be positive")
    return args


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metadata, _, _, family, result, observation_index = load_parametric_case(
        args.case_dir, device="cpu")
    case = str(metadata.get("case", args.case_dir.name))
    fmin, fmax = (float(value) for value in metadata["frequency_interval_hz"])
    frequencies = np.linspace(fmin, fmax, args.sweep_points)
    bounds = family.bounds
    ellipsoid = EllipsoidalConvexSet(
        center=np.asarray(bounds.center, dtype=float),
        scales=0.5 * (np.asarray(bounds.upper, dtype=float)
                      - np.asarray(bounds.lower, dtype=float)),
        correlation=correlation_matrix(bounds.dimension, e_rho=0.0),
    ).validated()
    calibration_mu = ellipsoid.sample(args.n_calibration, args.seed + 33,
                                      boundary_fraction=0.5)
    test_mu = ellipsoid.sample(args.n_test, args.seed + 44, boundary_fraction=0.5)
    all_mu = np.vstack((calibration_mu, test_mu))

    print(f"case={case}  band={fmin:.0f}-{fmax:.0f} Hz  points={args.sweep_points}")
    print(f"solving {all_mu.shape[0]} high-fidelity observation curves "
          f"with {args.workers} threads", flush=True)

    def solve_one(parameter: np.ndarray) -> np.ndarray:
        response = family.solve_grid(frequencies, np.asarray(parameter)[None, :])
        return np.abs(response[0, :, observation_index]) * 1.0e3

    done = 0
    curves = np.empty((all_mu.shape[0], frequencies.size), dtype=float)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(solve_one, mu): index
                   for index, mu in enumerate(all_mu)}
        for future in as_completed(futures):
            curves[futures[future]] = future.result()
            done += 1
            if done % 5 == 0 or done == all_mu.shape[0]:
                print(f"  solved {done}/{all_mu.shape[0]}", flush=True)

    calibration_hf = curves[: args.n_calibration]
    test_hf = curves[args.n_calibration:]
    calibration_members = np.abs(
        predict_parametric_observation_members(result, frequencies, calibration_mu,
                                               observation_index)) * 1.0e3
    test_members = np.abs(
        predict_parametric_observation_members(result, frequencies, test_mu,
                                               observation_index)) * 1.0e3

    reliability_rows = []
    bands = {}
    for confidence in (0.80, 0.90, 0.95):
        calibrator = CurveConformalCalibrator.fit(
            calibration_members, calibration_hf, confidence=confidence,
            relative_scale_floor=args.conformal_scale_floor)
        mean, _, lower, upper = calibrator.band(test_members)
        covered = np.all((test_hf >= lower) & (test_hf <= upper), axis=1)
        reliability_rows.append({
            "nominal_curve_coverage": confidence,
            "calibration_empirical_curve_coverage": calibrator.empirical_curve_coverage,
            "test_empirical_curve_coverage": empirical_curve_coverage(test_hf, lower, upper),
            "covered_test_curves": int(np.sum(covered)),
            "test_curve_count": args.n_test,
            "conformal_multiplier": calibrator.multiplier,
            "mean_test_band_width_mm": float(np.mean(upper - lower)),
        })
        bands[confidence] = (mean, lower, upper)

    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "ellipsoidal_calibration_reliability.csv", reliability_rows)
    representative = int(np.argsort(
        np.linalg.norm(test_hf - np.mean(test_members, axis=0), axis=1))[args.n_test // 2])
    mean95, lower95, upper95 = bands[0.95]
    example_rows = [
        {
            "frequency_hz": float(frequency),
            "hf_mm": float(test_hf[representative, index]),
            "ensemble_mean_mm": float(mean95[representative, index]),
            "calibrated_lower_mm": float(lower95[representative, index]),
            "calibrated_upper_mm": float(upper95[representative, index]),
        }
        for index, frequency in enumerate(frequencies)
    ]
    _write_csv(args.out / "ellipsoidal_calibration_example.csv", example_rows)
    np.savez_compressed(
        args.out / args.archive_name,
        frequencies_hz=frequencies,
        calibration_parameters=calibration_mu,
        test_parameters=test_mu,
        calibration_hf_mm=calibration_hf,
        test_hf_mm=test_hf,
    )
    summary = {
        "case": case,
        "network_retrained": False,
        "calibration_sample_count": args.n_calibration,
        "test_sample_count": args.n_test,
        "frequency_point_count": args.sweep_points,
        "frequency_band_hz": [fmin, fmax],
        "independent_parameter_sets": True,
        "seed": args.seed,
        "parallelism": f"thread pool, {args.workers} workers",
        "reliability": reliability_rows,
    }
    (args.out / "ellipsoidal_calibration_40x40_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

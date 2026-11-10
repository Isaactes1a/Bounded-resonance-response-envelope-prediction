# -*- coding: utf-8 -*-
"""Ellipsoid-constrained extremum search on the surrogate with direct back-substitution.

The review requested that sample-estimated UQ envelopes be complemented by a
dedicated extremum search and direct RBF-FD back-substitution at the predicted
extremal parameter points.  This script:

1. builds the same 1000-realization surrogate envelope that is reported in the
   manuscript and a 2000-point low-discrepancy interior/boundary scan;
2. refines four targets with differential evolution in the normalized
   unit-ball coordinate xi (with a feasibility penalty on ||xi|| <= 1):
     - dominant peak amplitude over (mu, f);
     - envelope upper value at the first-resonance frequency;
     - dominant resonance frequency upper bound;
     - envelope lower value at the first-resonance frequency;
3. back-substitutes the found parameter points into the direct RBF-FD solver
   on the full 201-point grid and reports surrogate-vs-direct agreement;
4. compares the surrogate-searched extrema with the observed extrema of the
   80 direct curves retained for calibration and testing (40 + 40);
5. reports Wilson binomial intervals for the curve-wise coverage counts.

Outputs are written to ``results/ellipsoid_extremum_search/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from rbf_fd_full_order import (
    EllipsoidalConvexSet,
    correlation_matrix,
    predict_parametric_observation_members,
)
from rbf_fd_full_order.visualization.result_support import load_parametric_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", choices=("flat", "stiffened"), default=("flat", "stiffened")
    )
    parser.add_argument("--out", type=Path, default=Path("results") / "ellipsoid_extremum_search")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--uq-root", type=Path, default=Path("results") / "ellipsoidal_uq")
    parser.add_argument("--scan-points", type=int, default=2000)
    parser.add_argument("--de-population", type=int, default=24)
    parser.add_argument("--de-iterations", type=int, default=80)
    parser.add_argument("--max-backsubstitutions", type=int, default=3)
    parser.add_argument("--sweep-points", type=int, default=201)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)


def _parabolic_peak(frequencies: np.ndarray, curve: np.ndarray) -> float:
    index = int(np.argmax(curve))
    if index <= 0 or index >= curve.size - 1:
        return float(frequencies[index])
    y = np.log10(np.maximum(curve[index - 1 : index + 2], 1.0e-30))
    denominator = y[0] - 2.0 * y[1] + y[2]
    if abs(denominator) < 1.0e-14:
        return float(frequencies[index])
    step = float(np.median(np.diff(frequencies)))
    offset = 0.5 * (y[0] - y[2]) / denominator
    return float(frequencies[index] + offset * step)


def wilson_interval(covered: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054  # norm.ppf(0.975)
    p_hat = covered / total
    center = (covered + z * z / 2.0) / (total + z * z)
    half = z / (1.0 + z * z / total) * np.sqrt(
        p_hat * (1.0 - p_hat) / total + z * z / (4.0 * total * total)
    )
    return float(center - half), float(center + half)


def run_case(case: str, args: argparse.Namespace, device: str) -> dict[str, object]:
    out_dir = args.out / case
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.results_root / f"parametric_many_query_{case}_5000_best" / case
    metadata, _, _, family, result, observation_index = load_parametric_case(source_dir, device=device)
    bounds = family.bounds
    center = np.asarray(bounds.center, dtype=float)
    scales = 0.5 * (np.asarray(bounds.upper, dtype=float) - np.asarray(bounds.lower, dtype=float))
    ellipsoid = EllipsoidalConvexSet(
        center=center,
        scales=scales,
        correlation=correlation_matrix(bounds.dimension, e_rho=0.0),
    ).validated()
    fmin, fmax = (float(value) for value in metadata["frequency_interval_hz"])
    frequencies = np.linspace(fmin, fmax, int(args.sweep_points))
    first_mode_hz = float(metadata.get("modal_resonance_frequencies_hz", [0.0])[0])
    if first_mode_hz <= 0.0:
        first_mode_hz = 48.4 if case == "flat" else 96.8
    target_frequency = first_mode_hz * np.sqrt(center[0] / center[1])
    print(f"[{case}] observation index = {observation_index}, target frequency = {target_frequency:.2f} Hz", flush=True)

    # 1) Low-discrepancy scan (half boundary) for a broad surrogate envelope.
    rng = np.random.default_rng(args.seed)
    direction = rng.normal(size=(args.scan_points, bounds.dimension))
    direction = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-30)
    radius = rng.random(args.scan_points) ** (1.0 / bounds.dimension)
    n_boundary = args.scan_points // 2
    radius[:n_boundary] = 1.0
    xi_scan = direction * radius[:, None]
    mu_scan = ellipsoid.to_physical(xi_scan)
    scan_members = predict_parametric_observation_members(
        result, frequencies, mu_scan, observation_index
    )
    scan_mean = np.mean(np.abs(scan_members), axis=0) * 1.0e3  # (n_scan, n_f)
    scan_lower = np.min(scan_mean, axis=0)
    scan_upper = np.max(scan_mean, axis=0)
    scan_argmin = np.argmin(scan_mean, axis=0)
    scan_argmax = np.argmax(scan_mean, axis=0)

    # 2) Differential-evolution refinement for four targets.
    def surrogate_at_mu(mu: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=float).reshape(1, -1)
        members = predict_parametric_observation_members(result, frequencies, mu, observation_index)
        return np.mean(np.abs(members), axis=0)[0] * 1.0e3

    def surrogate_curve(xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float).reshape(1, -1)
        xi = np.clip(xi, -1.0, 1.0)
        norm = float(np.linalg.norm(xi))
        if norm > 1.0:
            xi = xi / norm
        return surrogate_at_mu(ellipsoid.to_physical(xi))

    def feasible_penalty(xi: np.ndarray) -> float:
        return 1.0e3 * max(float(np.linalg.norm(xi)) - 1.0, 0.0)

    targets = {
        "peak_amplitude": lambda curve: -float(np.max(curve)),
        "envelope_upper_at_f1": lambda curve: -float(curve[np.argmin(np.abs(frequencies - target_frequency))]),
        "resonance_frequency_upper": lambda curve: -_parabolic_peak(frequencies, curve),
        "envelope_lower_at_f1": lambda curve: float(curve[np.argmin(np.abs(frequencies - target_frequency))]),
    }
    refined: dict[str, dict[str, object]] = {}
    for target_name, objective in targets.items():
        def cost(xi: np.ndarray, obj=objective) -> float:
            return float(obj(surrogate_curve(xi))) + feasible_penalty(xi)

        result_de = differential_evolution(
            cost,
            bounds=[(-1.0, 1.0)] * bounds.dimension,
            seed=int(args.seed) + len(refined),
            popsize=int(args.de_population),
            maxiter=int(args.de_iterations),
            tol=1.0e-8,
            polish=True,
            updating="immediate",
            workers=1,
        )
        xi_best = np.asarray(result_de.x, dtype=float)
        mu_best = ellipsoid.to_physical(xi_best[None, :])[0]
        curve_best = surrogate_curve(xi_best)
        refined[target_name] = {
            "xi": xi_best.tolist(),
            "mu": mu_best.tolist(),
            "surrogate_value": float(objective(curve_best) * -1.0 if "upper" in target_name or "peak" in target_name else objective(curve_best)),
            "target": target_name,
        }
        print(
            f"[{case}] refined {target_name}: mu = {np.round(mu_best, 4).tolist()}, "
            f"objective = {refined[target_name]['surrogate_value']:.6f}",
            flush=True,
        )

    # 3) Direct back-substitution at the deduplicated refined parameter points.
    candidates: list[tuple[str, np.ndarray]] = []
    for target_name, payload in refined.items():
        mu = np.asarray(payload["mu"], dtype=float)
        duplicate = False
        for _, existing in candidates:
            if np.linalg.norm(mu - existing) < 1.0e-4:
                duplicate = True
                break
        if not duplicate:
            candidates.append((target_name, mu))
        if len(candidates) >= args.max_backsubstitutions:
            break

    backsub_rows: list[dict[str, object]] = []
    for target_name, mu in candidates:
        print(f"[{case}] back-substituting {target_name} at mu = {np.round(mu, 4).tolist()}", flush=True)
        started = time.perf_counter()
        direct = family.solve_grid(frequencies, mu)
        solve_s = time.perf_counter() - started
        direct_obs = np.abs(direct[:, observation_index]) * 1.0e3
        surrogate_obs = surrogate_at_mu(mu)
        rel_l2 = float(np.linalg.norm(surrogate_obs - direct_obs) / max(float(np.linalg.norm(direct_obs)), 1.0e-30))
        backsub_rows.append(
            {
                "case": case,
                "target": target_name,
                "mu": mu.tolist(),
                "surrogate_peak_mm": float(np.max(surrogate_obs)),
                "direct_peak_mm": float(np.max(direct_obs)),
                "surrogate_peak_frequency_hz": float(_parabolic_peak(frequencies, surrogate_obs)),
                "direct_peak_frequency_hz": float(_parabolic_peak(frequencies, direct_obs)),
                "surrogate_value_at_f1_mm": float(surrogate_obs[np.argmin(np.abs(frequencies - target_frequency))]),
                "direct_value_at_f1_mm": float(direct_obs[np.argmin(np.abs(frequencies - target_frequency))]),
                "curve_relative_l2_percent": float(100.0 * rel_l2),
                "direct_solve_s": float(solve_s),
            }
        )
        print(
            f"[{case}] backsub {target_name}: direct peak = {direct_obs.max():.4f} mm, "
            f"surrogate = {surrogate_obs.max():.4f} mm, curve rel L2 = {100.0 * rel_l2:.4f}%",
            flush=True,
        )

    # 4) Observed direct extrema over the retained 40+40 calibration/test curves.
    archive = np.load(args.uq_root / case / "ellipsoidal_calibration_40x40.npz")
    direct_curves = np.vstack(
        [
            np.asarray(archive["calibration_hf_mm"], dtype=float),
            np.asarray(archive["test_hf_mm"], dtype=float),
        ]
    )
    direct_params = np.vstack(
        [
            np.asarray(archive["calibration_parameters"], dtype=float),
            np.asarray(archive["test_parameters"], dtype=float),
        ]
    )
    direct_lower = np.min(direct_curves, axis=0)
    direct_upper = np.max(direct_curves, axis=0)
    f1_index = int(np.argmin(np.abs(frequencies - target_frequency)))
    observed_summary = {
        "direct_curve_count": int(direct_curves.shape[0]),
        "direct_upper_at_f1_mm": float(direct_upper[f1_index]),
        "direct_lower_at_f1_mm": float(direct_lower[f1_index]),
        "scan_upper_at_f1_mm": float(scan_upper[f1_index]),
        "scan_lower_at_f1_mm": float(scan_lower[f1_index]),
        "direct_max_peak_mm": float(np.max(direct_curves)),
        "scan_max_peak_mm": float(np.max(scan_mean)),
    }
    print(f"[{case}] observed direct envelope at f1: [{direct_lower[f1_index]:.4f}, {direct_upper[f1_index]:.4f}] mm", flush=True)
    print(f"[{case}] 2000-scan surrogate envelope at f1: [{scan_lower[f1_index]:.4f}, {scan_upper[f1_index]:.4f}] mm", flush=True)

    # 5) Wilson intervals for the retained 40-curve coverage counts.
    reliability = csv.DictReader(
        (args.uq_root / case / "ellipsoidal_calibration_reliability.csv").open("r", encoding="utf-8")
    )
    wilson_rows = []
    for row in reliability:
        if float(row["nominal_curve_coverage"]) == 0.95:
            covered = int(row["covered_test_curves"])
            total = int(row["test_curve_count"])
            lower, upper = wilson_interval(covered, total)
            wilson_rows.append(
                {
                    "case": case,
                    "nominal_coverage": 0.95,
                    "covered": covered,
                    "total": total,
                    "empirical_coverage": float(covered / total),
                    "wilson_lower": lower,
                    "wilson_upper": upper,
                }
            )

    summary = {
        "case": case,
        "observation_index": int(observation_index),
        "target_frequency_hz": float(target_frequency),
        "scan_points": int(args.scan_points),
        "refined_targets": refined,
        "backsubstitution": backsub_rows,
        "envelope_comparison": observed_summary,
        "wilson": wilson_rows,
    }
    _write_json(out_dir / "extremum_summary.json", summary)
    _write_csv(out_dir / "backsubstitution.csv", backsub_rows)
    _write_csv(out_dir / "wilson_coverage.csv", wilson_rows)
    print(f"[{case}] wrote {out_dir / 'extremum_summary.json'}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if (args.device == "auto" and __import__("torch").cuda.is_available()) else "cpu"
    if args.device != "auto":
        device = args.device
    summaries = []
    for case in args.cases:
        summaries.append(run_case(case, args, device))
    _write_json(args.out / "extremum_summary_all.json", summaries)
    print(f"Wrote extremum summaries to {args.out}", flush=True)


if __name__ == "__main__":
    main()

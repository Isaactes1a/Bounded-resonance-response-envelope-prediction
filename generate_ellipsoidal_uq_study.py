# -*- coding: utf-8 -*-
"""Generate manuscript-ready ellipsoidal UQ data from a trained parametric operator.

This script is the final-paper UQ companion to ``run_parametric_many_query.py``.
The training script may cover the enclosing parameter box, while this script
restricts uncertainty propagation to the non-probabilistic ellipsoidal convex
set used in the manuscript.

Outputs per case
----------------
ellipsoidal_envelope.csv
ellipsoidal_resonance_ranges.csv
ellipsoidal_sample_convergence.csv
ellipsoidal_correlation_envelopes.csv
ellipsoidal_unseen_test_metrics.csv
ellipsoidal_calibration_reliability.csv
ellipsoidal_calibration_example.csv
ellipsoidal_uq_curves.npz
ellipsoidal_uq_summary.json
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

from rbf_fd_full_order import predict_parametric_members, predict_parametric_observation_members
from rbf_fd_full_order.uncertainty import (
    CurveConformalCalibrator,
    EllipsoidalConvexSet,
    correlation_matrix,
    empirical_curve_coverage,
)
from rbf_fd_full_order.visualization import apply_journal_style, save_figure, style_axis
from rbf_fd_full_order.visualization.result_support import load_parametric_case


CASES = ("flat", "annular", "stiffened")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results") / "parametric_many_query")
    parser.add_argument("--out", type=Path, default=Path("results") / "ellipsoidal_uq")
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-uq", type=int, default=1000)
    parser.add_argument("--n-hf-statistics", type=int, default=50)
    parser.add_argument("--n-calibration", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--boundary-fraction", type=float, default=0.5)
    parser.add_argument("--e-rho-correlation", type=float, default=0.0)
    parser.add_argument(
        "--correlation-values",
        type=float,
        nargs="*",
        default=(-0.5, 0.0, 0.5),
        help="Additional E-density correlations for Fig. 13 dependence study.",
    )
    parser.add_argument("--correlation-samples", type=int, default=500)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--conformal-scale-floor", type=float, default=0.02)
    parser.add_argument("--sweep-points", type=int, default=801)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--max-saved-curves", type=int, default=200)
    args = parser.parse_args()
    for name in ("n_uq", "n_hf_statistics", "n_calibration", "n_test", "correlation_samples", "sweep_points"):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.sweep_points < 2:
        parser.error("--sweep-points must be at least 2")
    if not 0.0 <= args.boundary_fraction <= 1.0:
        parser.error("--boundary-fraction must lie in [0, 1]")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must lie in (0, 1)")
    return args


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - reference) / max(float(np.linalg.norm(reference)), 1.0e-30))


def _parameter_set(bounds, correlation: float) -> EllipsoidalConvexSet:
    return EllipsoidalConvexSet(
        center=np.asarray(bounds.center, dtype=float),
        scales=0.5 * (np.asarray(bounds.upper, dtype=float) - np.asarray(bounds.lower, dtype=float)),
        correlation=correlation_matrix(bounds.dimension, e_rho=float(correlation)),
    ).validated()


def _observation_members(result, frequencies: np.ndarray, parameters: np.ndarray, observation_index: int) -> np.ndarray:
    response = predict_parametric_observation_members(result, frequencies, parameters, observation_index)
    return np.abs(response) * 1.0e3


def _observation_hf(family, frequencies: np.ndarray, parameters: np.ndarray, observation_index: int) -> np.ndarray:
    response = family.solve_grid(frequencies, parameters)
    return np.abs(response[..., observation_index]) * 1.0e3


def _envelope(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.min(curves, axis=0)
    upper = np.max(curves, axis=0)
    return lower, upper, upper - lower


def _peak_frequency(frequencies: np.ndarray, curve: np.ndarray) -> float:
    return float(frequencies[int(np.argmax(curve))])


def _resonance_ranges(
    family,
    parameters: np.ndarray,
    frequencies: np.ndarray,
    curves: np.ndarray,
    *,
    half_window_hz: float = 2.0,
) -> list[dict]:
    if family.modal is None or family.modal.n_modes == 0:
        return []
    natural = family.modal.natural_frequencies_hz(parameters)
    frequency_step = float(np.median(np.diff(frequencies)))
    half_window = max(1, int(round(float(half_window_hz) / max(frequency_step, 1.0e-30))))
    rows = []
    for mode_index in range(natural.shape[1]):
        mode_freq = natural[:, mode_index]
        active = (mode_freq >= frequencies[0]) & (mode_freq <= frequencies[-1])
        if not np.any(active):
            continue
        active_freq = mode_freq[active]
        active_curves = curves[active]
        amplitudes = []
        for fr, curve in zip(active_freq, active_curves):
            index = int(np.argmin(np.abs(frequencies - fr)))
            lo = max(index - half_window, 0)
            hi = min(index + half_window + 1, frequencies.size)
            amplitudes.append(float(np.max(curve[lo:hi])))
        rows.append(
            {
                "mode": mode_index + 1,
                "frequency_lower_hz": float(np.min(active_freq)),
                "frequency_upper_hz": float(np.max(active_freq)),
                "amplitude_lower_mm": float(np.min(amplitudes)),
                "amplitude_upper_mm": float(np.max(amplitudes)),
            }
        )
    return rows


def _sample_convergence(
    frequencies: np.ndarray,
    curves: np.ndarray,
    candidate_counts: tuple[int, ...] = (25, 50, 100, 200, 500, 1000),
) -> list[dict]:
    full_lower, full_upper, full_width = _envelope(curves)
    rows = []
    for count in candidate_counts:
        n = min(int(count), curves.shape[0])
        if n < 2:
            continue
        lower, upper, width = _envelope(curves[:n])
        rows.append(
            {
                "sample_count": n,
                "lower_relative_l2_vs_full": _relative_l2(lower, full_lower),
                "upper_relative_l2_vs_full": _relative_l2(upper, full_upper),
                "width_relative_l2_vs_full": _relative_l2(width, full_width),
                "max_boundary_abs_difference_mm": float(
                    max(np.max(np.abs(lower - full_lower)), np.max(np.abs(upper - full_upper)))
                ),
            }
        )
        if n == curves.shape[0]:
            break
    if not rows or rows[-1]["sample_count"] != curves.shape[0]:
        rows.extend(_sample_convergence(frequencies, curves, (curves.shape[0],)))
    # remove accidental duplicates
    unique = {}
    for row in rows:
        unique[int(row["sample_count"])] = row
    return [unique[key] for key in sorted(unique)]


def _calibration_reliability(
    member_curves: np.ndarray,
    reference_curves: np.ndarray,
    test_members: np.ndarray,
    test_reference: np.ndarray,
    scale_floor: float,
) -> list[dict]:
    rows = []
    for confidence in (0.80, 0.90, 0.95):
        calibrator = CurveConformalCalibrator.fit(
            member_curves,
            reference_curves,
            confidence=confidence,
            relative_scale_floor=scale_floor,
        )
        _, _, lower, upper = calibrator.band(test_members)
        rows.append(
            {
                "nominal_curve_coverage": confidence,
                "calibration_empirical_curve_coverage": calibrator.empirical_curve_coverage,
                "test_empirical_curve_coverage": empirical_curve_coverage(test_reference, lower, upper),
                "conformal_multiplier": calibrator.multiplier,
                "mean_test_band_width_mm": float(np.mean(upper - lower)),
            }
        )
    return rows


def run_case(case: str, args: argparse.Namespace) -> None:
    source_dir = args.results_root / case
    output_dir = args.out / case
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, system, operators, family, result, observation_index = load_parametric_case(
        source_dir, device=args.device
    )
    bounds = family.bounds
    fmin, fmax = [float(value) for value in metadata["frequency_interval_hz"]]
    frequencies = np.linspace(fmin, fmax, int(args.sweep_points))

    ellipsoid = _parameter_set(bounds, args.e_rho_correlation)
    seeds = {
        "uq": args.seed + 11,
        "hf": args.seed + 22,
        "calibration": args.seed + 33,
        "test": args.seed + 44,
    }
    uq_mu = ellipsoid.sample(args.n_uq, seeds["uq"], boundary_fraction=args.boundary_fraction)
    hf_mu = ellipsoid.sample(args.n_hf_statistics, seeds["hf"], boundary_fraction=args.boundary_fraction)
    calibration_mu = ellipsoid.sample(args.n_calibration, seeds["calibration"], boundary_fraction=args.boundary_fraction)
    test_mu = ellipsoid.sample(args.n_test, seeds["test"], boundary_fraction=args.boundary_fraction)

    # Ellipsoidal online propagation.
    uq_members = _observation_members(result, frequencies, uq_mu, observation_index)
    uq_mean_curves = np.mean(uq_members, axis=0)
    surr_lower, surr_upper, surr_width = _envelope(uq_mean_curves)
    nominal_curve = _observation_hf(family, frequencies, bounds.center[None, :], observation_index)[0]

    # Independent HF statistics over a separate ellipsoidal sample.
    hf_curves = _observation_hf(family, frequencies, hf_mu, observation_index)
    hf_lower, hf_upper, hf_width = _envelope(hf_curves)
    same_sample_surr = np.mean(_observation_members(result, frequencies, hf_mu, observation_index), axis=0)
    same_lower, same_upper, same_width = _envelope(same_sample_surr)

    envelope_rows = []
    for i, frequency in enumerate(frequencies):
        envelope_rows.append(
            {
                "frequency_hz": float(frequency),
                "nominal_hf_mm": float(nominal_curve[i]),
                "surrogate_lower_mm": float(surr_lower[i]),
                "surrogate_upper_mm": float(surr_upper[i]),
                "surrogate_width_mm": float(surr_width[i]),
                "hf_lower_mm": float(hf_lower[i]),
                "hf_upper_mm": float(hf_upper[i]),
                "hf_width_mm": float(hf_width[i]),
                "same_sample_surrogate_lower_mm": float(same_lower[i]),
                "same_sample_surrogate_upper_mm": float(same_upper[i]),
            }
        )
    _write_csv(output_dir / "ellipsoidal_envelope.csv", envelope_rows)

    resonance_rows = _resonance_ranges(family, uq_mu, frequencies, uq_mean_curves)
    _write_csv(output_dir / "ellipsoidal_resonance_ranges.csv", resonance_rows)
    convergence_rows = _sample_convergence(frequencies, uq_mean_curves)
    _write_csv(output_dir / "ellipsoidal_sample_convergence.csv", convergence_rows)

    # Unseen ellipsoidal test curves.
    test_hf_response = family.solve_grid(frequencies, test_mu)
    test_member_response = predict_parametric_members(result, frequencies, test_mu)
    test_mean_response = np.mean(test_member_response, axis=0)
    test_members_obs = np.abs(test_member_response[..., observation_index]) * 1.0e3
    test_mean_obs = np.mean(test_members_obs, axis=0)
    test_hf_obs = np.abs(test_hf_response[..., observation_index]) * 1.0e3
    test_rows = []
    for j, mu in enumerate(test_mu):
        peak_hf = _peak_frequency(frequencies, test_hf_obs[j])
        peak_pred = _peak_frequency(frequencies, test_mean_obs[j])
        test_rows.append(
            {
                "parameter_index": j,
                **{name: float(value) for name, value in zip(bounds.names, mu)},
                "observation_curve_relative_l2": _relative_l2(test_mean_obs[j], test_hf_obs[j]),
                "full_response_relative_l2": _relative_l2(test_mean_response[j], test_hf_response[j]),
                "peak_frequency_error_hz": abs(peak_pred - peak_hf),
                "peak_amplitude_relative_error": abs(float(np.max(test_mean_obs[j]) - np.max(test_hf_obs[j])))
                / max(float(np.max(test_hf_obs[j])), 1.0e-30),
            }
        )
    _write_csv(output_dir / "ellipsoidal_unseen_test_metrics.csv", test_rows)

    # Ellipsoidal curve-wise calibration on independent realizations.
    calibration_members = _observation_members(result, frequencies, calibration_mu, observation_index)
    calibration_hf = _observation_hf(family, frequencies, calibration_mu, observation_index)
    calibrator = CurveConformalCalibrator.fit(
        calibration_members,
        calibration_hf,
        confidence=args.confidence,
        relative_scale_floor=args.conformal_scale_floor,
    )
    _, _, test_lower, test_upper = calibrator.band(test_members_obs)
    test_coverage = empirical_curve_coverage(test_hf_obs, test_lower, test_upper)
    reliability_rows = _calibration_reliability(
        calibration_members,
        calibration_hf,
        test_members_obs,
        test_hf_obs,
        args.conformal_scale_floor,
    )
    _write_csv(output_dir / "ellipsoidal_calibration_reliability.csv", reliability_rows)

    errors = np.asarray([row["observation_curve_relative_l2"] for row in test_rows], dtype=float)
    representative = int(np.argsort(errors)[len(errors) // 2])
    rep_mean, rep_spread, rep_lower, rep_upper = calibrator.band(test_members_obs[:, representative, :])
    calibration_rows = []
    for i, frequency in enumerate(frequencies):
        calibration_rows.append(
            {
                "frequency_hz": float(frequency),
                "hf_mm": float(test_hf_obs[representative, i]),
                "ensemble_mean_mm": float(rep_mean[i]),
                "ensemble_spread_mm": float(rep_spread[i]),
                "calibrated_lower_mm": float(rep_lower[i]),
                "calibrated_upper_mm": float(rep_upper[i]),
            }
        )
    _write_csv(output_dir / "ellipsoidal_calibration_example.csv", calibration_rows)

    # E-density dependence study.
    correlation_rows = []
    for corr_index, correlation_value in enumerate(args.correlation_values):
        study_set = _parameter_set(bounds, correlation_value)
        study_mu = study_set.sample(
            args.correlation_samples,
            args.seed + 1000 + corr_index,
            boundary_fraction=args.boundary_fraction,
        )
        study_members = _observation_members(result, frequencies, study_mu, observation_index)
        study_curves = np.mean(study_members, axis=0)
        lo, up, width = _envelope(study_curves)
        for i, frequency in enumerate(frequencies):
            correlation_rows.append(
                {
                    "e_rho_correlation": float(correlation_value),
                    "frequency_hz": float(frequency),
                    "lower_mm": float(lo[i]),
                    "upper_mm": float(up[i]),
                    "width_mm": float(width[i]),
                }
            )
    _write_csv(output_dir / "ellipsoidal_correlation_envelopes.csv", correlation_rows)

    save_count = min(int(args.max_saved_curves), uq_mean_curves.shape[0])
    np.savez_compressed(
        output_dir / "ellipsoidal_uq_curves.npz",
        frequency_hz=frequencies,
        parameters=uq_mu[:save_count],
        surrogate_curves_mm=uq_mean_curves[:save_count],
        hf_parameters=hf_mu,
        hf_curves_mm=hf_curves,
        test_parameters=test_mu,
        test_hf_curves_mm=test_hf_obs,
        test_surrogate_curves_mm=test_mean_obs,
    )

    summary = {
        "case": case,
        "uncertainty_model": "non-probabilistic ellipsoidal convex set",
        "parameter_names": list(bounds.names),
        "center": ellipsoid.center.tolist(),
        "scales": ellipsoid.scales.tolist(),
        "R_mu": np.asarray(ellipsoid.correlation).tolist(),
        "e_rho_correlation": float(args.e_rho_correlation),
        "boundary_fraction": float(args.boundary_fraction),
        "uq_sample_count": int(args.n_uq),
        "hf_statistics_sample_count": int(args.n_hf_statistics),
        "calibration_sample_count": int(args.n_calibration),
        "test_sample_count": int(args.n_test),
        "lower_boundary_relative_l2_same_sample": _relative_l2(same_lower, hf_lower),
        "upper_boundary_relative_l2_same_sample": _relative_l2(same_upper, hf_upper),
        "width_relative_l2_same_sample": _relative_l2(same_width, hf_width),
        "median_unseen_observation_curve_error": float(np.median(errors)),
        "p95_unseen_observation_curve_error": float(np.quantile(errors, 0.95)),
        "median_unseen_full_response_error": float(
            np.median([row["full_response_relative_l2"] for row in test_rows])
        ),
        "target_curve_coverage": float(args.confidence),
        "test_curve_coverage": float(test_coverage),
        "conformal_multiplier": float(calibrator.multiplier),
        "sample_estimated_bounds": True,
        "certified_global_bounds": False,
    }
    with (output_dir / "ellipsoidal_uq_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Compact diagnostic figure; manuscript composite figures are generated separately.
    apply_journal_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    for curve in uq_mean_curves[: min(80, uq_mean_curves.shape[0])]:
        ax.plot(frequencies, curve, color="#808080", lw=0.45, alpha=0.22)
    ax.plot(frequencies, hf_lower, color="#000000", lw=1.0, linestyle="--", label="HF sample bounds")
    ax.plot(frequencies, hf_upper, color="#000000", lw=1.0, linestyle="--")
    ax.plot(frequencies, surr_lower, color="#C00000", lw=1.2, label="Surrogate ellipsoidal bounds")
    ax.plot(frequencies, surr_upper, color="#C00000", lw=1.2)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"$|w|$ (mm)")
    ax.legend(loc="best")
    style_axis(ax)
    save_figure(fig, output_dir / "ellipsoidal_envelope_diagnostic.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cases = CASES if args.case == "all" else (args.case,)
    for case in cases:
        run_case(case, args)
    print(f"Wrote ellipsoidal UQ studies to {args.out}")


if __name__ == "__main__":
    main()

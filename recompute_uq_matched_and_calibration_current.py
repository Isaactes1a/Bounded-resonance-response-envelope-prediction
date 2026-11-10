# -*- coding: utf-8 -*-
"""Recompute matched-sample envelopes and curve-wise calibration with the
retained current checkpoints.

The historical ``ellipsoidal_uq`` outputs were generated from a result root
that is no longer present.  This script reuses the stored direct RBF-FD
reference curves (matched 8-realization sets and the 40+40 calibration/test
archive) and evaluates the current production checkpoints on those parameter
realizations, so the envelope and calibration tables describe the same models
as the classical baselines and unseen-accuracy tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rbf_fd_full_order import predict_parametric_observation_members
from rbf_fd_full_order.uncertainty import (
    CurveConformalCalibrator,
    empirical_curve_coverage,
)
from rbf_fd_full_order.visualization.result_support import load_parametric_case


CASES = ("flat", "stiffened")


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - reference) / max(float(np.linalg.norm(reference)), 1.0e-30))


def _envelope(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.min(curves, axis=0)
    upper = np.max(curves, axis=0)
    return lower, upper, upper - lower


def _members(result, frequencies: np.ndarray, parameters: np.ndarray, observation_index: int) -> np.ndarray:
    response = predict_parametric_observation_members(result, frequencies, parameters, observation_index)
    return np.abs(response) * 1.0e3


def run_case(case: str, results_root: Path, out: Path, device: str) -> None:
    case_dir = results_root / f"parametric_many_query_{case}_5000_best" / case
    _, _, _, _, result, observation_index = load_parametric_case(case_dir, device=device)

    archive = np.load(results_root / "ellipsoidal_uq" / case / "ellipsoidal_calibration_40x40.npz")
    frequencies = np.asarray(archive["frequencies_hz"], dtype=float)
    calibration_parameters = np.asarray(archive["calibration_parameters"], dtype=float)
    test_parameters = np.asarray(archive["test_parameters"], dtype=float)
    calibration_hf = np.asarray(archive["calibration_hf_mm"], dtype=float)
    test_hf = np.asarray(archive["test_hf_mm"], dtype=float)

    curves = np.load(results_root / "ellipsoidal_uq" / case / "ellipsoidal_uq_curves.npz")
    matched_parameters = np.asarray(curves["hf_parameters"], dtype=float)
    matched_hf = np.asarray(curves["hf_curves_mm"], dtype=float)

    matched_members = _members(result, frequencies, matched_parameters, observation_index)
    matched_pred = np.mean(matched_members, axis=0)
    matched_lower, matched_upper, matched_width = _envelope(matched_pred)
    hf_lower, hf_upper, hf_width = _envelope(matched_hf)
    matched = {
        "matched_curve_count": int(matched_parameters.shape[0]),
        "lower_relative_l2_percent": 100.0 * _relative_l2(matched_lower, hf_lower),
        "upper_relative_l2_percent": 100.0 * _relative_l2(matched_upper, hf_upper),
        "width_relative_l2_percent": 100.0 * _relative_l2(matched_width, hf_width),
        "max_absolute_boundary_difference_mm": float(np.max(np.abs(matched_lower - hf_lower))),
    }

    calibration_members = _members(result, frequencies, calibration_parameters, observation_index)
    test_members = _members(result, frequencies, test_parameters, observation_index)
    reliability = []
    for confidence in (0.80, 0.90, 0.95):
        calibrator = CurveConformalCalibrator.fit(
            calibration_members,
            calibration_hf,
            confidence=confidence,
            relative_scale_floor=0.02,
        )
        _, _, lower, upper = calibrator.band(test_members)
        coverage = empirical_curve_coverage(test_hf, lower, upper)
        reliability.append(
            {
                "nominal_coverage": confidence,
                "conformal_multiplier": calibrator.multiplier,
                "covered_test_curves": int(round(coverage * test_parameters.shape[0])),
                "test_curve_count": int(test_parameters.shape[0]),
                "mean_test_band_width_mm": float(np.mean(upper - lower)),
            }
        )

    summary = {"case": case, "matched": matched, "reliability": reliability}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{case}_uq_matched_and_calibration.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results") / "unseen_accuracy_current")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cases = CASES if args.case == "all" else (args.case,)
    for case in cases:
        run_case(case, args.results_root, args.out, args.device)

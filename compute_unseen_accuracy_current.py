# -*- coding: utf-8 -*-
"""Recompute the 40-curve unseen-accuracy metrics with the retained checkpoints.

The earlier ``ellipsoidal_uq`` CSV metrics were generated from a historical
result root that no longer exists.  This script evaluates the current
production checkpoints (5000 epochs x 5 members) on the same 40 independent
test realizations stored in ``ellipsoidal_calibration_40x40.npz`` so that the
unseen-accuracy table and the classical-baseline table describe the same
models and the same curves.  Only observation-point metrics are recomputed
here; full-domain consistency is reported through the full-domain relative
MSE of the controlled ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rbf_fd_full_order import predict_parametric_observation_members
from rbf_fd_full_order.visualization.result_support import load_parametric_case


CASES = ("flat", "stiffened")


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - reference) / max(float(np.linalg.norm(reference)), 1.0e-30))


def run_case(case: str, results_root: Path, out: Path, device: str) -> None:
    case_dir = results_root / f"parametric_many_query_{case}_5000_best" / case
    _, _, _, _, result, observation_index = load_parametric_case(case_dir, device=device)
    archive = np.load(results_root / "ellipsoidal_uq" / case / "ellipsoidal_calibration_40x40.npz")
    frequencies = np.asarray(archive["frequencies_hz"], dtype=float)
    parameters = np.asarray(archive["test_parameters"], dtype=float)
    reference_obs = np.asarray(archive["test_hf_mm"], dtype=float)

    members = predict_parametric_observation_members(result, frequencies, parameters, observation_index)
    predicted_obs = np.mean(np.abs(members), axis=0) * 1.0e3

    rows = []
    obs_errors = []
    peak_errors = []
    amplitude_errors = []
    for index in range(parameters.shape[0]):
        obs_error = _relative_l2(predicted_obs[index], reference_obs[index])
        peak_ref = int(np.argmax(reference_obs[index]))
        peak_pred = int(np.argmax(predicted_obs[index]))
        peak_error = abs(float(frequencies[peak_pred] - frequencies[peak_ref]))
        amplitude_error = abs(float(np.max(predicted_obs[index]) - np.max(reference_obs[index])))
        amplitude_error /= max(float(np.max(reference_obs[index])), 1.0e-30)
        obs_errors.append(obs_error)
        peak_errors.append(peak_error)
        amplitude_errors.append(amplitude_error)
        rows.append(
            {
                "parameter_index": index,
                "observation_curve_relative_l2": obs_error,
                "peak_frequency_error_hz": peak_error,
                "peak_amplitude_relative_error": amplitude_error,
            }
        )

    summary = {
        "case": case,
        "test_curve_count": int(parameters.shape[0]),
        "frequency_point_count": int(frequencies.size),
        "observation_frf_median_percent": 100.0 * float(np.median(obs_errors)),
        "observation_frf_p95_percent": 100.0 * float(np.quantile(obs_errors, 0.95)),
        "observation_frf_max_percent": 100.0 * float(np.max(obs_errors)),
        "peak_frequency_error_max_hz": float(np.max(peak_errors)),
        "peak_amplitude_error_max_percent": 100.0 * float(np.max(amplitude_errors)),
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{case}_unseen_accuracy_current.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (out / f"{case}_unseen_accuracy_current.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
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

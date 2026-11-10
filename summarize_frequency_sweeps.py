"""Summarize full-field and observation-point errors from saved frequency sweeps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


CASES = ("flat", "hollow", "stiffened")
METHODS = (
    "full_data_only",
    "full_data_only_refined",
    "full_physics",
    "full_physics_refined",
    "pod_data_only",
    "pod_data_only_refined",
    "pod_physics",
    "pod_physics_refined",
)


def _response(directory: Path, stem: str) -> np.ndarray:
    return np.load(directory / f"frequency_sweep_{stem}_real.npy") + 1.0j * np.load(
        directory / f"frequency_sweep_{stem}_imag.npy"
    )


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results") / "unified_square_cases_modal_sweep_cuda",
        help="Root directory produced by run_frequency_validation.py.",
    )
    args = parser.parse_args()
    all_rows: list[dict[str, object]] = []
    for case in CASES:
        directory = args.root / case
        frequencies = np.load(directory / "frequency_sweep_hz.npy")
        reference = _response(directory, "reference")
        observation = int(np.argmax(np.max(np.abs(reference), axis=0)))
        peak_index = int(np.argmax(np.abs(reference[:, observation])))
        denominator = np.maximum(np.linalg.norm(reference, axis=1), 1.0e-30)
        reference_observation = np.abs(reference[:, observation])
        rows: list[dict[str, object]] = []
        for method in METHODS:
            prediction = _response(directory, method)
            per_frequency = np.linalg.norm(prediction - reference, axis=1) / denominator
            amplitude_error = np.abs(np.abs(prediction[:, observation]) - reference_observation) / np.maximum(
                reference_observation, 1.0e-30
            )
            rows.append(
                {
                    "case": case,
                    "method": method,
                    "sweep_points": int(frequencies.size),
                    "observation_index": observation,
                    "reference_peak_frequency_hz": float(frequencies[peak_index]),
                    "reference_peak_abs_mm": float(reference_observation[peak_index] * 1.0e3),
                    "global_relative_field_error": float(
                        np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), 1.0e-30)
                    ),
                    "mean_relative_field_error": float(np.mean(per_frequency)),
                    "median_relative_field_error": float(np.median(per_frequency)),
                    "max_relative_field_error": float(np.max(per_frequency)),
                    "relative_field_error_at_reference_peak": float(per_frequency[peak_index]),
                    "mean_observation_amplitude_relative_error": float(np.mean(amplitude_error)),
                    "observation_amplitude_relative_error_at_reference_peak": float(amplitude_error[peak_index]),
                }
            )
        _write(directory / "frequency_sweep_summary.csv", rows)
        all_rows.extend(rows)
    _write(args.root / "frequency_sweep_summary_all_cases.csv", all_rows)
    print(f"Wrote {len(all_rows)} sweep-summary rows to {args.root}")


if __name__ == "__main__":
    main()

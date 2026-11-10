# -*- coding: utf-8 -*-
"""Postprocess stiffened-plate modes 1--4 without retraining the surrogate.

The retained network was trained on 0--200 Hz with one rational mode.  This
script keeps that checkpoint unchanged, builds a four-mode eigenfrequency
tracker from the same fixed RBF-FD operators, and queries the existing network
over an extended frequency grid.  A small independent direct RBF-FD check is
also written for the out-of-band modes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from generate_ellipsoidal_uq_study import _parameter_set, _resonance_ranges
from rbf_fd_full_order import ModalResonanceConfig, predict_parametric_observation_members
from rbf_fd_full_order.core.modal_resonance import build_modal_resonance_interface
from rbf_fd_full_order.core.parametric_family import ParametricFrequencyFamily, ParametricModalRationalInterface
from rbf_fd_full_order.visualization.result_support import load_parametric_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode-count", type=int, default=4)
    parser.add_argument("--frequency-max-hz", type=float, default=450.0)
    parser.add_argument("--frequency-step-hz", type=float, default=1.0)
    parser.add_argument("--n-uq", type=int, default=1000)
    parser.add_argument("--n-validation", type=int, default=8)
    parser.add_argument("--boundary-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=701)
    args = parser.parse_args()
    if args.mode_count < 1 or args.n_uq < 1 or args.n_validation < 1:
        parser.error("mode and sample counts must be positive")
    if args.frequency_max_hz <= 200.0 or args.frequency_step_hz <= 0.0:
        parser.error("the extended frequency maximum must exceed 200 Hz and the step must be positive")
    return args


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metadata, _, operators, trained_family, result, observation_index = load_parametric_case(
        args.source, device=args.device
    )
    if str(metadata["case"]) != "stiffened":
        raise ValueError("This postprocessor is intentionally restricted to the stiffened case.")

    fmin, trained_fmax = [float(value) for value in metadata["frequency_interval_hz"]]
    config = ModalResonanceConfig(
        enabled=True,
        n_modes=int(args.mode_count),
        frequency_min_hz=fmin,
        frequency_max_hz=float(args.frequency_max_hz),
    )
    reference = build_modal_resonance_interface(
        operators,
        config,
        loss_factor=float(trained_family.bounds.center[3]),
    )
    extended_family = ParametricFrequencyFamily(
        nominal=operators,
        bounds=trained_family.bounds,
        modal=ParametricModalRationalInterface(reference=reference, bounds=trained_family.bounds),
    )
    if extended_family.modal is None or extended_family.modal.n_modes < args.mode_count:
        raise RuntimeError("The requested number of modes was not found below the extended frequency maximum.")

    frequencies = np.arange(
        fmin,
        float(args.frequency_max_hz) + 0.5 * float(args.frequency_step_hz),
        float(args.frequency_step_hz),
    )
    ellipsoid = _parameter_set(trained_family.bounds, 0.0)
    uq_mu = ellipsoid.sample(args.n_uq, args.seed + 11, boundary_fraction=args.boundary_fraction)
    members = predict_parametric_observation_members(result, frequencies, uq_mu, observation_index)
    mean_curves_mm = np.mean(np.abs(members) * 1.0e3, axis=0)
    ranges = _resonance_ranges(extended_family, uq_mu, frequencies, mean_curves_mm)
    for row in ranges:
        row["within_training_band"] = bool(row["frequency_upper_hz"] <= trained_fmax)
    write_csv(args.out / "ellipsoidal_resonance_ranges.csv", ranges)

    validation_mu = ellipsoid.sample(
        args.n_validation,
        args.seed + 44,
        boundary_fraction=args.boundary_fraction,
    )
    natural = extended_family.modal.natural_frequencies_hz(validation_mu)
    validation_rows = []
    half_window = max(2, int(round(2.0 / args.frequency_step_hz)))
    for sample_index, mu in enumerate(validation_mu):
        for mode_index in range(args.mode_count):
            frequency = float(natural[sample_index, mode_index])
            if frequency <= trained_fmax:
                continue
            center = round(frequency / args.frequency_step_hz) * args.frequency_step_hz
            local_frequency = center + args.frequency_step_hz * np.arange(-half_window, half_window + 1)
            hf_mm = np.asarray(
                [abs(trained_family.solve(float(f), mu)[observation_index]) * 1.0e3 for f in local_frequency],
                dtype=float,
            )
            predicted = predict_parametric_observation_members(
                result, local_frequency, mu, observation_index
            )
            predicted_mm = np.mean(np.abs(predicted) * 1.0e3, axis=0)[0]
            hf_peak = float(np.max(hf_mm))
            predicted_peak = float(np.max(predicted_mm))
            validation_rows.append(
                {
                    "sample": sample_index + 1,
                    "mode": mode_index + 1,
                    "natural_frequency_hz": frequency,
                    "hf_local_peak_mm": hf_peak,
                    "surrogate_local_peak_mm": predicted_peak,
                    "peak_relative_error": abs(predicted_peak - hf_peak) / max(hf_peak, 1.0e-30),
                }
            )
    write_csv(args.out / "extended_mode_validation.csv", validation_rows)

    summary = {
        "trained_frequency_interval_hz": [fmin, trained_fmax],
        "postprocessed_frequency_interval_hz": [fmin, float(args.frequency_max_hz)],
        "trained_rational_modes": int(metadata["modal_modes"]),
        "postprocessed_eigenmodes": int(args.mode_count),
        "network_retrained": False,
        "uq_samples": int(args.n_uq),
        "validation_samples": int(args.n_validation),
        "mode_validation": {},
    }
    for mode in sorted({int(row["mode"]) for row in validation_rows}):
        errors = np.asarray(
            [float(row["peak_relative_error"]) for row in validation_rows if int(row["mode"]) == mode]
        )
        summary["mode_validation"][str(mode)] = {
            "median_peak_relative_error": float(np.median(errors)),
            "maximum_peak_relative_error": float(np.max(errors)),
        }
    with (args.out / "extended_mode_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote extended stiffened resonance study to {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

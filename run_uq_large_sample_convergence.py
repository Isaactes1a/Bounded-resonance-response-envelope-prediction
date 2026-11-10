# -*- coding: utf-8 -*-
"""Larger surrogate UQ sample convergence (5000/10000 realizations).

One reproducible 10000-realization ellipsoidal draw is generated per case and
nested prefixes (1000/2000/5000/10000) are propagated through the retained
production surrogate.  Envelope stability is reported as relative L2
differences against the 10000-realization reference, together with the
first-resonance upper envelope value and the maximum envelope width.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rbf_fd_full_order import ParameterBounds, predict_parametric_observation_members
from rbf_fd_full_order.uncertainty import EllipsoidalConvexSet, correlation_matrix
from rbf_fd_full_order.visualization.result_support import load_parametric_case


CASES = ("flat", "stiffened")


def _ellipsoid(bounds: ParameterBounds) -> EllipsoidalConvexSet:
    return EllipsoidalConvexSet(
        center=np.asarray(bounds.center, dtype=float),
        scales=0.5 * (np.asarray(bounds.upper, dtype=float) - np.asarray(bounds.lower, dtype=float)),
        correlation=correlation_matrix(bounds.dimension, e_rho=0.0),
    ).validated()


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - reference) / max(float(np.linalg.norm(reference)), 1.0e-30))


def run_case(case: str, results_root: Path, out: Path, device: str) -> None:
    case_dir = results_root / f"parametric_many_query_{case}_5000_best" / case
    metadata, _, _, _, result, observation_index = load_parametric_case(case_dir, device=device)
    bounds = result.family.bounds
    ellipsoid = _ellipsoid(bounds)
    # Follow the band of the model instead of assuming 0-200 Hz.
    fmin, fmax = (float(value) for value in metadata["frequency_interval_hz"])
    points = int(round(fmax - fmin)) + 1
    frequencies = np.linspace(fmin, fmax, points)
    parameters = ellipsoid.sample(10000, 712, boundary_fraction=0.5)

    print(f"[{case}] propagating 10000 realizations", flush=True)
    members = predict_parametric_observation_members(result, frequencies, parameters, observation_index)
    curves = np.mean(np.abs(members), axis=0) * 1.0e3

    counts = (1000, 2000, 5000, 10000)
    f1_nominal = float(result.family.modal.reference.natural_frequencies_hz[0])
    f1_index = int(np.argmin(np.abs(frequencies - f1_nominal)))
    reference_upper = np.max(curves, axis=0)
    reference_lower = np.min(curves, axis=0)
    reference_width = reference_upper - reference_lower
    rows = []
    for count in counts:
        lower = np.min(curves[:count], axis=0)
        upper = np.max(curves[:count], axis=0)
        rows.append(
            {
                "sample_count": int(count),
                "lower_relative_l2_vs_10000": _relative_l2(lower, reference_lower),
                "upper_relative_l2_vs_10000": _relative_l2(upper, reference_upper),
                "width_relative_l2_vs_10000": _relative_l2(upper - lower, reference_width),
                "upper_envelope_at_f1_mm": float(upper[f1_index]),
                "max_width_mm": float(np.max(upper - lower)),
            }
        )
        print(json.dumps({"case": case, **rows[-1]}, ensure_ascii=False), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{case}_uq_large_convergence.json").write_text(
        json.dumps({"case": case, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez(
        out / f"{case}_uq_large_convergence_curves.npz",
        frequencies=frequencies,
        parameters=parameters,
        curves=curves,
    )
    print(f"[{case}] saved curves archive {out / f'{case}_uq_large_convergence_curves.npz'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results") / "unseen_accuracy_current")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = CASES if args.case == "all" else (args.case,)
    for case in cases:
        run_case(case, args.results_root, args.out, args.device)


if __name__ == "__main__":
    main()

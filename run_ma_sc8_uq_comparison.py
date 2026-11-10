# -*- coding: utf-8 -*-
"""ma_sc8 classical baseline in the complete ellipsoidal UQ task.

For 1000 ellipsoidal realizations, the mode-acceleration baseline with static
correction (ma_sc8, parameter-dependent pole scaling) propagates observation
FRFs and produces envelopes and resonance ranges.  The same quantities are
computed for the retained production surrogate (1000-curve archive).  The
output table reports the trade-off in envelope accuracy, resonance ranges,
peak-amplitude ranges and propagation wall time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
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


def _parabolic_peak(frequencies: np.ndarray, curve: np.ndarray) -> float:
    index = int(np.argmax(curve))
    if index <= 0 or index >= curve.size - 1:
        return float(frequencies[index])
    y = np.log10(np.maximum(curve[index - 1 : index + 2], 1.0e-30))
    denominator = y[0] - 2.0 * y[1] + y[2]
    if abs(denominator) < 1.0e-14:
        return float(frequencies[index])
    step = float(np.median(np.diff(frequencies)))
    return float(frequencies[index] + 0.5 * (y[0] - y[2]) / denominator * step)


def _ma_sc8_observation(
    interface,
    frequencies: np.ndarray,
    parameters: np.ndarray,
    observation_index: int,
    static_observation: float,
) -> np.ndarray:
    reference = interface.reference
    young = parameters[:, 0]
    density = parameters[:, 1]
    pressure = parameters[:, 2]
    eta = parameters[:, 3]
    omega_sq = (2.0 * np.pi * frequencies) ** 2
    eigenvalues = reference.eigenvalues
    modal_forces = reference.modal_forces
    phi_obs = reference.mode_shapes[observation_index, :]
    denominator = (
        (1.0 + 1.0j * eta)[:, None, None] * young[:, None, None] * eigenvalues[None, None, :]
        - density[:, None, None] * omega_sq[None, :, None]
    )
    coefficients = pressure[:, None, None] * modal_forces[None, None, :] / denominator
    modal = np.asarray(coefficients @ phi_obs, dtype=complex)
    static_modal = (
        pressure[:, None]
        * modal_forces
        / ((1.0 + 1.0j * eta)[:, None] * young[:, None] * eigenvalues[None, :])
    ) @ phi_obs
    static = (pressure / ((1.0 + 1.0j * eta) * young)) * static_observation
    return np.abs(static[:, None] - static_modal[:, None] + modal) * 1.0e3


def _ranges(frequencies: np.ndarray, curves: np.ndarray) -> dict[str, float]:
    peaks = np.asarray([_parabolic_peak(frequencies, curves[i]) for i in range(curves.shape[0])])
    amplitudes = np.asarray([float(np.max(curves[i])) for i in range(curves.shape[0])])
    return {
        "resonance_frequency_min_hz": float(np.min(peaks)),
        "resonance_frequency_max_hz": float(np.max(peaks)),
        "peak_amplitude_min_mm": float(np.min(amplitudes)),
        "peak_amplitude_max_mm": float(np.max(amplitudes)),
    }


def run_case(case: str, results_root: Path, out: Path, device: str) -> None:
    case_dir = results_root / f"parametric_many_query_{case}_5000_best" / case
    _, _, _, _, result, observation_index = load_parametric_case(case_dir, device=device)
    bounds = result.family.bounds
    ellipsoid = _ellipsoid(bounds)
    frequencies = np.linspace(0.0, 200.0, 201)
    parameters = ellipsoid.sample(1000, 712, boundary_fraction=0.5)

    # One static solve at A(0) = K0; eta/young scaling factors out.
    k0 = result.family.nominal.K.tocsc()
    static_full = spla.spsolve(k0, np.asarray(result.family.nominal.rhs, dtype=complex))
    static_observation = float(np.abs(static_full[observation_index]))

    started = time.perf_counter()
    ma_curves = _ma_sc8_observation(
        result.family.modal, frequencies, parameters, observation_index, static_observation
    )
    ma_time = time.perf_counter() - started
    ma_lower = np.min(ma_curves, axis=0)
    ma_upper = np.max(ma_curves, axis=0)
    ma_width = ma_upper - ma_lower
    ma_ranges = _ranges(frequencies, ma_curves)

    archive = np.load(results_root / "unseen_accuracy_current" / f"{case}_current_uq1000.npz")
    surr_curves = np.asarray(archive["curves"], dtype=float)
    started = time.perf_counter()
    _ = predict_parametric_observation_members(result, frequencies, parameters[:50], observation_index)
    surr_propagation_s = (time.perf_counter() - started) * (1000.0 / 50.0)
    surr_lower = np.min(surr_curves, axis=0)
    surr_upper = np.max(surr_curves, axis=0)
    surr_width = surr_upper - surr_lower
    surr_ranges = _ranges(frequencies, surr_curves)

    summary = {
        "case": case,
        "realization_count": int(parameters.shape[0]),
        "ma_sc8": {
            "propagation_time_s": float(ma_time),
            "envelope_lower_upper_width": {
                "upper_at_f1_mm": float(ma_upper[int(np.argmin(np.abs(frequencies - float(result.family.modal.reference.natural_frequencies_hz[0]))))]),
                "max_width_mm": float(np.max(ma_width)),
            },
            **_ranges(frequencies, ma_curves),
        },
        "surrogate": {
            "propagation_time_s_1000": float(surr_propagation_s),
            "envelope_lower_upper_width": {
                "upper_at_f1_mm": float(surr_upper[int(np.argmin(np.abs(frequencies - float(result.family.modal.reference.natural_frequencies_hz[0]))))]),
                "max_width_mm": float(np.max(surr_width)),
            },
            **_ranges(frequencies, surr_curves),
        },
        "cross_method": {
            "lower_relative_l2": _relative_l2(ma_lower, surr_lower),
            "upper_relative_l2": _relative_l2(ma_upper, surr_upper),
            "width_relative_l2": _relative_l2(ma_width, surr_width),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{case}_ma_sc8_uq_comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


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

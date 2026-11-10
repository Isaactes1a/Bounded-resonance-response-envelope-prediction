# -*- coding: utf-8 -*-
"""Precompute and copy the data files consumed by figure_scripts/.

Every per-figure script under ``figure_scripts/`` reads only from
``figure_scripts/data/``; this preparation step materializes the model-derived
arrays (surrogate curves, calibration bands, spot checks, geometry) that are
otherwise produced on the fly by the compute scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import scipy.sparse.linalg as spla
import torch

from rbf_fd_full_order import (
    ParameterBounds,
    predict_parametric_members,
    predict_parametric_observation_members,
)
from rbf_fd_full_order.uncertainty import (
    CurveConformalCalibrator,
    EllipsoidalConvexSet,
    correlation_matrix,
    empirical_curve_coverage,
)
from rbf_fd_full_order.visualization.result_support import load_parametric_case


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"
DATA = ROOT / "figure_scripts" / "data"
CASES = ("flat", "stiffened")


def _ellipsoid(bounds: ParameterBounds, e_rho: float = 0.0) -> EllipsoidalConvexSet:
    return EllipsoidalConvexSet(
        center=np.asarray(bounds.center, dtype=float),
        scales=0.5 * (np.asarray(bounds.upper, dtype=float) - np.asarray(bounds.lower, dtype=float)),
        correlation=correlation_matrix(bounds.dimension, e_rho=float(e_rho)),
    ).validated()


def _obs(result, frequencies: np.ndarray, parameters: np.ndarray, observation_index: int) -> np.ndarray:
    members = predict_parametric_observation_members(result, frequencies, parameters, observation_index)
    return np.mean(np.abs(members), axis=0) * 1.0e3


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - reference) / max(float(np.linalg.norm(reference)), 1.0e-30))


def _wilson(covered: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = covered / total
    denom = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return float((centre - margin) / denom), float((centre + margin) / denom)


def prepare_geometry() -> None:
    fractions = np.asarray([0.2, 0.4, 0.6, 0.8])
    axis = np.linspace(0.0, 1.0, 31)
    grid_x, grid_y = np.meshgrid(axis, axis)
    nodes = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    np.savez_compressed(
        DATA / "geometry_plate_nodes.npz",
        flat_nodes=nodes,
        stiffened_nodes=nodes,
        stiffener_fractions=fractions,
    )


def prepare_case(case: str, device: str) -> None:
    case_dir = RESULTS / f"parametric_many_query_{case}_5000_best" / case
    metadata, _, _, _, result, observation_index = load_parametric_case(
        case_dir, device=device)
    bounds = result.family.bounds
    # The analysis band is a property of the model, so it is read from the case
    # metadata rather than assumed. The grid keeps the 1 Hz spacing that the
    # earlier 201-point/200 Hz convention used, so the two cases stay
    # comparable point for point while spanning their own bands.
    fmin, fmax = (float(value) for value in metadata["frequency_interval_hz"])
    points = int(round(fmax - fmin)) + 1
    frequencies = np.linspace(fmin, fmax, points)
    print(f"[{case}] band {fmin:.0f}-{fmax:.0f} Hz, {points} points", flush=True)

    archive = np.load(RESULTS / "ellipsoidal_uq" / case / "ellipsoidal_calibration_40x40.npz")
    test_parameters = np.asarray(archive["test_parameters"], dtype=float)
    test_hf = np.asarray(archive["test_hf_mm"], dtype=float)
    calibration_parameters = np.asarray(archive["calibration_parameters"], dtype=float)
    calibration_hf = np.asarray(archive["calibration_hf_mm"], dtype=float)

    # Fig08 / Fig09: 40-curve test data + eight-curve full-response spot check.
    predicted_obs = _obs(result, frequencies, test_parameters, observation_index)
    errors = np.asarray(
        [_relative_l2(predicted_obs[i], test_hf[i]) for i in range(test_parameters.shape[0])]
    )
    np.savez_compressed(
        DATA / f"{case}_test_curves.npz",
        frequencies=frequencies,
        hf_mm=test_hf,
        predicted_mm=predicted_obs,
        errors=errors,
    )
    spot = np.load(RESULTS / "phase7" / f"{case}_test_reference.npz")
    spot_parameters = np.asarray(spot["test_parameters"], dtype=float)
    spot_reference = np.asarray(spot["test_reference_full"], dtype=complex)
    predicted_full = np.mean(predict_parametric_members(result, frequencies, spot_parameters), axis=0)
    spot_errors = [
        _relative_l2(predicted_full[i], spot_reference[i]) for i in range(spot_parameters.shape[0])
    ]
    (DATA / f"{case}_full_response_spotcheck.json").write_text(
        json.dumps(
            {"case": case, "errors": spot_errors, "curve_count": len(spot_errors)},
            indent=2,
        ),
        encoding="utf-8",
    )

    # Fig11/Fig12/Fig13: 1000-curve UQ archive.
    uq_data = np.load(RESULTS / "unseen_accuracy_current" / f"{case}_current_uq1000.npz")
    uq_parameters = np.asarray(uq_data["parameters"], dtype=float)
    uq_curves = np.asarray(uq_data["curves"], dtype=float)
    np.savez_compressed(
        DATA / f"{case}_uq1000.npz",
        frequencies=frequencies,
        parameters=uq_parameters,
        curves=uq_curves,
    )

    # Fig14: calibration plot data.
    calibration_members = np.abs(
        predict_parametric_observation_members(
            result, frequencies, calibration_parameters, observation_index
        )
    ) * 1.0e3
    test_members = np.abs(
        predict_parametric_observation_members(result, frequencies, test_parameters, observation_index)
    ) * 1.0e3
    test_mean = np.mean(test_members, axis=0)
    representative = int(np.argsort(errors)[len(errors) // 2])
    fractions = []
    counts = []
    multipliers = []
    mean_widths = []
    for confidence in (0.80, 0.90, 0.95):
        fit = CurveConformalCalibrator.fit(
            calibration_members, calibration_hf, confidence=confidence, relative_scale_floor=0.02
        )
        _, _, lower, upper = fit.band(test_members)
        inside = (test_hf >= lower) & (test_hf <= upper)
        fractions.append(np.mean(inside, axis=0))
        counts.append(int(round(empirical_curve_coverage(test_hf, lower, upper) * test_parameters.shape[0])))
        multipliers.append(fit.multiplier)
        mean_widths.append(float(np.mean(upper - lower)))
    fit95 = CurveConformalCalibrator.fit(
        calibration_members, calibration_hf, confidence=0.95, relative_scale_floor=0.02
    )
    mean, spread, lower95, upper95 = fit95.band(test_members[:, representative, :])
    np.savez_compressed(
        DATA / f"{case}_calibration_plot.npz",
        frequencies=frequencies,
        representative_hf=test_hf[representative],
        representative_mean=mean,
        representative_lower=lower95,
        representative_upper=upper95,
        coverage_fractions=np.asarray(fractions),
        covered_counts=np.asarray(counts),
        conformal_multipliers=np.asarray(multipliers),
        mean_band_widths=np.asarray(mean_widths),
        test_curve_count=np.asarray(test_parameters.shape[0]),
        wilson_95_lower=np.asarray(_wilson(counts[2], test_parameters.shape[0])[0]),
        wilson_95_upper=np.asarray(_wilson(counts[2], test_parameters.shape[0])[1]),
    )

    # Fig12: stiffened modes 2-4 tracker and amplitude ranges.
    if case == "stiffened":
        eigenvalues, _ = spla.eigsh(
            result.family.nominal.K, k=12, M=result.family.nominal.M, sigma=0.0, which="LM"
        )
        f0 = np.sqrt(np.sort(eigenvalues)) / (2.0 * np.pi)
        s_e = uq_parameters[:, 0]
        s_rho = uq_parameters[:, 1]
        tracker_rows = []
        amp_rows = []
        for mode in (2, 3, 4):
            tracker = f0[mode - 1] * np.sqrt(s_e / s_rho)
            local = np.linspace(float(np.min(tracker)) - 6.0, float(np.max(tracker)) + 6.0, 61)
            local_curves = _obs(result, local, uq_parameters, observation_index)
            local_amps = np.asarray([float(np.max(local_curves[i])) for i in range(uq_parameters.shape[0])])
            tracker_rows.append(
                {"mode": mode, "frequency_min_hz": float(np.min(tracker)), "frequency_max_hz": float(np.max(tracker))}
            )
            amp_rows.append(
                {"mode": mode, "amplitude_min_mm": float(np.min(local_amps)), "amplitude_max_mm": float(np.max(local_amps))}
            )
        (DATA / "stiffened_modes234_ranges.json").write_text(
            json.dumps({"tracker": tracker_rows, "amplitudes": amp_rows}, indent=2), encoding="utf-8"
        )

    # Fig13: correlation widths for the stiffened plate.
    if case == "stiffened":
        widths = []
        rhos = (-0.5, 0.0, 0.5)
        for corr_index, corr in enumerate(rhos):
            ellipsoid = _ellipsoid(bounds, corr)
            parameters = ellipsoid.sample(500, 701 + 1000 + corr_index, boundary_fraction=0.5)
            curves = _obs(result, frequencies, parameters, observation_index)
            widths.append(np.max(curves, axis=0) - np.min(curves, axis=0))
        np.savez_compressed(
            DATA / "stiffened_correlation_widths.npz",
            frequencies=frequencies,
            widths=np.asarray(widths),
            correlations=np.asarray(rhos),
        )


def copy_existing() -> None:
    copies = [
        (RESULTS / "parameter_effects" / "stiffened" / "parameter_effect_frf.csv",
         DATA / "parameter_effect_frf.csv"),
        (RESULTS / "unseen_accuracy_current" / "flat_unseen_accuracy_current.csv",
         DATA / "flat_unseen_accuracy_current.csv"),
        (RESULTS / "unseen_accuracy_current" / "stiffened_unseen_accuracy_current.csv",
         DATA / "stiffened_unseen_accuracy_current.csv"),
        (RESULTS / "parametric_ablation_compact" / "ablation_summary_combined.json",
         DATA / "ablation_summary_combined.json"),
        (RESULTS / "ellipsoidal_uq" / "flat" / "ellipsoidal_uq_curves.npz",
         DATA / "flat_direct8_curves.npz"),
        (RESULTS / "ellipsoidal_uq" / "stiffened" / "ellipsoidal_uq_curves.npz",
         DATA / "stiffened_direct8_curves.npz"),
        (RESULTS / "cost_protocol_201" / "cost_summary.json", DATA / "cost_summary.json"),
        (RESULTS / "classical_modal_baselines" / "flat" / "baseline_summary.json",
         DATA / "flat_baseline_summary.json"),
        (RESULTS / "classical_modal_baselines" / "stiffened" / "baseline_summary.json",
         DATA / "stiffened_baseline_summary.json"),
        (RESULTS / "rbffd_convergence_validation" / "convergence_summary.json",
         DATA / "convergence_summary.json"),
        (RESULTS / "ellipsoid_extremum_search" / "flat" / "extremum_summary.json",
         DATA / "flat_extremum_summary.json"),
        (RESULTS / "ellipsoid_extremum_search" / "stiffened" / "extremum_summary.json",
         DATA / "stiffened_extremum_summary.json"),
        (RESULTS / "route_b_pilot_1500" / "summary.json", DATA / "route_b_pilot_summary.json"),
    ]
    for source, target in copies:
        if source.exists():
            shutil.copy2(source, target)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--data", type=Path, default=None,
        help="output directory for the figure data; defaults to <project>/figure_scripts/data. "
             "Pass the submission tree's figure_scripts/data to refresh the manuscript figures.")
    args = parser.parse_args()
    global DATA
    if args.data is not None:
        DATA = args.data
        DATA.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    DATA.mkdir(parents=True, exist_ok=True)
    prepare_geometry()
    for case in CASES:
        prepare_case(case, args.device)
    copy_existing()
    print(f"Figure data written to {DATA}", flush=True)


if __name__ == "__main__":
    main()

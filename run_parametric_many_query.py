# -*- coding: utf-8 -*-
"""Train and evaluate the optional parameterized many-query frequency operator.

The script enforces the four requirements needed for the intended paper
positioning:

1. ``mu`` is part of the neural-network input;
2. calibration and test parameter points are unseen during training;
3. surrogate response statistics are compared with direct constrained RBF-FD
   statistics on an independent high-fidelity sample;
4. offline cost, online query cost and the measured break-even query count are
   written to each result folder.

Uncertainty uses an independently initialized deep ensemble followed by a
split-conformal simultaneous band for complete FRF curves.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rbf_fd_full_order import (
    ModalResonanceConfig,
    ParameterBounds,
    ParametricOperatorConfig,
    aggregate_response_boundary,
    build_parametric_frequency_family,
    load_parametric_operator,
    predict_parametric_members,
    prepare_constrained_operators,
    save_parametric_operator,
    train_parametric_operator,
)
from rbf_fd_full_order.examples.cases import MeshConfig, build_case_system, get_case_definition
from rbf_fd_full_order.uncertainty import CurveConformalCalibrator, empirical_curve_coverage
from rbf_fd_full_order.visualization import apply_journal_style, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("flat", "hollow", "annular", "stiffened", "all"), default="all")
    parser.add_argument("--representation", choices=("pod", "full"), default="pod")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Reuse the existing case checkpoint and run only post-processing/evaluation.",
    )
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--n-train-parameters", type=int, default=20)
    parser.add_argument("--n-calibration-parameters", type=int, default=8)
    parser.add_argument("--n-test-parameters", type=int, default=8)
    parser.add_argument("--n-uq-parameters", type=int, default=500)
    parser.add_argument("--n-hf-statistics-parameters", type=int, default=30)
    parser.add_argument("--n-labeled", type=int, default=24)
    parser.add_argument("--n-physics", type=int, default=80)
    parser.add_argument("--sweep-points", type=int, default=401)
    parser.add_argument("--frequency-min", type=float, default=0.0)
    parser.add_argument("--frequency-max", type=float, default=200.0)
    parser.add_argument("--physics-weight", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--fourier-modes", type=int, default=4)
    parser.add_argument("--pod-modes", type=int, default=8)
    parser.add_argument("--modal-modes", type=int, default=8)
    parser.add_argument("--disable-modal-rational", action="store_true")
    parser.add_argument("--modal-relative-half-width", type=float, default=0.04)
    parser.add_argument("--modal-min-half-width-hz", type=float, default=0.75)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--parameter-coverage", type=float, default=0.95)
    parser.add_argument("--conformal-scale-floor", type=float, default=0.02)
    parser.add_argument("--young-scale-min", type=float, default=0.85)
    parser.add_argument("--young-scale-max", type=float, default=1.15)
    parser.add_argument("--density-scale-min", type=float, default=0.90)
    parser.add_argument("--density-scale-max", type=float, default=1.10)
    parser.add_argument("--pressure-scale-min", type=float, default=0.80)
    parser.add_argument("--pressure-scale-max", type=float, default=1.20)
    parser.add_argument("--loss-factor-min", type=float, default=0.005)
    parser.add_argument("--loss-factor-max", type=float, default=0.020)
    parser.add_argument("--square-ndiv", type=int, default=30)
    parser.add_argument("--plate-stencil", type=int, default=20)
    parser.add_argument("--plate-poly-degree", type=int, default=2)
    parser.add_argument("--beam-stencil", type=int, default=7)
    parser.add_argument("--hollow-sweep-points", type=int, default=15)
    parser.add_argument("--hollow-tangent-points-per-patch", type=int, default=16)
    parser.add_argument("--hollow-stencil", type=int, default=20)
    parser.add_argument("--hollow-poly-degree", type=int, default=2)
    parser.add_argument("--hole-diameter-ratio", type=float, default=0.25)
    parser.add_argument("--annular-nrad", type=int, default=30)
    parser.add_argument("--annular-ntheta", type=int, default=30)
    parser.add_argument("--annular-stencil", type=int, default=15)
    parser.add_argument("--annular-outer-diameter", type=float, default=1.0)
    parser.add_argument("--annular-hole-diameter-ratio", type=float, default=0.25)
    parser.add_argument("--rbf-condition-limit", type=float, default=1.0e8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--out", type=Path, default=Path("results") / "parametric_many_query")
    args = parser.parse_args()
    if args.frequency_max <= args.frequency_min:
        parser.error("--frequency-max must exceed --frequency-min")
    for name in (
        "n_train_parameters", "n_calibration_parameters", "n_test_parameters",
        "n_uq_parameters", "n_hf_statistics_parameters", "n_labeled", "n_physics", "sweep_points",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.sweep_points < 2:
        parser.error("--sweep-points must be at least two")
    if not 0.0 < args.confidence < 1.0 or not 0.0 < args.parameter_coverage < 1.0:
        parser.error("confidence and parameter coverage must lie in (0, 1)")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    return args


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bounds(args: argparse.Namespace) -> ParameterBounds:
    return ParameterBounds(
        lower=np.asarray([
            args.young_scale_min,
            args.density_scale_min,
            args.pressure_scale_min,
            args.loss_factor_min,
        ], dtype=float),
        upper=np.asarray([
            args.young_scale_max,
            args.density_scale_max,
            args.pressure_scale_max,
            args.loss_factor_max,
        ], dtype=float),
    ).validated()


def _mesh(args: argparse.Namespace) -> MeshConfig:
    return MeshConfig(
        square_ndiv=args.square_ndiv,
        plate_stencil=args.plate_stencil,
        plate_poly_degree=args.plate_poly_degree,
        beam_stencil=args.beam_stencil,
        hollow_sweep_points=args.hollow_sweep_points,
        hollow_tangent_points_per_patch=args.hollow_tangent_points_per_patch,
        hollow_stencil=args.hollow_stencil,
        hollow_poly_degree=args.hollow_poly_degree,
        hole_diameter_ratio=args.hole_diameter_ratio,
        annular_nrad=args.annular_nrad,
        annular_ntheta=args.annular_ntheta,
        annular_stencil=args.annular_stencil,
        annular_outer_diameter=args.annular_outer_diameter,
        annular_hole_diameter_ratio=args.annular_hole_diameter_ratio,
        rbf_condition_limit=args.rbf_condition_limit,
    )


def _modal_config(args: argparse.Namespace) -> ModalResonanceConfig:
    return ModalResonanceConfig(
        enabled=not bool(args.disable_modal_rational),
        n_modes=int(args.modal_modes),
        relative_half_width=float(args.modal_relative_half_width),
        minimum_half_width_hz=float(args.modal_min_half_width_hz),
        exclude_labeled=True,
        exclude_physics=True,
    )


def _operator_config(args: argparse.Namespace) -> ParametricOperatorConfig:
    return ParametricOperatorConfig(
        representation=args.representation,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        width=args.width,
        depth=args.depth,
        fourier_modes=args.fourier_modes,
        pod_modes=args.pod_modes,
        ensemble_size=args.ensemble_size,
        physics_weight=args.physics_weight,
        warmup_epochs=max(args.epochs // 4, 1),
        log_every=max(args.epochs // 30, 1),
        device=args.device,
        modal_resonance=_modal_config(args),
    )


def _transverse_observation_index(operators, node_count: int, nominal_responses: np.ndarray) -> int:
    if operators.reduction_mode == "fixed_dofs" and operators.free_dofs is not None:
        candidates = np.flatnonzero(np.asarray(operators.free_dofs) < int(node_count))
    else:
        candidates = np.arange(operators.n_free, dtype=int)
    if not candidates.size:
        raise RuntimeError("No transverse response coordinates were found.")
    local = int(np.argmax(np.max(np.abs(nominal_responses[:, candidates]), axis=0)))
    return int(candidates[local])


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - reference) / max(float(np.linalg.norm(reference)), 1.0e-30))


def _peak_frequency(frequencies: np.ndarray, curves: np.ndarray) -> np.ndarray:
    return frequencies[np.argmax(curves, axis=-1)]


def _confidence_label(target: float, test_coverage: float, in_domain: bool) -> str:
    if not in_domain:
        return "low_extrapolation"
    if test_coverage >= target - 0.05:
        return "high"
    if test_coverage >= target - 0.15:
        return "moderate"
    return "low"


def run_case(case: str, args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.out / case
    output_dir.mkdir(parents=True, exist_ok=True)
    definition = get_case_definition(case)
    bounds = _bounds(args)

    assembly_started = time.perf_counter()
    system = build_case_system(case, _mesh(args))
    operators = prepare_constrained_operators(system)
    assembly_s = time.perf_counter() - assembly_started

    family_started = time.perf_counter()
    family = build_parametric_frequency_family(
        operators,
        bounds=bounds,
        modal_config=_modal_config(args),
        frequency_min_hz=args.frequency_min,
        frequency_max_hz=args.frequency_max,
    )
    modal_preparation_s = time.perf_counter() - family_started

    train_mu = bounds.sample_latin_hypercube(args.n_train_parameters, args.seed + 101)
    calibration_mu = bounds.sample_latin_hypercube(args.n_calibration_parameters, args.seed + 202)
    test_mu = bounds.sample_latin_hypercube(args.n_test_parameters, args.seed + 303)
    uq_mu = bounds.sample_latin_hypercube(args.n_uq_parameters, args.seed + 404)
    hf_mu = bounds.sample_latin_hypercube(args.n_hf_statistics_parameters, args.seed + 505)

    labeled = np.linspace(args.frequency_min, args.frequency_max, args.n_labeled)
    physics = np.unique(np.r_[labeled, np.linspace(args.frequency_min, args.frequency_max, args.n_physics)])
    sweep = np.linspace(args.frequency_min, args.frequency_max, args.sweep_points)

    checkpoint_path = output_dir / "parametric_operator.pt"
    if args.skip_training:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"--skip-training requested, but checkpoint does not exist: {checkpoint_path}"
            )
        result = load_parametric_operator(checkpoint_path, family, device=args.device)
    else:
        result = train_parametric_operator(
            family,
            train_mu,
            labeled,
            physics,
            config=_operator_config(args),
            seed=args.seed,
        )
        save_parametric_operator(result, checkpoint_path)

    # Select a fixed transverse observation coordinate using the nominal median parameter.
    observation_started = time.perf_counter()
    nominal_reference = family.solve_grid(sweep, bounds.center)
    observation_index = _transverse_observation_index(operators, system.nodes.shape[0], nominal_reference)
    observation_selection_s = time.perf_counter() - observation_started

    # Unseen-parameter full-field test at frequencies never used as parameter labels.
    test_frequencies = np.asarray(definition.test_frequencies_hz, dtype=float)
    test_reference_started = time.perf_counter()
    test_reference = family.solve_grid(test_frequencies, test_mu)
    test_reference_s = time.perf_counter() - test_reference_started
    test_prediction_started = time.perf_counter()
    test_members = predict_parametric_members(result, test_frequencies, test_mu)
    test_prediction_s = time.perf_counter() - test_prediction_started
    test_mean = np.mean(test_members, axis=0)
    field_rows: list[dict[str, object]] = []
    for parameter_index, point in enumerate(test_mu):
        for frequency_index, frequency in enumerate(test_frequencies):
            field_rows.append(
                {
                    "case": case,
                    "parameter_index": parameter_index,
                    **{name: float(value) for name, value in zip(bounds.names, point)},
                    "frequency_hz": float(frequency),
                    "relative_full_field_error": _relative_l2(
                        test_mean[parameter_index, frequency_index],
                        test_reference[parameter_index, frequency_index],
                    ),
                }
            )
    _write_csv(output_dir / "unseen_parameter_field_errors.csv", field_rows)

    # Split-conformal calibration on unseen calibration parameters.
    calibration_reference_started = time.perf_counter()
    calibration_reference = family.solve_grid(sweep, calibration_mu)
    calibration_reference_s = time.perf_counter() - calibration_reference_started
    calibration_prediction_started = time.perf_counter()
    calibration_members_response = predict_parametric_members(result, sweep, calibration_mu)
    calibration_prediction_s = time.perf_counter() - calibration_prediction_started
    calibration_member_curves = np.abs(calibration_members_response[..., observation_index]) * 1.0e3
    calibration_reference_curves = np.abs(calibration_reference[..., observation_index]) * 1.0e3
    calibrator = CurveConformalCalibrator.fit(
        calibration_member_curves,
        calibration_reference_curves,
        confidence=args.confidence,
        relative_scale_floor=args.conformal_scale_floor,
    )

    # Independent unseen test curves for post-calibration coverage reporting.
    test_curve_reference_started = time.perf_counter()
    test_curve_reference = family.solve_grid(sweep, test_mu)
    test_curve_reference_s = time.perf_counter() - test_curve_reference_started
    test_curve_prediction_started = time.perf_counter()
    test_curve_members_response = predict_parametric_members(result, sweep, test_mu)
    test_curve_prediction_s = time.perf_counter() - test_curve_prediction_started
    test_member_curves = np.abs(test_curve_members_response[..., observation_index]) * 1.0e3
    test_reference_curves = np.abs(test_curve_reference[..., observation_index]) * 1.0e3
    _, _, test_lower, test_upper = calibrator.band(test_member_curves)
    test_curve_coverage = empirical_curve_coverage(test_reference_curves, test_lower, test_upper)

    # Online UQ envelope.  Only the selected observation coordinate is needed
    # here; predicting all fields for 500 UQ parameters would allocate tens of
    # GiB (ensemble, parameter, frequency, DOF) and can exceed host memory.
    online_started = time.perf_counter()
    uq_observation_chunks = []
    uq_chunk_size = 8
    for start in range(0, uq_mu.shape[0], uq_chunk_size):
        chunk_response = predict_parametric_members(
            result,
            sweep,
            uq_mu[start : start + uq_chunk_size],
        )
        uq_observation_chunks.append(
            np.abs(chunk_response[..., observation_index]) * 1.0e3
        )
    online_uq_s = time.perf_counter() - online_started
    uq_member_curves = np.concatenate(uq_observation_chunks, axis=1)
    boundary = aggregate_response_boundary(
        uq_member_curves,
        sweep,
        calibrator,
        parameter_coverage=args.parameter_coverage,
    )

    # High-fidelity statistical reference on an independent parameter sample.
    hf_started = time.perf_counter()
    hf_response = family.solve_grid(sweep, hf_mu)
    hf_statistics_s = time.perf_counter() - hf_started
    hf_curves = np.abs(hf_response[..., observation_index]) * 1.0e3
    surrogate_hf_started = time.perf_counter()
    surrogate_hf_response = predict_parametric_members(result, sweep, hf_mu)
    surrogate_hf_s = time.perf_counter() - surrogate_hf_started
    surrogate_hf_curves = np.mean(np.abs(surrogate_hf_response[..., observation_index]) * 1.0e3, axis=0)
    tail = 0.5 * (1.0 - args.parameter_coverage)
    hf_mean = np.mean(hf_curves, axis=0)
    hf_std = np.std(hf_curves, axis=0)
    hf_lower = np.quantile(hf_curves, tail, axis=0)
    hf_upper = np.quantile(hf_curves, 1.0 - tail, axis=0)
    surrogate_same_mean = np.mean(surrogate_hf_curves, axis=0)
    surrogate_same_std = np.std(surrogate_hf_curves, axis=0)
    surrogate_same_lower = np.quantile(surrogate_hf_curves, tail, axis=0)
    surrogate_same_upper = np.quantile(surrogate_hf_curves, 1.0 - tail, axis=0)

    boundary_rows = []
    for index, frequency in enumerate(sweep):
        boundary_rows.append(
            {
                "case": case,
                "frequency_hz": float(frequency),
                "surrogate_mean_mm": float(boundary.mean[index]),
                "surrogate_median_mm": float(boundary.median[index]),
                "parameter_lower_mm": float(boundary.parameter_lower[index]),
                "parameter_upper_mm": float(boundary.parameter_upper[index]),
                "conformal_lower_mm": float(boundary.confidence_lower[index]),
                "conformal_upper_mm": float(boundary.confidence_upper[index]),
                "ensemble_spread_mm": float(boundary.ensemble_spread[index]),
                "confidence_score": float(boundary.confidence_score[index]),
                "hf_mean_mm": float(hf_mean[index]),
                "hf_std_mm": float(hf_std[index]),
                "hf_lower_mm": float(hf_lower[index]),
                "hf_upper_mm": float(hf_upper[index]),
            }
        )
    _write_csv(output_dir / "frf_response_boundary.csv", boundary_rows)

    statistics = {
        "mean_curve_relative_l2": _relative_l2(surrogate_same_mean, hf_mean),
        "std_curve_relative_l2": _relative_l2(surrogate_same_std, hf_std),
        "lower_boundary_relative_l2": _relative_l2(surrogate_same_lower, hf_lower),
        "upper_boundary_relative_l2": _relative_l2(surrogate_same_upper, hf_upper),
        "mean_peak_frequency_error_hz": float(
            abs(_peak_frequency(sweep, surrogate_same_mean[None, :])[0] - _peak_frequency(sweep, hf_mean[None, :])[0])
        ),
        "mean_unseen_full_field_error": float(np.mean([row["relative_full_field_error"] for row in field_rows])),
        "max_unseen_full_field_error": float(np.max([row["relative_full_field_error"] for row in field_rows])),
        "target_curve_confidence": float(args.confidence),
        "calibration_empirical_curve_coverage": float(calibrator.empirical_curve_coverage),
        "unseen_test_curve_coverage": float(test_curve_coverage),
        "conformal_multiplier": float(calibrator.multiplier),
        "conformal_absolute_scale_floor_mm": float(calibrator.absolute_scale_floor),
    }
    with (output_dir / "high_fidelity_statistical_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2)

    # Measured many-query economics.
    offline_s = (
        assembly_s
        + modal_preparation_s
        + result.high_fidelity_generation_s
        + result.ensemble_training_s
        + observation_selection_s
        + calibration_reference_s
    )
    direct_curve_s = hf_statistics_s / max(int(args.n_hf_statistics_parameters), 1)
    operator_curve_s = online_uq_s / max(int(args.n_uq_parameters), 1)
    savings = direct_curve_s - operator_curve_s
    break_even = None if savings <= 0.0 else int(math.ceil(offline_s / savings))
    in_domain = bool(np.all(bounds.contains(np.vstack((calibration_mu, test_mu, uq_mu, hf_mu)))))
    confidence_label = _confidence_label(args.confidence, test_curve_coverage, in_domain)
    cost = {
        "assembly_s": assembly_s,
        "modal_preparation_s": modal_preparation_s,
        "high_fidelity_training_data_s": result.high_fidelity_generation_s,
        "high_fidelity_training_solve_count": result.high_fidelity_solve_count,
        "ensemble_training_s": result.ensemble_training_s,
        "observation_selection_s": observation_selection_s,
        "calibration_high_fidelity_s": calibration_reference_s,
        "offline_total_s": offline_s,
        "direct_full_sweep_per_parameter_s": direct_curve_s,
        "operator_ensemble_full_sweep_per_parameter_s": operator_curve_s,
        "per_query_savings_s": savings,
        "break_even_parameter_queries": break_even,
        "test_reference_s": test_reference_s,
        "test_prediction_s": test_prediction_s,
        "calibration_prediction_s": calibration_prediction_s,
        "test_curve_reference_s": test_curve_reference_s,
        "test_curve_prediction_s": test_curve_prediction_s,
        "high_fidelity_statistics_s": hf_statistics_s,
        "surrogate_same_sample_statistics_s": surrogate_hf_s,
    }
    with (output_dir / "cost_and_break_even.json").open("w", encoding="utf-8") as handle:
        json.dump(cost, handle, indent=2)

    parameter_rows = []
    for split, values in (
        ("train", train_mu),
        ("calibration_unseen", calibration_mu),
        ("test_unseen", test_mu),
        ("uq_online", uq_mu),
        ("hf_statistics", hf_mu),
    ):
        for index, point in enumerate(values):
            parameter_rows.append({"split": split, "index": index, **{name: float(value) for name, value in zip(bounds.names, point)}})
    _write_csv(output_dir / "parameter_samples.csv", parameter_rows)

    metadata = {
        "case": case,
        "representation": args.representation,
        "operator_mapping": "(frequency, mu) -> complex constrained response",
        "parameter_names": list(bounds.names),
        "parameter_lower": bounds.lower.tolist(),
        "parameter_upper": bounds.upper.tolist(),
        "training_parameter_count": int(train_mu.shape[0]),
        "calibration_parameter_count": int(calibration_mu.shape[0]),
        "test_parameter_count": int(test_mu.shape[0]),
        "uq_parameter_count": int(uq_mu.shape[0]),
        "high_fidelity_statistics_parameter_count": int(hf_mu.shape[0]),
        "unseen_parameter_test": True,
        "modal_rational_enabled": family.modal is not None,
        "modal_modes": None if family.modal is None else int(family.modal.n_modes),
        "pod_modes": None if result.pod_basis is None else int(result.pod_basis.n_modes),
        "ensemble_size": int(len(result.members)),
        "training_best_epoch_selection": "minimum monitored total_loss (data + ramped physics objective)",
        "best_epoch_per_member": [int(member.best_epoch) for member in result.members],
        "best_total_loss_per_member": [float(member.best_total_loss) for member in result.members],
        "observation_constrained_index": int(observation_index),
        "confidence_method": "independent deep ensemble + split-conformal simultaneous curve band",
        "confidence_label": confidence_label,
        "all_reported_parameter_samples_inside_training_box": in_domain,
        "frequency_interval_hz": [float(args.frequency_min), float(args.frequency_max)],
        "sweep_points": int(args.sweep_points),
        "spatial_autograd": False,
        "mesh": {
            "square_ndiv": int(args.square_ndiv),
            "square_node_count_per_axis": int(args.square_ndiv) + 1,
            "plate_stencil": int(args.plate_stencil),
            "plate_poly_degree": int(args.plate_poly_degree),
            "beam_stencil": int(args.beam_stencil),
            "hollow_sweep_points": int(args.hollow_sweep_points),
            "hollow_tangent_points_per_patch": int(args.hollow_tangent_points_per_patch),
            "hollow_stencil": int(args.hollow_stencil),
            "hollow_poly_degree": int(args.hollow_poly_degree),
            "hole_diameter_ratio": float(args.hole_diameter_ratio),
            "annular_nrad": int(args.annular_nrad),
            "annular_ntheta": int(args.annular_ntheta),
            "annular_stencil": int(args.annular_stencil),
            "annular_outer_diameter": float(args.annular_outer_diameter),
            "annular_hole_diameter_ratio": float(args.annular_hole_diameter_ratio),
            "rbf_condition_limit": float(args.rbf_condition_limit),
        },
        "parameter_scaling_assumption": {
            "K_mu": "young_scale * K0",
            "M_mu": "density_scale * M0",
            "f_mu": "pressure_scale * f0",
            "structural_loss_factor": "mu[loss_factor]",
            "geometry": "fixed within each case",
        },
        "statistics_file": "high_fidelity_statistical_comparison.json",
        "cost_file": "cost_and_break_even.json",
        "boundary_file": "frf_response_boundary.csv",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    apply_journal_style()
    fig, axis = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    axis.fill_between(
        sweep, boundary.confidence_lower, boundary.confidence_upper,
        color="#808080", alpha=0.18, label=f"Conformalized {100*args.parameter_coverage:.0f}% boundary",
    )
    axis.fill_between(
        sweep, boundary.parameter_lower, boundary.parameter_upper,
        color="#0072BD", alpha=0.22, label="Operator parameter interval",
    )
    axis.plot(sweep, boundary.median, color="#D62728", linewidth=1.2, label="Operator median")
    axis.plot(sweep, hf_mean, color="#000000", linestyle="--", linewidth=1.0, label="High-fidelity mean")
    axis.plot(sweep, hf_lower, color="#4D4D4D", linestyle=":", linewidth=0.9, label="High-fidelity bounds")
    axis.plot(sweep, hf_upper, color="#4D4D4D", linestyle=":", linewidth=0.9)
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel(r"$|w|$ (mm)")
    axis.set_title(definition.title)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", ncol=2)
    save_figure(fig, output_dir / "parametric_frf_boundary.png")
    plt.close(fig)

    return {
        "case": case,
        "mean_unseen_full_field_error": statistics["mean_unseen_full_field_error"],
        "unseen_test_curve_coverage": test_curve_coverage,
        "confidence_label": confidence_label,
        "offline_total_s": offline_s,
        "direct_curve_s": direct_curve_s,
        "operator_curve_s": operator_curve_s,
        "break_even_parameter_queries": break_even,
        "best_epoch_per_member": [int(member.best_epoch) for member in result.members],
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = ("flat", "annular", "stiffened") if args.case == "all" else (args.case,)
    summary = [run_case(case, args) for case in cases]
    _write_csv(args.out / "three_case_parametric_summary.csv", summary)
    with (args.out / "run_configuration.json").open("w", encoding="utf-8") as handle:
        serializable = vars(args).copy()
        serializable["out"] = str(serializable["out"])
        json.dump(serializable, handle, indent=2)


if __name__ == "__main__":
    main()

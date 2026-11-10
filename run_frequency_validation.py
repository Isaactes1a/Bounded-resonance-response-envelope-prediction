# -*- coding: utf-8 -*-
"""Run one case of the current FOM/POD eight-configuration experiment.

The runner covers data-only versus physics-constrained training, optional
modal rational reconstruction, and online residual correction.  A dense sweep
is opt-in so that the same entry point can also be used for quick smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rbf_fd_full_order import (
    CoefficientRefinementConfig,
    FullOrderNetworkConfig,
    ModalResonanceConfig,
    PODFrequencyNetworkConfig,
    constraint_metric,
    predict_full_order_response,
    predict_pod_coordinates,
    predict_pod_response,
    prepare_constrained_operators,
    refine_full_order_response,
    refine_full_order_responses,
    refine_pod_response,
    refine_pod_responses,
    relative_field_error,
    relative_physics_residual,
    train_pod_frequency_net,
    train_full_order_frequency_net,
    build_modal_resonance_interface,
)
from rbf_fd_full_order.examples.cases import (
    MeshConfig,
    build_case_system,
    get_case_definition,
    labeled_and_physics_frequencies,
)
from rbf_fd_full_order.visualization import (
    COLOR_DATA_ONLY,
    COLOR_PROPOSED,
    apply_journal_style,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("flat", "hollow", "annular", "stiffened"), required=True)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--n-labeled", type=int, default=48)
    parser.add_argument("--n-physics", type=int, default=160)
    parser.add_argument("--physics-weight", type=float, default=0.02)
    parser.add_argument("--net-lr", type=float, default=2.0e-4)
    parser.add_argument("--net-width", type=int, default=96)
    parser.add_argument("--net-depth", type=int, default=3)
    parser.add_argument("--fourier-modes", type=int, default=4)
    parser.add_argument(
        "--modal-resonance",
        action="store_true",
        help="Enable the shared modal rational branch for both full-order and POD models.",
    )
    parser.add_argument("--modal-modes", type=int, default=8)
    parser.add_argument("--modal-relative-half-width", type=float, default=0.04)
    parser.add_argument("--modal-min-half-width-hz", type=float, default=0.75)
    parser.add_argument("--modal-transition-fraction", type=float, default=0.20)
    parser.add_argument("--modal-exclusion-threshold", type=float, default=0.50)
    parser.add_argument(
        "--modal-keep-resonance-labels",
        action="store_true",
        help="Keep resonance-window labels in direct background fitting.",
    )
    parser.add_argument(
        "--modal-keep-resonance-physics",
        action="store_true",
        help="Keep resonance-window frequencies in the physics batch.",
    )
    parser.add_argument("--refine-steps", type=int, default=0)
    parser.add_argument("--refine-lr", type=float, default=2.0e-4)
    parser.add_argument("--refine-prior-weight", type=float, default=0.0)
    parser.add_argument("--refine-weight-decay", type=float, default=0.0)
    parser.add_argument("--compare-pod", action="store_true")
    parser.add_argument("--pod-modes", type=int, default=6)
    parser.add_argument("--pod-epochs", type=int, default=None)
    parser.add_argument("--pod-lr", type=float, default=None)
    parser.add_argument("--pod-physics-weight", type=float, default=None)
    parser.add_argument(
        "--pod-gradient-balance",
        action="store_true",
        help="Set the POD physics weight from initial parameter-gradient norms.",
    )
    parser.add_argument(
        "--pod-gradient-balance-ratio",
        type=float,
        default=1.0,
        help="Target ||lambda grad(L_phys)|| / ||grad(L_data)|| for POD at initialization.",
    )
    parser.add_argument("--pod-width", type=int, default=64)
    parser.add_argument("--pod-depth", type=int, default=3)
    parser.add_argument("--pod-refine-steps", type=int, default=None)
    parser.add_argument("--pod-refine-lr", type=float, default=None)
    parser.add_argument("--loss-factor", type=float, default=0.01)
    parser.add_argument("--square-ndiv", type=int, default=30)
    parser.add_argument("--plate-stencil", type=int, default=20)
    parser.add_argument("--plate-poly-degree", type=int, default=2)
    parser.add_argument("--beam-stencil", type=int, default=7)
    parser.add_argument("--hollow-sweep-points", type=int, default=15)
    parser.add_argument(
        "--hollow-tangent-points-per-patch",
        type=int,
        default=16,
        help="Nodes along each quarter-circle and corresponding outer square edge.",
    )
    parser.add_argument("--hollow-stencil", type=int, default=20)
    parser.add_argument("--hollow-poly-degree", type=int, default=2)
    parser.add_argument("--hole-diameter-ratio", type=float, default=0.25)
    parser.add_argument("--annular-nrad", type=int, default=30)
    parser.add_argument("--annular-ntheta", type=int, default=30)
    parser.add_argument("--annular-stencil", type=int, default=15)
    parser.add_argument("--annular-outer-diameter", type=float, default=1.0)
    parser.add_argument("--annular-hole-diameter-ratio", type=float, default=0.25)
    parser.add_argument("--rbf-condition-limit", type=float, default=1.0e8)
    parser.add_argument(
        "--frequency-min",
        type=float,
        default=None,
        help="Optional lower bound for labeled and physics samples.",
    )
    parser.add_argument(
        "--frequency-max",
        type=float,
        default=None,
        help="Optional upper bound for labeled and physics samples.",
    )
    parser.add_argument(
        "--sweep-frequency-min",
        type=float,
        default=None,
        help="Lower bound of an optional dense post-training response sweep.",
    )
    parser.add_argument(
        "--sweep-frequency-max",
        type=float,
        default=None,
        help="Upper bound of an optional dense post-training response sweep.",
    )
    parser.add_argument(
        "--sweep-points",
        type=int,
        default=0,
        help="Number of response-sweep points; zero disables sweep generation.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for network training and residual refinement.",
    )
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "unified_square_cases_modal_sweep_cuda",
    )
    args = parser.parse_args()
    if args.refine_steps < 0:
        parser.error("--refine-steps must be nonnegative")
    if args.pod_refine_steps is not None and args.pod_refine_steps < 0:
        parser.error("--pod-refine-steps must be nonnegative")
    if args.pod_modes < 1:
        parser.error("--pod-modes must be positive")
    if args.pod_gradient_balance_ratio < 0.0:
        parser.error("--pod-gradient-balance-ratio must be nonnegative")
    if (args.frequency_min is None) != (args.frequency_max is None):
        parser.error("--frequency-min and --frequency-max must be provided together")
    if args.frequency_min is not None and args.frequency_max <= args.frequency_min:
        parser.error("--frequency-max must exceed --frequency-min")
    if args.sweep_points and args.sweep_points < 2:
        parser.error("--sweep-points must be zero or at least two")
    if args.sweep_points:
        if (args.sweep_frequency_min is None) != (args.sweep_frequency_max is None):
            parser.error("--sweep-frequency-min and --sweep-frequency-max must be provided together")
        if args.sweep_frequency_min is not None and args.sweep_frequency_max <= args.sweep_frequency_min:
            parser.error("--sweep-frequency-max must exceed --sweep-frequency-min")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested but CUDA is not available in this Python environment")
    return args


def _write_rows(path: Path, rows) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _modal_config(args: argparse.Namespace) -> ModalResonanceConfig:
    return ModalResonanceConfig(
        enabled=bool(args.modal_resonance),
        n_modes=int(args.modal_modes),
        relative_half_width=float(args.modal_relative_half_width),
        minimum_half_width_hz=float(args.modal_min_half_width_hz),
        transition_fraction=float(args.modal_transition_fraction),
        exclusion_threshold=float(args.modal_exclusion_threshold),
        exclude_labeled=not bool(args.modal_keep_resonance_labels),
        exclude_physics=not bool(args.modal_keep_resonance_physics),
    )


def _config(args: argparse.Namespace, physics_weight: float) -> FullOrderNetworkConfig:
    return FullOrderNetworkConfig(
        epochs=args.epochs,
        learning_rate=args.net_lr,
        physics_weight=physics_weight,
        warmup_epochs=max(args.epochs // 4, 1),
        log_every=max(args.epochs // 30, 1),
        width=args.net_width,
        depth=args.net_depth,
        fourier_modes=args.fourier_modes,
        device=args.device,
        modal_resonance=_modal_config(args),
    )


def _pod_config(args: argparse.Namespace, physics_weight: float) -> PODFrequencyNetworkConfig:
    epochs = args.epochs if args.pod_epochs is None else int(args.pod_epochs)
    learning_rate = args.net_lr if args.pod_lr is None else float(args.pod_lr)
    if physics_weight and args.pod_physics_weight is not None:
        physics_weight = float(args.pod_physics_weight)
    return PODFrequencyNetworkConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        physics_weight=physics_weight,
        balance_physics_by_initial_gradient=bool(args.pod_gradient_balance and physics_weight > 0.0),
        target_physics_to_data_gradient_ratio=float(args.pod_gradient_balance_ratio),
        warmup_epochs=max(epochs // 4, 1),
        log_every=max(epochs // 30, 1),
        width=args.pod_width,
        depth=args.pod_depth,
        fourier_modes=args.fourier_modes,
        device=args.device,
        modal_resonance=_modal_config(args),
    )


def _refinement_config(args: argparse.Namespace, steps: int, learning_rate: float) -> CoefficientRefinementConfig:
    return CoefficientRefinementConfig(
        steps=int(steps),
        learning_rate=float(learning_rate),
        weight_decay=float(args.refine_weight_decay),
        prior_weight=float(args.refine_prior_weight),
        log_every=max(int(steps) // 25, 1),
        device=args.device,
    )


def _tag_history(case: str, method: str, frequency_hz: float, history) -> list[dict[str, object]]:
    return [
        {
            "case": case,
            "method": method,
            "frequency_hz": float(frequency_hz),
            **row,
        }
        for row in history
    ]


def _trainable_parameter_count(network: torch.nn.Module) -> int:
    """Return the number of scalar weights optimized during offline training."""

    return int(sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad))


def _configuration_row(
    *,
    case: str,
    method: str,
    representation: str,
    training_objective: str,
    online_correction: bool,
    validation_rows: list[dict[str, object]],
    timing: dict[str, float | int | str | bool],
    network_trainable_parameters: int,
    frozen_pod_state_scalars: int,
    online_refinement_variables: int,
    configured_physics_weight: float,
    effective_physics_weight: float,
    initial_data_gradient_norm: float | None,
    initial_physics_gradient_norm: float | None,
) -> dict[str, object]:
    """Aggregate one of the eight workflow configurations over test frequencies."""

    field_errors = np.asarray([float(row["relative_field_error"]) for row in validation_rows], dtype=float)
    residuals = np.asarray([float(row["relative_physics_residual"]) for row in validation_rows], dtype=float)
    constraints = np.asarray([float(row["constraint_metric"]) for row in validation_rows], dtype=float)
    return {
        "case": case,
        "method": method,
        "representation": representation,
        "offline_training_objective": training_objective,
        "online_residual_correction": bool(online_correction),
        "test_frequency_count": int(field_errors.size),
        "mean_relative_field_error": float(np.mean(field_errors)),
        "max_relative_field_error": float(np.max(field_errors)),
        "mean_relative_physics_residual": float(np.mean(residuals)),
        "max_relative_physics_residual": float(np.max(residuals)),
        "max_constraint_metric": float(np.max(constraints)),
        "offline_training_s": float(timing["offline_training_s"]),
        "prediction_s": float(timing["prediction_s"]),
        "refinement_s": float(timing["refinement_s"]),
        "end_to_end_s": float(timing["end_to_end_s"]),
        "network_trainable_parameters": int(network_trainable_parameters),
        "frozen_pod_state_scalars": int(frozen_pod_state_scalars),
        "persistent_workflow_scalars": int(network_trainable_parameters + frozen_pod_state_scalars),
        "online_refinement_variables": int(online_refinement_variables),
        "configured_physics_weight": float(configured_physics_weight),
        "effective_physics_weight": float(effective_physics_weight),
        "initial_data_gradient_norm": initial_data_gradient_norm,
        "initial_physics_gradient_norm": initial_physics_gradient_norm,
        "shared_offline_model": str(timing["shared_offline_model"]),
    }


def _write_frequency_sweep(
    *,
    args: argparse.Namespace,
    definition,
    operators,
    full_results,
    pod_results,
    pod_refine_steps: int,
    output_dir: Path,
) -> dict[str, object]:
    """Evaluate all eight routes over a dense interval after offline training."""

    if int(args.sweep_points) == 0:
        return {"enabled": False}
    sweep_min = (
        float(args.sweep_frequency_min)
        if args.sweep_frequency_min is not None
        else float(args.frequency_min if args.frequency_min is not None else definition.frequency_min_hz)
    )
    sweep_max = (
        float(args.sweep_frequency_max)
        if args.sweep_frequency_max is not None
        else float(args.frequency_max if args.frequency_max is not None else definition.frequency_max_hz)
    )
    frequencies = np.linspace(sweep_min, sweep_max, int(args.sweep_points))
    reference = np.asarray([operators.solve(float(frequency), args.loss_factor) for frequency in frequencies])
    observation_index = int(np.argmax(np.max(np.abs(reference), axis=0)))
    sweep_predictions: dict[str, np.ndarray] = {
        "full_data_only": predict_full_order_response(full_results["full_data_only"], frequencies),
        "full_physics": predict_full_order_response(full_results["full_physics"], frequencies),
    }
    if args.refine_steps > 0:
        refinement_config = _refinement_config(args, args.refine_steps, args.refine_lr)
        for base_method in ("full_data_only", "full_physics"):
            sweep_predictions[f"{base_method}_refined"] = refine_full_order_responses(
                operators,
                frequencies,
                sweep_predictions[base_method],
                config=refinement_config,
                loss_factor=args.loss_factor,
            )
    if pod_results:
        sweep_predictions["pod_data_only"] = predict_pod_response(pod_results["pod_data_only"], frequencies)
        sweep_predictions["pod_physics"] = predict_pod_response(pod_results["pod_physics"], frequencies)
        if pod_refine_steps > 0:
            refinement_config = _refinement_config(
                args,
                pod_refine_steps,
                args.refine_lr if args.pod_refine_lr is None else float(args.pod_refine_lr),
            )
            for base_method in ("pod_data_only", "pod_physics"):
                alpha0 = predict_pod_coordinates(pod_results[base_method], frequencies)
                sweep_predictions[f"{base_method}_refined"] = refine_pod_responses(
                    operators,
                    frequencies,
                    pod_results[base_method],
                    alpha0,
                    config=refinement_config,
                    loss_factor=args.loss_factor,
                )
    method_order = (
        "full_data_only",
        "full_data_only_refined",
        "full_physics",
        "full_physics_refined",
        "pod_data_only",
        "pod_data_only_refined",
        "pod_physics",
        "pod_physics_refined",
    )
    expected = tuple(method for method in method_order if method in sweep_predictions)
    rows = []
    for index, frequency in enumerate(frequencies):
        row: dict[str, object] = {
            "case": args.case,
            "frequency_hz": float(frequency),
            "observation_index": observation_index,
            "reference_abs_mm": float(abs(reference[index, observation_index]) * 1.0e3),
        }
        for method in expected:
            value = sweep_predictions[method][index, observation_index]
            row[f"{method}_real_m"] = float(value.real)
            row[f"{method}_imag_m"] = float(value.imag)
            row[f"{method}_abs_mm"] = float(abs(value) * 1.0e3)
        rows.append(row)
    _write_rows(output_dir / "frequency_sweep_observation.csv", rows)
    np.save(output_dir / "frequency_sweep_hz.npy", frequencies)
    np.save(output_dir / "frequency_sweep_reference_real.npy", reference.real)
    np.save(output_dir / "frequency_sweep_reference_imag.npy", reference.imag)
    for method, response in sweep_predictions.items():
        np.save(output_dir / f"frequency_sweep_{method}_real.npy", response.real)
        np.save(output_dir / f"frequency_sweep_{method}_imag.npy", response.imag)
    return {
        "enabled": True,
        "frequency_min_hz": sweep_min,
        "frequency_max_hz": sweep_max,
        "points": int(args.sweep_points),
        "observation_index": observation_index,
        "csv": "frequency_sweep_observation.csv",
    }


def main() -> None:
    args = parse_args()
    output_dir = args.out / args.case
    output_dir.mkdir(parents=True, exist_ok=True)
    definition = get_case_definition(args.case)
    mesh = MeshConfig(
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
    system = build_case_system(args.case, mesh)
    operators = prepare_constrained_operators(system)
    if args.frequency_min is None:
        labeled, physics = labeled_and_physics_frequencies(definition, args.n_labeled, args.n_physics)
    else:
        labeled = np.linspace(float(args.frequency_min), float(args.frequency_max), int(args.n_labeled))
        physics = np.unique(
            np.r_[
                labeled,
                np.linspace(float(args.frequency_min), float(args.frequency_max), int(args.n_physics)),
            ]
        )
    tests = np.asarray(definition.test_frequencies_hz, dtype=float)
    reference = np.asarray([operators.solve(float(f), args.loss_factor) for f in tests])

    shared_modal_interface = None
    modal_config = _modal_config(args).validated()
    if modal_config.enabled:
        combined_modal_frequencies = np.concatenate((labeled, physics))
        modal_config = modal_config.with_frequency_bounds(
            float(np.min(combined_modal_frequencies)),
            float(np.max(combined_modal_frequencies)),
        )
        shared_modal_interface = build_modal_resonance_interface(
            operators,
            modal_config,
            loss_factor=args.loss_factor,
        )

    started = time.perf_counter()
    full_results = {}
    pod_results = {}
    predictions: dict[str, np.ndarray] = {}
    timings: dict[str, dict[str, float | int | str | bool]] = {}
    method_specs: dict[str, dict[str, str | bool]] = {}

    # Four independently trained offline models.  Their refined counterparts
    # reuse the corresponding trained model but run a separate online solve.
    for method, weight, objective in (
        ("full_data_only", 0.0, "data_only"),
        ("full_physics", args.physics_weight, "physics_constrained"),
    ):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        train_started = time.perf_counter()
        full_results[method] = train_full_order_frequency_net(
            operators,
            labeled,
            physics,
            config=_config(args, float(weight)),
            loss_factor=args.loss_factor,
            modal_resonance=shared_modal_interface,
        )
        train_s = time.perf_counter() - train_started
        prediction_started = time.perf_counter()
        predictions[method] = predict_full_order_response(full_results[method], tests)
        prediction_s = time.perf_counter() - prediction_started
        timings[method] = {
            "offline_training_s": train_s,
            "prediction_s": prediction_s,
            "refinement_s": 0.0,
            "end_to_end_s": train_s + prediction_s,
            "shared_offline_model": method,
        }
        method_specs[method] = {
            "representation": "full_order",
            "training_objective": objective,
            "online_correction": False,
        }
        _write_rows(output_dir / f"training_history_{method}.csv", full_results[method].history)

    modal_interface = shared_modal_interface
    if modal_interface is not None:
        np.save(output_dir / "modal_natural_frequencies_hz.npy", modal_interface.natural_frequencies_hz)
        np.save(output_dir / "modal_resonance_frequencies_hz.npy", modal_interface.resonance_frequencies_hz)
        np.save(output_dir / "labeled_training_mask.npy", full_results["full_physics"].labeled_training_mask)
        np.save(output_dir / "physics_training_mask.npy", full_results["full_physics"].physics_training_mask)

    refinement_rows = []
    if args.refine_steps > 0:
        refine_config = _refinement_config(args, args.refine_steps, args.refine_lr)
        for base_method in ("full_data_only", "full_physics"):
            method = f"{base_method}_refined"
            refinement_started = time.perf_counter()
            refined_predictions = []
            for frequency, initial in zip(tests, predictions[base_method]):
                refined = refine_full_order_response(
                    operators,
                    float(frequency),
                    initial,
                    config=refine_config,
                    loss_factor=args.loss_factor,
                )
                refined_predictions.append(refined.response)
                refinement_rows.extend(_tag_history(args.case, method, float(frequency), refined.history))
            refinement_s = time.perf_counter() - refinement_started
            predictions[method] = np.asarray(refined_predictions, dtype=complex)
            timings[method] = {
                **timings[base_method],
                "refinement_s": refinement_s,
                "end_to_end_s": float(timings[base_method]["end_to_end_s"]) + refinement_s,
                "shared_offline_model": base_method,
            }
            method_specs[method] = {
                "representation": "full_order",
                "training_objective": str(method_specs[base_method]["training_objective"]),
                "online_correction": True,
            }

    pod_refine_steps = args.refine_steps if args.pod_refine_steps is None else int(args.pod_refine_steps)
    if args.compare_pod:
        for method, weight, objective in (
            ("pod_data_only", 0.0, "data_only"),
            ("pod_physics", args.physics_weight, "physics_constrained"),
        ):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            train_started = time.perf_counter()
            pod_results[method] = train_pod_frequency_net(
                operators,
                labeled,
                physics,
                config=_pod_config(args, float(weight)),
                n_modes=args.pod_modes,
                loss_factor=args.loss_factor,
                modal_resonance=shared_modal_interface,
            )
            train_s = time.perf_counter() - train_started
            prediction_started = time.perf_counter()
            predictions[method] = predict_pod_response(pod_results[method], tests)
            prediction_s = time.perf_counter() - prediction_started
            timings[method] = {
                "offline_training_s": train_s,
                "prediction_s": prediction_s,
                "refinement_s": 0.0,
                "end_to_end_s": train_s + prediction_s,
                "shared_offline_model": method,
            }
            method_specs[method] = {
                "representation": (
                    "pod_plus_modal_rational"
                    if pod_results[method].modal_resonance is not None
                    else "pod"
                ),
                "training_objective": objective,
                "online_correction": False,
            }
            _write_rows(output_dir / f"training_history_{method}.csv", pod_results[method].history)
        pod_modal_interface = pod_results["pod_physics"].modal_resonance
        if pod_modal_interface is not None:
            np.save(output_dir / "pod_modal_natural_frequencies_hz.npy", pod_modal_interface.natural_frequencies_hz)
            np.save(output_dir / "pod_seed_training_mask.npy", pod_results["pod_physics"].seed_training_mask)
            np.save(output_dir / "pod_physics_training_mask.npy", pod_results["pod_physics"].physics_training_mask)
        if pod_refine_steps > 0:
            pod_refine_config = _refinement_config(
                args,
                pod_refine_steps,
                args.refine_lr if args.pod_refine_lr is None else float(args.pod_refine_lr),
            )
            for base_method in ("pod_data_only", "pod_physics"):
                method = f"{base_method}_refined"
                refinement_started = time.perf_counter()
                pod_alpha = predict_pod_coordinates(pod_results[base_method], tests)
                pod_refined_predictions = []
                for frequency, alpha0 in zip(tests, pod_alpha):
                    refined = refine_pod_response(
                        operators,
                        float(frequency),
                        pod_results[base_method],
                        alpha0,
                        config=pod_refine_config,
                        loss_factor=args.loss_factor,
                    )
                    pod_refined_predictions.append(refined.response)
                    refinement_rows.extend(_tag_history(args.case, method, float(frequency), refined.history))
                refinement_s = time.perf_counter() - refinement_started
                predictions[method] = np.asarray(pod_refined_predictions, dtype=complex)
                timings[method] = {
                    **timings[base_method],
                    "refinement_s": refinement_s,
                    "end_to_end_s": float(timings[base_method]["end_to_end_s"]) + refinement_s,
                    "shared_offline_model": base_method,
                }
                method_specs[method] = {
                    "representation": str(method_specs[base_method]["representation"]),
                    "training_objective": str(method_specs[base_method]["training_objective"]),
                    "online_correction": True,
                }
    _write_rows(output_dir / "refinement_history.csv", refinement_rows)

    rows = []
    for method, prediction_batch in predictions.items():
        for frequency, prediction, direct in zip(tests, prediction_batch, reference):
            rows.append(
                {
                    "case": args.case,
                    "frequency_hz": float(frequency),
                    "method": method,
                    "relative_field_error": relative_field_error(prediction, direct),
                    "relative_physics_residual": relative_physics_residual(
                        operators,
                        float(frequency),
                        prediction,
                        args.loss_factor,
                    ),
                    "constraint_metric": constraint_metric(operators, prediction),
                }
            )
    _write_rows(output_dir / "validation_summary.csv", rows)
    configuration_rows = []
    for method, spec in method_specs.items():
        selected = [row for row in rows if row["method"] == method]
        base_method = str(timings[method]["shared_offline_model"])
        if spec["representation"] == "full_order":
            network_parameters = _trainable_parameter_count(full_results[base_method].network)
            pod_state_scalars = 0
            online_variables = 2 * operators.n_free if bool(spec["online_correction"]) else 0
            configured_weight = 0.0 if spec["training_objective"] == "data_only" else float(args.physics_weight)
            effective_weight = configured_weight
            initial_data_gradient_norm = None
            initial_physics_gradient_norm = None
        else:
            pod_result = pod_results[base_method]
            network_parameters = _trainable_parameter_count(pod_result.network)
            pod_state_scalars = int(
                pod_result.basis.basis.size + pod_result.basis.mean.size + pod_result.basis.alpha_scale.size
            )
            online_variables = int(pod_result.basis.n_modes) if bool(spec["online_correction"]) else 0
            configured_weight = float(pod_result.config.physics_weight)
            effective_weight = float(pod_result.effective_physics_weight)
            initial_data_gradient_norm = pod_result.initial_data_gradient_norm
            initial_physics_gradient_norm = pod_result.initial_physics_gradient_norm
        configuration_rows.append(
            _configuration_row(
                case=args.case,
                method=method,
                representation=str(spec["representation"]),
                training_objective=str(spec["training_objective"]),
                online_correction=bool(spec["online_correction"]),
                validation_rows=selected,
                timing=timings[method],
                network_trainable_parameters=network_parameters,
                frozen_pod_state_scalars=pod_state_scalars,
                online_refinement_variables=online_variables,
                configured_physics_weight=configured_weight,
                effective_physics_weight=effective_weight,
                initial_data_gradient_norm=initial_data_gradient_norm,
                initial_physics_gradient_norm=initial_physics_gradient_norm,
            )
        )
    _write_rows(output_dir / "configuration_summary.csv", configuration_rows)
    _write_rows(output_dir / "training_history.csv", full_results["full_physics"].history)

    np.save(output_dir / "frequency.npy", tests)
    np.save(output_dir / "train_frequency.npy", labeled)
    np.save(output_dir / "test_frequency.npy", tests)
    np.save(output_dir / "unlabeled_frequency.npy", np.setdiff1d(physics, labeled))
    np.save(output_dir / "reference_response_real.npy", reference.real)
    np.save(output_dir / "reference_response_imag.npy", reference.imag)
    for method, prediction in predictions.items():
        np.save(output_dir / f"prediction_{method}_real.npy", prediction.real)
        np.save(output_dir / f"prediction_{method}_imag.npy", prediction.imag)
    observation_index = int(np.argmax(np.max(np.abs(reference), axis=0)))
    observation_columns = [
        tests,
        reference[:, observation_index].real,
        reference[:, observation_index].imag,
    ]
    for prediction in predictions.values():
        observation_columns.extend((prediction[:, observation_index].real, prediction[:, observation_index].imag))
    observation = np.column_stack(observation_columns)
    np.save(output_dir / "observation_frf.npy", observation)
    observation_rows = []
    for i, frequency in enumerate(tests):
        row = {
            "case": args.case,
            "frequency_hz": float(frequency),
            "reference_real": float(reference[i, observation_index].real),
            "reference_imag": float(reference[i, observation_index].imag),
        }
        for method, prediction in predictions.items():
            row[f"{method}_real"] = float(prediction[i, observation_index].real)
            row[f"{method}_imag"] = float(prediction[i, observation_index].imag)
        observation_rows.append(row)
    _write_rows(output_dir / "observation_frf.csv", observation_rows)
    physics_rows = [row for row in rows if row["method"] == "full_physics"]
    np.save(output_dir / "relative_field_error.npy", np.asarray([row["relative_field_error"] for row in physics_rows]))
    np.save(output_dir / "physics_residual.npy", np.asarray([row["relative_physics_residual"] for row in physics_rows]))
    frequency_sweep = _write_frequency_sweep(
        args=args,
        definition=definition,
        operators=operators,
        full_results=full_results,
        pod_results=pod_results,
        pod_refine_steps=pod_refine_steps,
        output_dir=output_dir,
    )

    apply_journal_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    plot_styles = {
        "full_data_only": (COLOR_DATA_ONLY, "o", "Data-only full-order"),
        "full_data_only_refined": ("#C0504D", "^", "Data-only full-order + refinement"),
        "full_physics": (COLOR_PROPOSED, "s", "Physics-constrained full-order"),
        "full_physics_refined": ("#7F0000", "P", "Physics full-order + refinement"),
        "pod_data_only": ("#8064A2", "D", "Data-only POD"),
        "pod_data_only_refined": ("#B1A0C7", "v", "Data-only POD + refinement"),
        "pod_physics": ("#00A651", "X", "Physics-constrained POD"),
        "pod_physics_refined": ("#008C95", "<", "Physics POD + refinement"),
    }
    for method in [key for key in plot_styles if key in predictions]:
        color, marker, label = plot_styles[method]
        selected = [row for row in rows if row["method"] == method]
        frequencies = [row["frequency_hz"] for row in selected]
        axes[0].semilogy(
            frequencies,
            [row["relative_field_error"] for row in selected],
            color=color,
            marker=marker,
            label=label,
        )
        axes[1].semilogy(
            frequencies,
            [row["relative_physics_residual"] for row in selected],
            color=color,
            marker=marker,
            label=label,
        )
    axes[0].set_ylabel("relative full-field error")
    axes[1].set_ylabel("relative RBF-FD residual")
    for axis in axes:
        axis.set_xlabel("frequency (Hz)")
        axis.grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    save_figure(fig, output_dir / "validation_error_and_residual.png")
    plt.close(fig)

    metadata = {
        "method": "full_order_RBF_FD_operator_embedded_frequency_network",
        "case": args.case,
        "labeled_frequencies_hz": labeled.tolist(),
        "physics_frequencies_hz": physics.tolist(),
        "test_frequencies_hz": tests.tolist(),
        "training_frequency_interval_hz": [
            float(args.frequency_min if args.frequency_min is not None else definition.frequency_min_hz),
            float(args.frequency_max if args.frequency_max is not None else definition.frequency_max_hz),
        ],
        "n_full_physical_dofs": operators.full_dimension,
        "n_constrained_dofs": operators.n_free,
        "physical_node_count": int(system.nodes.shape[0]),
        "mesh": {
            "square_ndiv": int(args.square_ndiv),
            "square_node_count_per_axis": int(args.square_ndiv) + 1,
            "hollow_sweep_points": int(args.hollow_sweep_points),
            "hollow_tangent_points_per_patch": int(args.hollow_tangent_points_per_patch),
            "hollow_circumference_intervals": int(4 * (args.hollow_tangent_points_per_patch - 1)),
            "hole_diameter_ratio": float(args.hole_diameter_ratio),
            "annular_nrad": int(args.annular_nrad),
            "annular_ntheta": int(args.annular_ntheta),
            "annular_node_count": int(args.annular_nrad * args.annular_ntheta),
            "annular_stencil": int(args.annular_stencil),
            "annular_outer_diameter": float(args.annular_outer_diameter),
            "annular_hole_diameter_ratio": float(args.annular_hole_diameter_ratio),
            "plate_stencil": int(args.plate_stencil),
            "plate_poly_degree": int(args.plate_poly_degree),
            "beam_stencil": int(args.beam_stencil),
            "hollow_stencil": int(args.hollow_stencil),
            "hollow_poly_degree": int(args.hollow_poly_degree),
            "rbf_condition_limit": float(args.rbf_condition_limit),
        },
        "network_output_dimension": 2 * operators.n_free,
        "constraint_mode": operators.reduction_mode,
        "system_form": system.metadata["form"],
        "spatial_autograd": False,
        "device": args.device,
        "data_reduction_basis": (
            "real_POD_background_plus_shared_modal_rational_branch"
            if args.modal_resonance and args.compare_pod
            else "real_POD_for_POD_configurations_only"
        ),
        "comparison_methods": list(predictions),
        "experiment_matrix": {
            "offline_training_objectives": ["data_only", "physics_constrained"],
            "response_representations": (
                ["full_order_plus_modal_rational", "pod_plus_modal_rational"]
                if args.modal_resonance
                else ["full_order", "pod"]
            ),
            "online_residual_correction": [False, True],
            "configuration_count": len(configuration_rows),
            "configuration_summary": "configuration_summary.csv",
            "timing_definition": (
                "end_to_end_s equals offline model training plus prediction at all test frequencies "
                "plus online refinement when enabled; refined and unrefined rows share the same "
                "offline model but report their own equivalent end-to-end cost."
            ),
        },
        "full_order_refinement": {
            "enabled": bool(args.refine_steps > 0),
            "steps": int(args.refine_steps),
            "learning_rate": float(args.refine_lr),
            "prior_weight": float(args.refine_prior_weight),
            "weight_decay": float(args.refine_weight_decay),
        },
        "pod_comparison": {
            "enabled": bool(args.compare_pod),
            "pod_modes": None if not pod_results else int(next(iter(pod_results.values())).basis.n_modes),
            "requested_pod_modes": int(args.pod_modes),
            "refine_steps": int(pod_refine_steps),
            "network_output_dimension": None if not pod_results else int(next(iter(pod_results.values())).basis.n_modes),
        },
        "modal_resonance": (
            {"enabled": False}
            if modal_interface is None
            else modal_interface.summary()
        ),
        "pod_initial_gradient_balance": {
            "enabled": bool(args.pod_gradient_balance),
            "target_physics_to_data_gradient_ratio": float(args.pod_gradient_balance_ratio),
            "models": {
                method: {
                    "configured_physics_weight": float(result.config.physics_weight),
                    "effective_physics_weight": float(result.effective_physics_weight),
                    "initial_data_gradient_norm": result.initial_data_gradient_norm,
                    "initial_physics_gradient_norm": result.initial_physics_gradient_norm,
                }
                for method, result in pod_results.items()
            },
        },
        "frequency_sweep": frequency_sweep,
        "elapsed_s": time.perf_counter() - started,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

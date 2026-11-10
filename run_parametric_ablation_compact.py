# -*- coding: utf-8 -*-
"""Controlled ablation of the modal-rational branch and the RBF-FD physics term.

Four variants are trained on three fixed topologies (flat, stiffened, hollow):

    direct_data     : no modal-rational branch, physics weight = 0
    direct_physics  : no modal-rational branch, physics weight > 0
    pole_data       : modal-rational branch, physics weight = 0
    full            : modal-rational branch, physics weight > 0

All variants share the same seed, parameter samples, labeled/physics frequency
sets, network architecture and training budget.  Evaluation is performed on
complete unseen parameter realizations over the full analysis band of the case,
which is read from the deployed metadata of that case (0--200 Hz for the
unstiffened plate, 0--600 Hz for the stiffened panel):

    - full-domain relative MSE (all constrained complex DOFs, all frequencies);
    - dominant-peak frequency error of the observation FRF (grid-resolved and
      parabolic-interpolated);
    - dominant-peak amplitude relative error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
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
    build_parametric_frequency_family,
    load_parametric_operator,
    predict_parametric_members,
    prepare_constrained_operators,
    save_parametric_operator,
)
from rbf_fd_full_order.examples.cases import MeshConfig, build_case_system
from rbf_fd_full_order.training.parametric_operator import (
    ParametricOperatorResult,
    _frequency_bounds,
    _train_one_member,
    build_parametric_training_dataset,
)
from rbf_fd_full_order.visualization import apply_journal_style, save_figure


VARIANTS = ("direct_data", "direct_physics", "pole_data", "full")
CASE_LABELS = {
    "flat": "Solid plate",
    "stiffened": "Stiffened plate",
    "hollow": "Perforated plate",
}
# Deployed checkpoints whose metadata carries the analysis band of each case.
CASE_METADATA = {
    "flat": Path("results") / "parametric_many_query_flat_5000_best" / "flat" / "metadata.json",
    "stiffened": Path("results") / "parametric_many_query_stiffened_5000_best" / "stiffened" / "metadata.json",
}


def case_band(case: str, override: float | None = None) -> float:
    """Upper end of the analysis band, read from the deployed metadata."""
    if override is not None:
        return float(override)
    path = CASE_METADATA.get(case)
    if path is None or not path.is_file():
        return 200.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    interval = payload.get("frequency_interval_hz")
    if not interval:
        return 200.0
    return float(interval[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("flat", "hollow", "stiffened", "all"), default="all")
    parser.add_argument("--out", type=Path, default=Path("results") / "parametric_ablation_compact")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--n-train-parameters", type=int, default=20)
    parser.add_argument("--n-test-parameters", type=int, default=8)
    parser.add_argument("--n-labeled", type=int, default=24)
    parser.add_argument("--n-physics", type=int, default=102)
    parser.add_argument("--sweep-points", type=int, default=0,
                        help="evaluation grid size; 0 means a 1 Hz grid of the case band")
    parser.add_argument("--frequency-max-hz", type=float, default=None,
                        help="override the analysis band read from the case metadata")
    parser.add_argument("--physics-weight", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--fourier-modes", type=int, default=4)
    parser.add_argument("--pod-modes", type=int, default=8)
    parser.add_argument("--modal-modes", type=int, default=8)
    parser.add_argument("--modal-relative-half-width", type=float, default=0.04)
    parser.add_argument("--modal-min-half-width-hz", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--workers", type=int, default=8,
                        help="threads used to generate the shared test reference")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def _bounds() -> ParameterBounds:
    return ParameterBounds(
        lower=np.asarray([0.85, 0.90, 0.80, 0.005], dtype=float),
        upper=np.asarray([1.15, 1.10, 1.20, 0.020], dtype=float),
    ).validated()


def _mesh() -> MeshConfig:
    return MeshConfig()


def _modal_config(enabled: bool, args: argparse.Namespace) -> ModalResonanceConfig:
    return ModalResonanceConfig(
        enabled=enabled,
        n_modes=int(args.modal_modes),
        relative_half_width=float(args.modal_relative_half_width),
        minimum_half_width_hz=float(args.modal_min_half_width_hz),
        transition_fraction=0.2,
        exclusion_threshold=0.5,
        exclude_labeled=True,
        exclude_physics=True,
        minimum_participation_ratio=1.0e-6,
    )


def _operator_config(
    args: argparse.Namespace,
    *,
    modal_enabled: bool,
    physics_weight: float,
) -> ParametricOperatorConfig:
    return ParametricOperatorConfig(
        representation="pod",
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        width=int(args.width),
        depth=int(args.depth),
        fourier_modes=int(args.fourier_modes),
        pod_modes=int(args.pod_modes),
        ensemble_size=int(args.ensemble_size),
        physics_weight=float(physics_weight),
        warmup_epochs=max(int(args.epochs) // 4, 1),
        log_every=max(int(args.epochs) // 20, 1),
        device=str(args.device),
        modal_resonance=_modal_config(modal_enabled, args),
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


def _parabolic_peak(frequencies: np.ndarray, curve: np.ndarray) -> float:
    """Sub-grid parabolic peak estimate on log10 amplitudes."""
    index = int(np.argmax(curve))
    if index <= 0 or index >= curve.size - 1:
        return float(frequencies[index])
    y = np.log10(np.maximum(curve[index - 1 : index + 2], 1.0e-30))
    denominator = y[0] - 2.0 * y[1] + y[2]
    if abs(denominator) < 1.0e-14:
        return float(frequencies[index])
    step = float(np.median(np.diff(frequencies)))
    offset = 0.5 * (y[0] - y[2]) / denominator
    return float(frequencies[index] + offset * step)


def _metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    frequencies: np.ndarray,
    observation_index: int,
) -> dict[str, float]:
    """Compute full-domain MSE and dominant-peak metrics.

    ``predicted`` and ``reference`` have shape (n_mu, n_f, n_dof).
    """
    epsilon = 1.0e-30
    squared_relative = (
        np.sum(np.abs(predicted - reference) ** 2, axis=-1)
        / (np.sum(np.abs(reference) ** 2, axis=-1) + epsilon)
    )
    full_domain_mse = float(np.mean(squared_relative))
    curve_errors = [
        _relative_l2(predicted[index], reference[index]) for index in range(predicted.shape[0])
    ]
    ref_obs = np.abs(reference[..., observation_index]) * 1.0e3
    pred_obs = np.abs(predicted[..., observation_index]) * 1.0e3
    grid_errors = []
    interp_errors = []
    amplitude_errors = []
    for index in range(predicted.shape[0]):
        ref_grid = float(frequencies[np.argmax(ref_obs[index])])
        pred_grid = float(frequencies[np.argmax(pred_obs[index])])
        grid_errors.append(abs(pred_grid - ref_grid))
        ref_peak = _parabolic_peak(frequencies, ref_obs[index])
        pred_peak = _parabolic_peak(frequencies, pred_obs[index])
        interp_errors.append(abs(pred_peak - ref_peak))
        ref_amplitude = float(np.max(ref_obs[index]))
        pred_amplitude = float(np.max(pred_obs[index]))
        amplitude_errors.append(abs(pred_amplitude - ref_amplitude) / max(ref_amplitude, 1.0e-30))
    return {
        "full_domain_relative_mse": full_domain_mse,
        "curve_relative_l2_mean": float(np.mean(curve_errors)),
        "curve_relative_l2_max": float(np.max(curve_errors)),
        "peak_frequency_error_grid_mean_hz": float(np.mean(grid_errors)),
        "peak_frequency_error_grid_max_hz": float(np.max(grid_errors)),
        "peak_frequency_error_interp_mean_hz": float(np.mean(interp_errors)),
        "peak_frequency_error_interp_max_hz": float(np.max(interp_errors)),
        "peak_amplitude_relative_error_mean": float(np.mean(amplitude_errors)),
        "peak_amplitude_relative_error_max": float(np.max(amplitude_errors)),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_case(case: str, args: argparse.Namespace) -> list[dict[str, object]]:
    output_dir = args.out / case
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = _bounds()
    mesh = _mesh()
    fmax = case_band(case, args.frequency_max_hz)
    sweep_points = int(args.sweep_points) if int(args.sweep_points) > 0 else int(round(fmax)) + 1
    print(f"[{case}] band 0-{fmax:.0f} Hz, {sweep_points} evaluation points", flush=True)
    print(f"[{case}] building operators and families", flush=True)
    system = build_case_system(case, mesh)
    operators = prepare_constrained_operators(system)
    family_direct = build_parametric_frequency_family(
        operators,
        bounds=bounds,
        modal_config=None,
        frequency_min_hz=0.0,
        frequency_max_hz=fmax,
    )
    family_pole = build_parametric_frequency_family(
        operators,
        bounds=bounds,
        modal_config=_modal_config(True, args),
        frequency_min_hz=0.0,
        frequency_max_hz=fmax,
    )

    train_mu = bounds.sample_latin_hypercube(int(args.n_train_parameters), int(args.seed) + 101)
    test_mu = bounds.sample_latin_hypercube(int(args.n_test_parameters), int(args.seed) + 303)
    labeled = np.linspace(0.0, fmax, int(args.n_labeled))
    physics = np.unique(np.r_[labeled, np.linspace(0.0, fmax, int(args.n_physics))])
    sweep = np.linspace(0.0, fmax, sweep_points)

    print(f"[{case}] generating shared high-fidelity test reference", flush=True)
    # Each curve is an independent sparse solve and the scipy solve releases the
    # GIL, so the test curves and the nominal curve are generated in a thread
    # pool instead of one after another.
    stack = np.vstack([test_mu, bounds.center])
    with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        curves = list(pool.map(
            lambda mu: np.asarray(family_pole.solve_grid(sweep, mu[None, :])[0], dtype=complex),
            stack,
        ))
    reference = np.stack(curves[:-1])
    nominal = curves[-1]
    print(f"[{case}] test reference {reference.shape} from {args.workers} workers", flush=True)
    observation_index = _transverse_observation_index(
        operators,
        system.nodes.shape[0],
        nominal,
    )
    print(f"[{case}] observation constrained index = {observation_index}", flush=True)

    print(f"[{case}] building shared training datasets", flush=True)
    dataset_direct = build_parametric_training_dataset(
        family_direct,
        train_mu,
        labeled,
        physics,
        representation="pod",
        pod_modes=int(args.pod_modes),
    )
    dataset_pole = build_parametric_training_dataset(
        family_pole,
        train_mu,
        labeled,
        physics,
        representation="pod",
        pod_modes=int(args.pod_modes),
    )

    rows = []
    for variant in VARIANTS:
        modal_enabled = variant in ("pole_data", "full")
        physics_weight = float(args.physics_weight if variant in ("direct_physics", "full") else 0.0)
        family = family_pole if modal_enabled else family_direct
        dataset = dataset_pole if modal_enabled else dataset_direct
        config = _operator_config(args, modal_enabled=modal_enabled, physics_weight=physics_weight)
        frequency_min, frequency_range = _frequency_bounds(labeled, physics)
        checkpoint = output_dir / f"{variant}.pt"
        print(
            f"[{case}] training {variant} "
            f"(modal={modal_enabled}, physics={physics_weight}, "
            f"epochs={args.epochs}, members={args.ensemble_size})",
            flush=True,
        )
        started = time.perf_counter()
        members = [
            _train_one_member(
                family,
                dataset,
                config,
                frequency_min,
                frequency_range,
                seed=int(args.seed) + 1009 * index,
            )
            for index in range(int(args.ensemble_size))
        ]
        result = ParametricOperatorResult(
            family=family,
            members=members,
            config=config,
            bounds=bounds,
            frequency_min=frequency_min,
            frequency_range=frequency_range,
            n_dofs=family.n_free,
            response_scaler=dataset.response_scaler,
            coefficient_scale=dataset.coefficient_scale,
            pod_basis=dataset.pod_basis,
            training_parameters=np.asarray(train_mu, dtype=float),
            labeled_frequencies_hz=labeled,
            physics_frequencies_hz=physics,
            high_fidelity_generation_s=dataset.high_fidelity_generation_s,
            high_fidelity_solve_count=dataset.high_fidelity_solve_count,
            ensemble_training_s=float(sum(member.training_s for member in members)),
        )
        save_parametric_operator(result, checkpoint)
        training_s = time.perf_counter() - started

        print(f"[{case}] evaluating {variant}", flush=True)
        predicted = np.mean(predict_parametric_members(result, sweep, test_mu), axis=0)
        metrics = _metrics(predicted, reference, sweep, observation_index)
        row = {
            "case": case,
            "case_label": CASE_LABELS[case],
            "variant": variant,
            "modal_enabled": modal_enabled,
            "physics_weight": physics_weight,
            "epochs": int(args.epochs),
            "ensemble_size": int(args.ensemble_size),
            "training_s": round(float(training_s), 2),
            "best_epoch_per_member": [int(member.best_epoch) for member in members],
            "best_total_loss_per_member": [float(member.best_total_loss) for member in members],
            **metrics,
        }
        rows.append(row)
        _write_json(output_dir / f"{variant}_metrics.json", row)
        print(
            f"[{case}] {variant}: full-domain MSE = {metrics['full_domain_relative_mse']:.6e}, "
            f"interp peak-freq err = {metrics['peak_frequency_error_interp_mean_hz']:.3f} Hz "
            f"(max {metrics['peak_frequency_error_interp_max_hz']:.3f} Hz), "
            f"peak-amp err = {100 * metrics['peak_amplitude_relative_error_mean']:.3f}% "
            f"[{training_s:.0f} s]",
            flush=True,
        )

    with (output_dir / "ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "case": case,
                "observation_index": int(observation_index),
                "frequency_interval_hz": [0.0, fmax],
                "sweep_points": sweep_points,
                "rows": rows,
            },
            handle,
            indent=2,
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict[str, object]], out: Path) -> None:
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    colors = {"direct_data": "#B0B0B0", "direct_physics": "#4DBEEE", "pole_data": "#EDB120", "full": "#D62728"}
    apply_journal_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), constrained_layout=True)
    width = 0.20
    for case_index, case in enumerate(cases):
        case_rows = [row for row in rows if row["case"] == case]
        for variant_index, variant in enumerate(VARIANTS):
            metrics = next(row for row in case_rows if row["variant"] == variant)
            x = case_index + (variant_index - 1.5) * width
            axes[0].bar(
                x,
                float(metrics["full_domain_relative_mse"]),
                width,
                color=colors[variant],
            )
            axes[1].bar(
                x,
                float(metrics["peak_frequency_error_interp_mean_hz"]),
                width,
                color=colors[variant],
            )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Full-domain relative MSE")
    axes[0].set_xticks(range(len(cases)))
    axes[0].set_xticklabels([CASE_LABELS[case] for case in cases], rotation=12)
    axes[0].grid(True, axis="y", which="both", alpha=0.3)
    axes[1].set_ylabel("Peak-frequency error (Hz, interpolated)")
    axes[1].set_xticks(range(len(cases)))
    axes[1].set_xticklabels([CASE_LABELS[case] for case in cases], rotation=12)
    axes[1].grid(True, axis="y", which="both", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[variant])
        for variant in VARIANTS
    ]
    figure.legend(
        handles,
        ["Direct data", "Direct physics", "Pole + data", "Full"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=False,
    )
    save_figure(figure, out / "Fig10_ablation_compact.png")
    save_figure(figure, out / "Fig10_ablation_compact.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = ("flat", "hollow", "stiffened") if args.case == "all" else (args.case,)
    rows: list[dict[str, object]] = []
    for case in cases:
        try:
            rows.extend(run_case(case, args))
        except Exception as exc:  # keep going across cases
            print(f"[{case}] FAILED: {exc}", file=sys.stderr, flush=True)
    if rows:
        _write_csv(args.out / "ablation_summary.csv", rows)
        with (args.out / "ablation_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        make_figure(rows, args.out)
        print(f"Wrote ablation summary to {args.out}", flush=True)


if __name__ == "__main__":
    main()

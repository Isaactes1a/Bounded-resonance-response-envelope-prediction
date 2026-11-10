# -*- coding: utf-8 -*-
"""Generate comparable multi-coverage FRF envelopes from retained operators.

The script performs observation-only reconstruction from the POD coordinates
and the analytic modal branch.  It therefore avoids allocating the complete
``ensemble x parameter x frequency x DOF`` response tensor.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rbf_fd_full_order import (
    ParameterBounds,
    build_parametric_frequency_family,
    load_parametric_operator,
    prepare_constrained_operators,
)
from rbf_fd_full_order.examples.cases import MeshConfig, build_case_system
from rbf_fd_full_order.visualization import apply_journal_style, save_figure, style_axis


CASE_ORDER = ("flat", "hollow", "stiffened")
CASE_LABELS = {
    "flat": "Flat plate",
    "hollow": "Plate with circular cutout",
    "stiffened": "Stiffened plate",
}
CHECKPOINT_DIRS = {
    "flat": "parametric_many_query_flat_5000_best",
    "hollow": "parametric_many_query_hollow_5000_best",
    "stiffened": "parametric_many_query_stiffened_5000_best",
}
DEFAULT_OBSERVATION_INDICES = {"flat": 420, "hollow": 224, "stiffened": 420}
COLORS = {0.80: "#08519c", 0.90: "#3182bd", 0.95: "#6baed6", 0.99: "#c6dbef"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "three_plate_uncertainty_envelopes",
    )
    parser.add_argument("--frequency-min", type=float, default=0.0)
    parser.add_argument("--frequency-max", type=float, default=200.0)
    parser.add_argument("--frequency-points", type=int, default=401)
    parser.add_argument("--uq-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--coverages", type=float, nargs="+", default=(0.80, 0.90, 0.95, 0.99))
    parser.add_argument("--batch-parameters", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable.")
    if args.frequency_points < 2 or args.frequency_max <= args.frequency_min:
        parser.error("The frequency interval must be increasing and contain at least two points.")
    if args.uq_samples < 10:
        parser.error("At least ten UQ samples are required.")
    coverages = sorted(set(float(value) for value in args.coverages))
    if any(not 0.0 < value < 1.0 for value in coverages):
        parser.error("Every coverage must lie strictly between zero and one.")
    args.coverages = tuple(coverages)
    return args


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    values = list(rows)
    if not values:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def _observation_index(case: str, checkpoint: Path) -> int:
    metadata = checkpoint.parent / "metadata.json"
    if metadata.exists():
        return int(json.loads(metadata.read_text(encoding="utf-8"))["observation_constrained_index"])
    evaluation = checkpoint.parent / "frequency_range_evaluation.json"
    if evaluation.exists():
        return int(json.loads(evaluation.read_text(encoding="utf-8"))["observation_constrained_index"])
    return int(DEFAULT_OBSERVATION_INDICES[case])


def _observation_location(operators, system, observation_index: int) -> tuple[int, float, float]:
    full_dof = int(observation_index)
    if operators.reduction_mode == "fixed_dofs" and operators.free_dofs is not None:
        full_dof = int(np.asarray(operators.free_dofs)[observation_index])
    node_index = int(full_dof)
    if 0 <= node_index < int(system.nodes.shape[0]):
        return node_index, float(system.nodes[node_index, 0]), float(system.nodes[node_index, 1])
    return node_index, float("nan"), float("nan")


def _predict_observation_members(
    result,
    frequencies_hz: np.ndarray,
    parameters: np.ndarray,
    observation_index: int,
    *,
    batch_parameters: int,
) -> np.ndarray:
    """Return |w| in mm with shape (ensemble, n_parameter, n_frequency)."""

    if result.representation != "pod" or result.pod_basis is None:
        raise ValueError("The retained three-plate checkpoints are expected to use POD representation.")
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    mu = np.asarray(parameters, dtype=float)
    result.family.validate_parameters(mu)
    n_mu = int(mu.shape[0])
    n_f = int(frequencies.size)
    n_dof = int(result.n_dofs)
    basis = result.pod_basis
    real_mean = float(basis.mean[observation_index])
    imag_mean = float(basis.mean[n_dof + observation_index])
    real_basis = np.asarray(basis.basis[observation_index], dtype=float)
    imag_basis = np.asarray(basis.basis[n_dof + observation_index], dtype=float)
    alpha_scale = np.asarray(basis.alpha_scale, dtype=float)
    modal = result.family.modal
    output = np.empty((len(result.members), n_mu, n_f), dtype=float)

    for start in range(0, n_mu, int(batch_parameters)):
        stop = min(start + int(batch_parameters), n_mu)
        chunk = mu[start:stop]
        paired_frequency = np.tile(frequencies, chunk.shape[0])
        paired_mu = np.repeat(chunk, n_f, axis=0)
        normalized_frequency = result.normalize_frequency(paired_frequency)
        normalized_mu = result.normalize_parameters(paired_mu)

        if modal is None:
            modal_observation = np.zeros(paired_frequency.size, dtype=complex)
        else:
            reference = modal.reference
            omega_sq = (2.0 * np.pi * paired_frequency)[:, None] ** 2
            young = paired_mu[:, 0:1]
            density = paired_mu[:, 1:2]
            pressure = paired_mu[:, 2:3]
            loss_factor = paired_mu[:, 3:4]
            denominator = (
                (1.0 + 1.0j * loss_factor) * young * reference.eigenvalues[None, :]
                - density * omega_sq
            )
            modal_coefficients = pressure * reference.modal_forces[None, :] / denominator
            modal_observation = modal_coefficients @ reference.mode_shapes[observation_index, :]

        for member_index, member in enumerate(result.members):
            network_parameter = next(member.network.parameters())
            with torch.no_grad():
                raw = member.network(
                    torch.as_tensor(
                        normalized_frequency,
                        dtype=network_parameter.dtype,
                        device=network_parameter.device,
                    ),
                    torch.as_tensor(
                        normalized_mu,
                        dtype=network_parameter.dtype,
                        device=network_parameter.device,
                    ),
                ).cpu().numpy()
            coordinates = raw * alpha_scale[None, :]
            background = (
                real_mean
                + coordinates @ real_basis
                + 1.0j * (imag_mean + coordinates @ imag_basis)
            ) * float(result.coefficient_scale)
            response = background + modal_observation
            output[member_index, start:stop, :] = np.abs(response.reshape(stop - start, n_f)) * 1.0e3
    return output


def _training_error_summary(case_dir: Path) -> tuple[float, float, int]:
    error_file = case_dir / "unseen_parameter_field_errors.csv"
    if not error_file.exists():
        return float("nan"), float("nan"), 0
    with error_file.open(newline="", encoding="utf-8-sig") as handle:
        values = np.asarray([float(row["relative_full_field_error"]) for row in csv.DictReader(handle)])
    return float(np.mean(values)), float(np.max(values)), int(values.size)


def _make_case_plot(
    case: str,
    frequencies: np.ndarray,
    median: np.ndarray,
    envelopes: dict[float, tuple[np.ndarray, np.ndarray]],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 3.9), constrained_layout=True)
    for coverage in sorted(envelopes, reverse=True):
        lower, upper = envelopes[coverage]
        axis.fill_between(
            frequencies,
            lower,
            upper,
            color=COLORS.get(coverage, "#9ecae1"),
            alpha=0.62,
            linewidth=0.0,
            label=f"{100.0 * coverage:.0f}% parameter envelope",
        )
    axis.plot(frequencies, median, color="#c51b1d", linewidth=1.25, label="Operator median", zorder=5)
    axis.set_title(CASE_LABELS[case])
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel(r"$|w|$ (mm)")
    axis.set_xlim(float(frequencies[0]), float(frequencies[-1]))
    axis.grid(True, linestyle=":", linewidth=0.45, alpha=0.55)
    style_axis(axis)
    axis.legend(loc="upper right", frameon=False, ncol=2)
    save_figure(figure, output)
    plt.close(figure)


def _make_combined_plot(
    case_data: dict[str, dict[str, object]],
    output: Path,
    *,
    logarithmic: bool,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(7.0, 8.4), sharex=True, constrained_layout=True)
    for axis, case in zip(axes, CASE_ORDER):
        data = case_data[case]
        frequencies = np.asarray(data["frequencies"], dtype=float)
        envelopes = data["envelopes"]
        for coverage in sorted(envelopes, reverse=True):
            lower, upper = envelopes[coverage]
            axis.fill_between(
                frequencies,
                np.maximum(lower, 1.0e-8) if logarithmic else lower,
                np.maximum(upper, 1.0e-8) if logarithmic else upper,
                color=COLORS.get(coverage, "#9ecae1"),
                alpha=0.62,
                linewidth=0.0,
                label=f"{100.0 * coverage:.0f}%",
            )
        median = np.asarray(data["median"], dtype=float)
        axis.plot(frequencies, np.maximum(median, 1.0e-8) if logarithmic else median, color="#c51b1d", linewidth=1.15)
        axis.text(0.015, 0.93, CASE_LABELS[case], transform=axis.transAxes, ha="left", va="top")
        axis.set_ylabel(r"$|w|$ (mm)")
        axis.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.55)
        if logarithmic:
            axis.set_yscale("log")
        style_axis(axis)
    axes[0].legend(
        title="Central parameter coverage",
        loc="upper right",
        frameon=False,
        ncol=4,
    )
    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xlim(
        float(np.asarray(case_data[CASE_ORDER[0]]["frequencies"])[0]),
        float(np.asarray(case_data[CASE_ORDER[0]]["frequencies"])[-1]),
    )
    save_figure(figure, output)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    apply_journal_style()
    frequencies = np.linspace(args.frequency_min, args.frequency_max, args.frequency_points)
    case_data: dict[str, dict[str, object]] = {}
    envelope_rows: list[dict[str, object]] = []
    peak_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for case in CASE_ORDER:
        checkpoint = args.results_root / CHECKPOINT_DIRS[case] / case / "parametric_operator.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {case}: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        bounds: ParameterBounds = payload["bounds"]
        system = build_case_system(case, MeshConfig())
        operators = prepare_constrained_operators(system)
        family = build_parametric_frequency_family(
            operators,
            bounds=bounds,
            modal_config=payload["config"].modal_resonance,
            frequency_min_hz=float(args.frequency_min),
            frequency_max_hz=float(args.frequency_max),
        )
        result = load_parametric_operator(checkpoint, family, device=args.device)
        observation_index = _observation_index(case, checkpoint)
        node_index, observation_x, observation_y = _observation_location(
            operators, system, observation_index
        )
        uq_parameters = bounds.sample_latin_hypercube(args.uq_samples, args.seed + 404)
        for index, point in enumerate(uq_parameters):
            sample_rows.append(
                {
                    "case": case,
                    "sample_index": index,
                    **{name: float(value) for name, value in zip(bounds.names, point)},
                }
            )

        member_curves = _predict_observation_members(
            result,
            frequencies,
            uq_parameters,
            observation_index,
            batch_parameters=args.batch_parameters,
        )
        np.savez_compressed(
            args.out / f"{case}_operator_observation_curves.npz",
            frequency_hz=frequencies,
            parameters=uq_parameters,
            member_curves_mm=member_curves,
            observation_constrained_index=np.asarray(observation_index),
        )
        mean_per_parameter = np.mean(member_curves, axis=0)
        median = np.median(mean_per_parameter, axis=0)
        mean = np.mean(mean_per_parameter, axis=0)
        ensemble_spread = np.mean(np.std(member_curves, axis=0, ddof=0), axis=0)
        envelopes: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        for coverage in args.coverages:
            tail = 0.5 * (1.0 - coverage)
            lower = np.quantile(mean_per_parameter, tail, axis=0)
            upper = np.quantile(mean_per_parameter, 1.0 - tail, axis=0)
            all_member_lower = np.quantile(member_curves.reshape(-1, frequencies.size), tail, axis=0)
            all_member_upper = np.quantile(member_curves.reshape(-1, frequencies.size), 1.0 - tail, axis=0)
            envelopes[coverage] = (lower, upper)
            width = upper - lower
            upper_peak_index = int(np.argmax(upper))
            width_peak_index = int(np.argmax(width))
            median_peak_index = int(np.argmax(median))
            peak_rows.append(
                {
                    "case": case,
                    "coverage": coverage,
                    "median_peak_mm": float(median[median_peak_index]),
                    "median_peak_frequency_hz": float(frequencies[median_peak_index]),
                    "upper_envelope_peak_mm": float(upper[upper_peak_index]),
                    "upper_envelope_peak_frequency_hz": float(frequencies[upper_peak_index]),
                    "maximum_envelope_width_mm": float(width[width_peak_index]),
                    "maximum_width_frequency_hz": float(frequencies[width_peak_index]),
                    "mean_envelope_width_mm": float(np.mean(width)),
                }
            )
            for frequency_index, frequency in enumerate(frequencies):
                envelope_rows.append(
                    {
                        "case": case,
                        "coverage": coverage,
                        "frequency_hz": float(frequency),
                        "operator_mean_mm": float(mean[frequency_index]),
                        "operator_median_mm": float(median[frequency_index]),
                        "parameter_lower_mm": float(lower[frequency_index]),
                        "parameter_upper_mm": float(upper[frequency_index]),
                        "ensemble_inclusive_lower_mm": float(all_member_lower[frequency_index]),
                        "ensemble_inclusive_upper_mm": float(all_member_upper[frequency_index]),
                        "mean_ensemble_spread_mm": float(ensemble_spread[frequency_index]),
                    }
                )

        mean_error, max_error, error_count = _training_error_summary(checkpoint.parent)
        best_epochs = [int(member.best_epoch) for member in result.members]
        best_losses = [float(member.best_total_loss) for member in result.members]
        training_rows.append(
            {
                "case": case,
                "checkpoint": str(checkpoint),
                "representation": result.representation,
                "ensemble_size": len(result.members),
                "epochs_configured": int(result.config.epochs),
                "best_epochs": ";".join(str(value) for value in best_epochs),
                "mean_best_total_loss": float(np.mean(best_losses)),
                "min_best_total_loss": float(np.min(best_losses)),
                "max_best_total_loss": float(np.max(best_losses)),
                "mean_unseen_full_field_relative_l2": mean_error,
                "max_unseen_full_field_relative_l2": max_error,
                "unseen_test_value_count": error_count,
                "constrained_dofs": int(result.n_dofs),
                "pod_modes": int(result.pod_basis.n_modes),
                "retained_modal_modes": 0 if family.modal is None else int(family.modal.n_modes),
                "observation_constrained_index": observation_index,
                "observation_node_index": node_index,
                "observation_x": observation_x,
                "observation_y": observation_y,
            }
        )
        case_data[case] = {
            "frequencies": frequencies,
            "median": median,
            "envelopes": envelopes,
        }
        _make_case_plot(
            case,
            frequencies,
            median,
            envelopes,
            args.out / f"{case}_multilevel_frf_envelopes.png",
        )

    _write_csv(args.out / "frf_envelopes_all_cases.csv", envelope_rows)
    _write_csv(args.out / "envelope_peak_summary.csv", peak_rows)
    _write_csv(args.out / "training_results_summary.csv", training_rows)
    _write_csv(args.out / "uq_parameter_samples.csv", sample_rows)
    _make_combined_plot(
        case_data,
        args.out / "three_plate_multilevel_frf_envelopes_linear.png",
        logarithmic=False,
    )
    _make_combined_plot(
        case_data,
        args.out / "three_plate_multilevel_frf_envelopes_log.png",
        logarithmic=True,
    )
    metadata = {
        "source": "retained 5000-epoch, five-member parametric neural-operator checkpoints",
        "cases": list(CASE_ORDER),
        "frequency_interval_hz": [float(args.frequency_min), float(args.frequency_max)],
        "frequency_points": int(args.frequency_points),
        "uq_parameter_samples_per_case": int(args.uq_samples),
        "uq_sampling": f"Latin hypercube, seed={int(args.seed) + 404}",
        "parameter_distribution_assumption": "joint uniform distribution inside the trained parameter box",
        "parameter_names": list(payload["bounds"].names),
        "parameter_lower": payload["bounds"].lower.tolist(),
        "parameter_upper": payload["bounds"].upper.tolist(),
        "coverage_levels": list(args.coverages),
        "envelope_definition": "central pointwise quantiles of the deep-ensemble mean |w(f, mu)| over parameter samples",
        "ensemble_inclusive_columns": "central pointwise quantiles after pooling ensemble members and parameter samples",
        "response_unit": "mm",
        "device": args.device,
        "conformal_inflation_applied": False,
        "note": "These are comparable operator parameter-propagation envelopes; they are not strict interval extrema or conformal simultaneous bands.",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.out), "training": training_rows, "peaks": peak_rows}, indent=2))


if __name__ == "__main__":
    main()

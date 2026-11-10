# -*- coding: utf-8 -*-
"""Evaluate a retained parametric operator over a frequency range."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from rbf_fd_full_order import (
    ParameterBounds,
    build_parametric_frequency_family,
    load_parametric_operator,
    predict_parametric_members,
    prepare_constrained_operators,
)
from rbf_fd_full_order.examples.cases import MeshConfig, build_case_system
from rbf_fd_full_order.visualization import apply_journal_style, save_figure, style_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("flat", "hollow", "stiffened"), default="hollow")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results") / "parametric_many_query_hollow_5000_best" / "hollow" / "parametric_operator.pt",
    )
    parser.add_argument("--frequency-min", type=float, default=0.0)
    parser.add_argument("--frequency-max", type=float, default=200.0)
    parser.add_argument("--frequency-points", type=int, default=41)
    parser.add_argument("--observation-index", type=int, default=224)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "parametric_many_query_hollow_5000_best" / "hollow" / "frequency_range_evaluation",
    )
    return parser.parse_args()


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - reference) / max(float(np.linalg.norm(reference)), 1.0e-30))


def main() -> None:
    args = parse_args()
    if args.frequency_points < 2 or args.frequency_max <= args.frequency_min:
        raise ValueError("The frequency range must be increasing and contain at least two points.")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    bounds: ParameterBounds = payload["bounds"]
    # Use three unseen parameter points already used by the completed test run.
    parameter_csv = args.checkpoint.parent / "unseen_parameter_field_errors.csv"
    points: dict[int, np.ndarray] = {}
    with parameter_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = int(row["parameter_index"])
            points[index] = np.asarray(
                [float(row[name]) for name in bounds.names],
                dtype=float,
            )
    selected_indices = [index for index in (0, 3, 7) if index in points]
    if len(selected_indices) != 3:
        raise RuntimeError("The completed unseen-parameter test points are unavailable.")
    parameters = np.vstack([points[index] for index in selected_indices])

    system = build_case_system(args.case, MeshConfig())
    operators = prepare_constrained_operators(system)
    family = build_parametric_frequency_family(
        operators,
        bounds=bounds,
        modal_config=payload["config"].modal_resonance,
        frequency_min_hz=float(args.frequency_min),
        frequency_max_hz=float(args.frequency_max),
    )
    result = load_parametric_operator(args.checkpoint, family, device="cuda" if torch.cuda.is_available() else "cpu")
    frequencies = np.linspace(float(args.frequency_min), float(args.frequency_max), int(args.frequency_points))

    reference = family.solve_grid(frequencies, parameters)
    members = predict_parametric_members(result, frequencies, parameters)
    prediction = np.mean(members, axis=0)
    field_errors = np.asarray(
        [[_relative_l2(prediction[p, f], reference[p, f]) for f in range(frequencies.size)] for p in range(parameters.shape[0])],
        dtype=float,
    )

    if operators.reduction_mode == "fixed_dofs" and operators.free_dofs is not None:
        candidates = np.flatnonzero(np.asarray(operators.free_dofs) < int(system.nodes.shape[0]))
    else:
        candidates = np.arange(operators.n_free, dtype=int)
    if int(args.observation_index) in set(int(value) for value in candidates):
        observation_index = int(args.observation_index)
    else:
        nominal = family.solve_grid(frequencies, bounds.center)
        observation_index = int(candidates[np.argmax(np.max(np.abs(nominal[:, candidates]), axis=0))])
    reference_curve = np.abs(reference[..., observation_index]) * 1.0e3
    member_curves = np.abs(members[..., observation_index]) * 1.0e3
    prediction_curve = np.mean(member_curves, axis=0)
    spread_curve = np.std(member_curves, axis=0)
    ensemble_lower = np.min(member_curves, axis=0)
    ensemble_upper = np.max(member_curves, axis=0)

    rows: list[dict[str, object]] = []
    for p, parameter_index in enumerate(selected_indices):
        for f, frequency in enumerate(frequencies):
            rows.append(
                {
                    "parameter_index": int(parameter_index),
                    **{name: float(value) for name, value in zip(bounds.names, parameters[p])},
                    "frequency_hz": float(frequency),
                    "relative_full_field_error": float(field_errors[p, f]),
                    "reference_observation_mm": float(reference_curve[p, f]),
                    "prediction_observation_mm": float(prediction_curve[p, f]),
                    "ensemble_std_mm": float(spread_curve[p, f]),
                    "ensemble_lower_mm": float(ensemble_lower[p, f]),
                    "ensemble_upper_mm": float(ensemble_upper[p, f]),
                    "ensemble_range_mm": float(ensemble_upper[p, f] - ensemble_lower[p, f]),
                }
            )
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics: list[dict[str, object]] = []
    for p, parameter_index in enumerate(selected_indices):
        reference_peak = int(np.argmax(reference_curve[p]))
        prediction_peak = int(np.argmax(prediction_curve[p]))
        metrics.append(
            {
                "parameter_index": int(parameter_index),
                **{name: float(value) for name, value in zip(bounds.names, parameters[p])},
                "mean_full_field_relative_l2": float(np.mean(field_errors[p])),
                "median_full_field_relative_l2": float(np.median(field_errors[p])),
                "max_full_field_relative_l2": float(np.max(field_errors[p])),
                "observation_curve_relative_l2": _relative_l2(prediction_curve[p], reference_curve[p]),
                "observation_curve_max_absolute_error_mm": float(np.max(np.abs(prediction_curve[p] - reference_curve[p]))),
                "mean_ensemble_range_mm": float(np.mean(ensemble_upper[p] - ensemble_lower[p])),
                "max_ensemble_range_mm": float(np.max(ensemble_upper[p] - ensemble_lower[p])),
                "reference_peak_frequency_hz": float(frequencies[reference_peak]),
                "prediction_peak_frequency_hz": float(frequencies[prediction_peak]),
                "peak_frequency_error_hz": float(abs(frequencies[prediction_peak] - frequencies[reference_peak])),
            }
        )
    aggregate = {
        "case": args.case,
        "frequency_range_hz": [float(args.frequency_min), float(args.frequency_max)],
        "frequency_points": int(args.frequency_points),
        "evaluated_parameter_indices": selected_indices,
        "observation_constrained_index": observation_index,
        "mean_full_field_relative_l2": float(np.mean(field_errors)),
        "median_full_field_relative_l2": float(np.median(field_errors)),
        "max_full_field_relative_l2": float(np.max(field_errors)),
        "per_parameter": metrics,
    }
    with args.out.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    apply_journal_style()
    figure, axes = plt.subplots(3, 1, figsize=(6.4, 7.2), sharex=True, constrained_layout=True)
    for axis, p, parameter_index in zip(axes, range(parameters.shape[0]), selected_indices):
        axis.plot(frequencies, reference_curve[p], color="#000000", linewidth=1.2, label="reference")
        axis.fill_between(
            frequencies,
            ensemble_lower[p],
            ensemble_upper[p],
            color="#C00000",
            alpha=0.20,
            linewidth=0.0,
            label="ensemble min-max range",
            zorder=1,
        )
        axis.plot(frequencies, ensemble_lower[p], color="#C00000", linewidth=0.65, linestyle=":", label="ensemble bounds")
        axis.plot(frequencies, ensemble_upper[p], color="#C00000", linewidth=0.65, linestyle=":")
        axis.plot(frequencies, prediction_curve[p], color="#C00000", linewidth=1.0, linestyle="--", label="operator mean", zorder=3)
        zoom = inset_axes(axis, width="33%", height="34%", loc="upper right", borderpad=1.15)
        zoom_mask = (frequencies >= 40.0) & (frequencies <= 60.0)
        zoom.fill_between(
            frequencies[zoom_mask],
            ensemble_lower[p, zoom_mask],
            ensemble_upper[p, zoom_mask],
            color="#C00000",
            alpha=0.24,
            linewidth=0.0,
        )
        zoom.plot(frequencies[zoom_mask], reference_curve[p, zoom_mask], color="#000000", linewidth=0.75)
        zoom.plot(frequencies[zoom_mask], prediction_curve[p, zoom_mask], color="#C00000", linewidth=0.75, linestyle="--")
        zoom.plot(frequencies[zoom_mask], ensemble_lower[p, zoom_mask], color="#C00000", linewidth=0.55, linestyle=":")
        zoom.plot(frequencies[zoom_mask], ensemble_upper[p, zoom_mask], color="#C00000", linewidth=0.55, linestyle=":")
        zoom.set_xlim(40.0, 60.0)
        zoom_min = float(np.min(ensemble_lower[p, zoom_mask]))
        zoom_max = float(np.max(ensemble_upper[p, zoom_mask]))
        zoom_margin = max(0.08 * (zoom_max - zoom_min), 1.0e-5)
        zoom.set_ylim(zoom_min - zoom_margin, zoom_max + zoom_margin)
        zoom.set_xticks([40.0, 50.0, 60.0])
        zoom.tick_params(which="both", direction="in", labelsize=7, pad=1)
        zoom.grid(True, linestyle=":", linewidth=0.3, alpha=0.5)
        axis.set_ylabel(r"$|w|$ (mm)")
        axis.set_title(f"unseen parameter {parameter_index}", loc="left")
        axis.grid(True, linestyle=":", linewidth=0.45, alpha=0.55)
        style_axis(axis)
        axis.legend(loc="upper left", bbox_to_anchor=(0.04, 0.96), frameon=False, ncol=2)
    axes[-1].set_xlabel("frequency (Hz)")
    save_figure(figure, args.out.with_suffix(".png"))
    plt.close(figure)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

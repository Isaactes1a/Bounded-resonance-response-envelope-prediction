# -*- coding: utf-8 -*-
"""Visual summary of the completed flat, stiffened, and hollow runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

from rbf_fd_full_order.visualization import apply_journal_style, save_figure, style_axis


ROOT = Path(__file__).resolve().parent
CASES = {
    "Flat plate": ROOT / "results" / "parametric_many_query_flat_5000_best" / "flat",
    "Stiffened plate": ROOT / "results" / "parametric_many_query_stiffened_5000_best" / "stiffened",
    "Hollow plate": ROOT / "results" / "parametric_many_query_hollow_5000_best" / "hollow",
}
COLORS = {
    "Flat plate": "#0072B2",
    "Stiffened plate": "#D55E00",
    "Hollow plate": "#009E73",
}
MARKERS = {"Flat plate": "o", "Stiffened plate": "^", "Hollow plate": "s"}
COST_DIRECTORIES = {
    "Hollow plate": ROOT / "results" / "parametric_many_query_hollow_smoke_best" / "hollow",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_history(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    members = payload["members"]
    epochs = np.asarray([point["epoch"] for point in members[0]["history"]], dtype=float)
    losses = np.asarray(
        [[point["total_loss"] for point in member["history"]] for member in members],
        dtype=float,
    )
    return epochs, losses


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.13, 1.05, label, transform=axis.transAxes, fontweight="bold", va="top")


def load_accuracy_values(directory: Path) -> np.ndarray:
    evaluation = read_json(directory / "frequency_range_evaluation.json")
    statistics_path = directory / "high_fidelity_statistical_comparison.json"
    if statistics_path.exists():
        statistics = read_json(statistics_path)
        unseen_mean = statistics["mean_unseen_full_field_error"]
        unseen_max = statistics["max_unseen_full_field_error"]
    else:
        unseen = pd.read_csv(directory / "unseen_parameter_field_errors.csv")
        unseen_mean = float(unseen["relative_full_field_error"].mean())
        unseen_max = float(unseen["relative_full_field_error"].max())
    return np.asarray(
        [
            evaluation["mean_full_field_relative_l2"],
            evaluation["median_full_field_relative_l2"],
            evaluation["max_full_field_relative_l2"],
            unseen_mean,
            unseen_max,
        ]
    ) * 100.0


def load_cost(directory: Path, case: str) -> dict:
    cost_path = directory / "cost_and_break_even.json"
    if not cost_path.exists():
        cost_path = COST_DIRECTORIES[case] / "cost_and_break_even.json"
    return read_json(cost_path)


def main() -> None:
    apply_journal_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 6.0), constrained_layout=True)

    # (a) Frequency-resolved full-field error: mean and full min--max envelope.
    error_axis = axes[0, 0]
    for case, directory in CASES.items():
        frame = pd.read_csv(directory / "frequency_range_evaluation.csv")
        grouped = frame.groupby("frequency_hz")["relative_full_field_error"]
        frequency = grouped.mean().index.to_numpy(dtype=float)
        mean_error = grouped.mean().to_numpy(dtype=float) * 100.0
        minimum = grouped.min().to_numpy(dtype=float) * 100.0
        maximum = grouped.max().to_numpy(dtype=float) * 100.0
        error_axis.fill_between(frequency, minimum, maximum, color=COLORS[case], alpha=0.14, linewidth=0)
        error_axis.semilogy(
            frequency,
            mean_error,
            color=COLORS[case],
            linewidth=1.15,
            marker=MARKERS[case],
            markersize=2.7,
            markevery=4,
            markerfacecolor="white",
            label=case,
        )
    error_axis.set(xlabel="Frequency (Hz)", ylabel="Full-field relative error (%)", xlim=(0, 200))
    error_axis.set_xticks([0, 50, 100, 150, 200])
    error_axis.grid(True, which="major", linestyle=":", linewidth=0.45, alpha=0.55)
    error_axis.legend(frameon=False, loc="best")
    style_axis(error_axis)
    add_panel_label(error_axis, "(a)")

    # (b) Ensemble training convergence: mean and member envelope.
    loss_axis = axes[0, 1]
    for case, directory in CASES.items():
        epochs, losses = load_history(directory / "parametric_operator.pt")
        mean_loss = np.mean(losses, axis=0)
        loss_axis.fill_between(
            epochs,
            np.min(losses, axis=0),
            np.max(losses, axis=0),
            color=COLORS[case],
            alpha=0.14,
            linewidth=0,
        )
        loss_axis.semilogy(epochs, mean_loss, color=COLORS[case], linewidth=1.15, label=case)
    loss_axis.set(xlabel="Epoch", ylabel="Total loss", xlim=(0, 5000))
    loss_axis.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    loss_axis.grid(True, which="major", linestyle=":", linewidth=0.45, alpha=0.55)
    loss_axis.legend(frameon=False, loc="best")
    style_axis(loss_axis)
    add_panel_label(loss_axis, "(b)")

    # (c) Compact accuracy comparison from the retained evaluation summaries.
    metric_axis = axes[1, 0]
    metrics = ["Mean", "Median", "Maximum", "Unseen mean", "Unseen max"]
    x_position = np.arange(len(metrics), dtype=float)
    width = 0.25
    offsets = np.linspace(-width, width, len(CASES))
    for offset, (case, directory) in zip(offsets, CASES.items()):
        values = load_accuracy_values(directory)
        metric_axis.bar(
            x_position + offset,
            values,
            width=width,
            color=COLORS[case],
            edgecolor="black",
            linewidth=0.45,
            label=case,
        )
    metric_axis.set_yscale("log")
    metric_axis.set_ylabel("Relative error (%)")
    metric_axis.set_xticks(x_position, metrics, rotation=18, ha="right")
    metric_axis.grid(True, axis="y", which="major", linestyle=":", linewidth=0.45, alpha=0.55)
    style_axis(metric_axis)
    add_panel_label(metric_axis, "(c)")

    # (d) Online time and derived speedup; offline break-even is annotated.
    time_axis = axes[1, 1]
    categories = ["Direct full sweep", "Surrogate sweep"]
    x_time = np.arange(len(categories), dtype=float)
    for offset, (case, directory) in zip(offsets, CASES.items()):
        cost = load_cost(directory, case)
        direct = float(cost["direct_full_sweep_per_parameter_s"])
        surrogate = float(cost["operator_ensemble_full_sweep_per_parameter_s"])
        bars = time_axis.bar(
            x_time + offset,
            [direct, surrogate],
            width=width,
            color=COLORS[case],
            edgecolor="black",
            linewidth=0.45,
            label=case,
        )
        speedup = direct / surrogate
        time_axis.text(
            x_time[1] + offset,
            surrogate * 1.35,
            f"{speedup:.0f}x\nBE={cost['break_even_parameter_queries']}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS[case],
        )
    time_axis.set_yscale("log")
    time_axis.set_ylabel("Time per 0--200 Hz sweep (s)")
    time_axis.set_xticks(x_time, categories)
    time_axis.grid(True, axis="y", which="major", linestyle=":", linewidth=0.45, alpha=0.55)
    time_axis.legend(frameon=False, loc="upper right")
    style_axis(time_axis)
    add_panel_label(time_axis, "(d)")

    for axis in axes.flat:
        axis.tick_params(direction="in", which="both", top=True, right=True)
        axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        axis.yaxis.set_minor_formatter(NullFormatter())

    output = ROOT / "results" / "parametric_flat_stiffened_comparison" / "flat_stiffened_visual_summary.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(figure, output)
    plt.close(figure)
    print(f"saved: {output.with_suffix('.png')}")
    print(f"saved: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

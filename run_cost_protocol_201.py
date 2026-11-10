# -*- coding: utf-8 -*-
"""Recompute offline/online cost and break-even on the current 40x201 protocol.

The retained manuscript previously quoted offline construction totals and
break-even counts measured with an older protocol (8 calibration realizations
x 401 frequency points).  The current paper reports 40 calibration + 40 test
realizations on a 201-point grid.  This script recomputes the cost table from

- per-query timings measured in this session on the same hardware
  (``results/classical_modal_baselines/*_timing.json``);
- solve-count arithmetic for the direct RBF-FD components
  (labels: 20 realizations x 24 frequencies = 480 solves;
   calibration: 40 realizations x 201 frequencies = 8040 solves);
- recorded GPU training times (unchanged model, five-member ensemble);
- measured classical modal-baseline online costs.

All quantities are reported with the hardware description and the
measurement protocol (warm-up, repeats, mean +- std).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rbf_fd_full_order.visualization import apply_journal_style, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results") / "cost_protocol_201")
    parser.add_argument("--baseline-root", type=Path, default=Path("results") / "classical_modal_baselines")
    parser.add_argument("--production-root", type=Path, default=Path("results"))
    parser.add_argument("--calibration-realizations", type=int, default=40)
    parser.add_argument("--test-realizations", type=int, default=40)
    parser.add_argument("--frequency-points", type=int, default=201)
    parser.add_argument("--uq-statistics-realizations", type=int, default=30)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)


def run_case(case: str, args: argparse.Namespace) -> dict[str, object]:
    timing = _read_json(args.baseline_root / f"{case}_timing.json")
    baseline_summary = _read_json(args.baseline_root / case / "baseline_summary.json")
    case_dir = args.production_root / f"parametric_many_query_{case}_5000_best" / case
    old_cost = _read_json(case_dir / "cost_and_break_even.json")
    # The number of frequency points is a property of the model, so the solve
    # arithmetic follows the case rather than a single global convention.
    case_metadata = _read_json(case_dir / "metadata.json")
    # One observation query covers the evaluation grid of the case, which is the
    # band sampled at 1 Hz. The training sweep may be finer than that.
    band = case_metadata.get("frequency_interval_hz") or [0.0, float(args.frequency_points)]
    case_points = int(round(float(band[1]) - float(band[0]))) + 1
    n_labels = 20 * 24
    n_calibration = args.calibration_realizations * case_points
    n_test = args.test_realizations * case_points
    n_uq_stats = args.uq_statistics_realizations * case_points

    direct_query_s = float(timing["timing_s"]["direct"]["mean_s"])
    direct_query_std = float(timing["timing_s"]["direct"]["std_s"])
    surrogate_query_s = float(timing["timing_s"]["proposed"]["mean_s"])
    surrogate_query_std = float(timing["timing_s"]["proposed"]["std_s"])
    modal_query_s = float(timing["timing_s"]["ma_sc8"]["mean_s"])
    modal_query_std = float(timing["timing_s"]["ma_sc8"]["std_s"])
    per_solve_s = direct_query_s / case_points

    labels_s = n_labels * per_solve_s
    calibration_s = n_calibration * per_solve_s
    test_reference_s = n_test * per_solve_s
    uq_stats_reference_s = n_uq_stats * per_solve_s
    obs_ratio = float(old_cost["observation_selection_s"]) / float(old_cost["high_fidelity_training_data_s"])
    observation_s = n_labels * per_solve_s * obs_ratio

    offline_s = (
        float(old_cost["assembly_s"])
        + float(old_cost["modal_preparation_s"])
        + labels_s
        + float(old_cost["ensemble_training_s"])
        + observation_s
        + calibration_s
    )
    modal_offline_s = (
        float(timing["offline_s"]["operator_assembly_s"])
        + float(timing["offline_s"]["modal_eigensolve_ms8_s"])
        + float(timing["offline_s"]["static_correction_solve_s"])
    )
    speedup = direct_query_s / max(surrogate_query_s, 1.0e-30)
    break_even = int(math.ceil(offline_s / max(direct_query_s - surrogate_query_s, 1.0e-30)))

    summary = {
        "case": case,
        "protocol": {
            "calibration_realizations": args.calibration_realizations,
            "test_realizations": args.test_realizations,
            "frequency_points": case_points,
            "uq_statistics_realizations": args.uq_statistics_realizations,
            "label_solve_count": n_labels,
            "calibration_solve_count": n_calibration,
        },
        "hardware": baseline_summary["hardware"],
        "timing": {
            "direct_query_mean_s": direct_query_s,
            "direct_query_std_s": direct_query_std,
            "direct_timing_runs": 3 if case == "flat" else 2,
            "surrogate_query_mean_s": surrogate_query_s,
            "surrogate_query_std_s": surrogate_query_std,
            "surrogate_timing_runs": 3,
            "modal_ma_sc8_query_mean_s": modal_query_s,
            "modal_ma_sc8_query_std_s": modal_query_std,
            "modal_timing_runs": 5,
            "per_solve_mean_s": per_solve_s,
        },
        "offline_s": {
            "operator_assembly_s": float(old_cost["assembly_s"]),
            "modal_preparation_s": float(old_cost["modal_preparation_s"]),
            "training_labels_s": labels_s,
            "ensemble_training_s": float(old_cost["ensemble_training_s"]),
            "observation_selection_s": observation_s,
            "calibration_reference_s": calibration_s,
            "offline_total_s": offline_s,
            "assessment_test_reference_s": test_reference_s,
            "assessment_uq_statistics_reference_s": uq_stats_reference_s,
            "modal_baseline_offline_s": modal_offline_s,
        },
        "marginal": {
            "direct_s": direct_query_s,
            "surrogate_s": surrogate_query_s,
            "modal_ma_sc8_s": modal_query_s,
            "marginal_speedup": speedup,
            "break_even_parameter_queries": break_even,
        },
    }
    _write_json(args.out / f"{case}_cost.json", summary)
    print(json.dumps(summary, indent=2, default=float))
    return summary


def make_figure(summaries: list[dict[str, object]], out: Path) -> None:
    apply_journal_style()
    figure, axes = plt.subplots(1, 1, figsize=(7.2, 4.2), constrained_layout=True)
    n_max = 120
    queries = np.arange(1, n_max + 1)
    for summary in summaries:
        case = summary["case"]
        color = "#D62728" if case == "flat" else "#1F77B4"
        direct = float(summary["timing"]["direct_query_mean_s"])
        surrogate = float(summary["timing"]["surrogate_query_mean_s"])
        modal = float(summary["timing"]["modal_ma_sc8_query_mean_s"])
        offline = float(summary["offline_s"]["offline_total_s"])
        modal_offline = float(summary["offline_s"]["modal_baseline_offline_s"])
        axes.plot(queries, queries * direct, color=color, linestyle="-", linewidth=1.4)
        axes.plot(queries, offline + queries * surrogate, color=color, linestyle="--", linewidth=1.4)
        axes.plot(queries, modal_offline + queries * modal, color=color, linestyle=":", linewidth=1.4)
        be = int(summary["marginal"]["break_even_parameter_queries"])
        axes.axvline(be, color=color, alpha=0.45, linewidth=1.0)
        axes.annotate(
            f"{case} N_BE={be}",
            xy=(be, offline + be * surrogate),
            xytext=(be + 2, 0.55 * offline),
            fontsize=8,
            color=color,
        )
    axes.set_xlabel("Number of complete 201-point parameter queries")
    axes.set_ylabel("Cumulative cost (s)")
    axes.grid(True, alpha=0.3)
    handles = [
        plt.Line2D([0], [0], color="k", linestyle="-", label="Direct RBF-FD"),
        plt.Line2D([0], [0], color="k", linestyle="--", label="Surrogate (offline + queries)"),
        plt.Line2D([0], [0], color="k", linestyle=":", label="Static-corrected modal method"),
    ]
    axes.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)
    save_figure(figure, out / "cumulative_cost_201.png")
    save_figure(figure, out / "cumulative_cost_201.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = [run_case("flat", args), run_case("stiffened", args)]
    _write_json(args.out / "cost_summary.json", summaries)
    make_figure(summaries, args.out)
    print(f"Wrote cost summaries to {args.out}", flush=True)


if __name__ == "__main__":
    main()

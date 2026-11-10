# -*- coding: utf-8 -*-
"""Compact Route-B pilot figure: modal-basis change and method comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rbf_fd_full_order.visualization import apply_journal_style, save_figure, style_axis


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads(
    (ROOT / "code" / "results" / "route_b_pilot_1500" / "summary.json").read_text("utf-8")
)
OUT = ROOT / "manuscript" / "current" / "result_assets" / "figures" / "results_section"


def main() -> None:
    apply_journal_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    ax = axes[0]
    mac = SUMMARY["mac_low_vs_high"]
    modes = [int(row["mode"]) for row in mac]
    values = [float(row["mac_low_vs_high"]) for row in mac]
    ax.semilogy(modes, np.maximum(values, 1.0e-10), "o-", color="#C00000", lw=1.1)
    ax.axhline(1.0, color="black", linestyle=":", lw=1.0)
    ax.set_xlabel("mode (nominal ordering)")
    ax.set_ylabel("MAC($s_b=0.5$, $s_b=1.5$)")
    ax.set_ylim(1.0e-10, 2.0)
    ax.annotate("modes 3/4 exchange", xy=(3.5, 1.0e-7), fontsize=8, ha="center")
    style_axis(ax)

    ax = axes[1]
    methods = {
        "ms_nominal": "MS, nominal modes",
        "pole_only_nominal": "Pole-only, nominal modes",
        "ms_param": "MS, parameter modes",
        "ma_sc_param": "MA + static corr., parameter modes",
        "direct_data": "Pole-free NN (data)",
        "direct_physics": "Pole-free NN (physics)",
    }
    rows = {row["method"]: row for row in SUMMARY["baselines"] + SUMMARY["neural"]}
    labels = [methods[name] for name in methods]
    errors = [float(rows[name]["median_observation_relative_l2_percent"]) for name in methods]
    colors = ["#808080", "#808080", "#0070C0", "#0070C0", "#4DBEEE", "#4DBEEE"]
    bars = ax.bar(range(len(labels)), errors, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=16, fontsize=7)
    ax.set_ylabel("Median observation-FRF rel. $L_2$ error (%)")
    for bar, value in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.25,
                f"{value:.3g}", ha="center", fontsize=7)
    style_axis(ax)

    save_figure(fig, OUT / "FigB1_route_b_pilot.pdf", tight=False)
    plt.close(fig)
    print("FigB1 written", flush=True)


if __name__ == "__main__":
    main()

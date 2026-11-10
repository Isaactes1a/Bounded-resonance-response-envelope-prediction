# -*- coding: utf-8 -*-
"""Supplementary vector figures for the revision.

Generates:
    Fig00_method_overview.pdf          workflow schematic (methodology)
    Fig16_classical_baselines.pdf      copied from the classical-baseline run
    Fig17_convergence.pdf              discretization convergence
    Fig18_extremum_backsubstitution.pdf  surrogate vs direct extrema

and replaces Fig15_cumulative_cost_break_even.pdf with the 201-point
protocol cost figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from rbf_fd_full_order.visualization import apply_journal_style, save_figure, style_axis


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"
OUT = ROOT / "manuscript" / "current" / "result_assets" / "figures" / "results_section"


def method_overview() -> None:
    apply_journal_style()
    fig, axis = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)
    axis.axis("off")

    def box(x: float, y: float, text: str, width: float = 1.72, height: float = 0.78,
            face: str = "#F2F2F2", edge: str = "black") -> None:
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.0, edgecolor=edge, facecolor=face, mutation_aspect=1.0,
        )
        axis.add_patch(patch)
        axis.text(x, y, text, ha="center", va="center", fontsize=9, linespacing=1.25)

    def arrow(x1: float, y1: float, x2: float, y2: float, text: str = "",
              curvature: float = 0.0) -> None:
        axis.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                linewidth=1.0, color="black", connectionstyle=f"arc3,rad={curvature}",
            )
        )
        if text:
            axis.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center",
                      va="bottom", fontsize=8, style="italic")

    # Top row: operator assembly and the two branches.
    box(0.0, 1.55, "PHS-RBF-FD operator assembly\n$K_0$, $M_0$, $f_0$, $T$",
        width=2.1, height=0.9)
    box(-2.35, 0.0, "Numerically anchored rational branch\n(analytically parameterized poles)", width=2.25, height=0.85)
    box(2.35, 0.0, "POD-reduced neural background\n(standardized coefficients)", width=1.9, height=0.85)
    box(0.0, -1.55, "Reconstructed response constrained by\nthe discrete dynamic-equilibrium residual",
        width=2.35, height=0.9)

    arrow(0.0, 1.08, -1.45, 0.42, "eigenpairs")
    arrow(0.0, 1.08, 1.45, 0.42, "labels")
    arrow(-2.35, -0.43, -0.75, -1.12, "pole response")
    arrow(2.35, -0.43, 0.75, -1.12, "background")
    arrow(0.0, -1.08, 0.0, -0.43, "same operators", curvature=-0.12)

    # Bottom row: UQ outputs.
    box(-1.55, -2.9, "Ellipsoid propagation + constrained\nextremum search + direct back-substitution",
        width=2.55, height=0.85)
    box(1.55, -2.9, "Curve-wise split-conformal calibration\n(ensemble spread + direct reference)",
        width=2.35, height=0.85)
    arrow(-0.85, -2.0, -1.0, -2.48)
    arrow(0.85, -2.0, 1.0, -2.48)

    axis.set_xlim(-3.8, 3.8)
    axis.set_ylim(-3.55, 2.35)
    save_figure(fig, OUT / "Fig00_method_overview.pdf", tight=False)
    plt.close(fig)


def convergence() -> None:
    apply_journal_style()
    summary = json.loads((RESULTS / "rbffd_convergence_validation" / "convergence_summary.json").read_text("utf-8"))
    rbffd = [r for r in summary["rbffd_rows"] if r["study"] == "ndiv_convergence"]
    fem = summary["fem_rows"]
    frf = summary["frf_convergence_rows"]
    kirchhoff = summary["kirchhoff_hz"]["f_11"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True)

    axis = axes[0]
    ndiv = [r["ndiv"] for r in rbffd]
    axis.plot(ndiv, [r["f_1_hz"] for r in rbffd], "o-", color="#C00000", label="PHS-RBF-FD")
    axis.plot([r["ndiv"] for r in fem], [r["f_1_hz"] for r in fem], "s--", color="#0070C0",
              label="MITC4 FEM")
    axis.axhline(kirchhoff, color="black", linestyle=":", linewidth=1.0,
                 label=f"Kirchhoff {kirchhoff:.3f} Hz")
    axis.set_xlabel("Node division ndiv")
    axis.set_ylabel("$f_1$ (Hz)")
    axis.set_xticks([10, 15, 20, 25, 30, 40])
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)

    axis = axes[1]
    ndiv = [r["ndiv"] for r in frf]
    errors = [r["relative_l2_error_vs_ndiv40_percent"] for r in frf]
    axis.semilogy(ndiv, errors, "o-", color="#C00000", label="FRF rel. $L_2$ error")
    axis.axhline(2.0, color="black", linestyle=":", linewidth=1.0, label="2% level")
    axis.set_xlabel("Node division ndiv")
    axis.set_ylabel("Relative $L_2$ error (%)")
    axis.set_xticks([10, 20, 30, 40])
    axis.legend(frameon=False, fontsize=8)
    style_axis(axis)
    for x, value in zip(ndiv, errors):
        if value > 0:
            axis.annotate(f"{value:.1f}%", (x, value), textcoords="offset points",
                          xytext=(0, 6), ha="center", fontsize=8)

    save_figure(fig, OUT / "Fig17_convergence.pdf", tight=False)
    plt.close(fig)


def extremum_backsubstitution() -> None:
    apply_journal_style()
    cases = ["flat", "stiffened"]
    labels = ["Unstiffened plate", "Stiffened plate"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True, sharey=True)

    for axis, case, label in zip(axes, cases, labels):
        data = json.loads(
            (RESULTS / "ellipsoid_extremum_search" / case / "extremum_summary.json").read_text("utf-8")
        )
        rows = data["backsubstitution"]
        mm_targets = ("peak_amplitude", "envelope_upper_at_f1")
        rows = [row for row in rows if row["target"] in mm_targets]
        for row in rows:
            target = "peak amplitude" if row["target"] == "peak_amplitude" else "envelope at $f_1$"
            surrogate = float(row["surrogate_value_at_f1_mm"])
            direct = float(row["direct_value_at_f1_mm"])
            axis.plot([direct], [surrogate], "o", color="#C00000", markersize=7)
            axis.annotate(target, (direct, surrogate),
                          textcoords="offset points", xytext=(7, 4), fontsize=8)
        upper = max(float(row["direct_value_at_f1_mm"]) for row in rows)
        axis.plot([0, upper * 1.08], [0, upper * 1.08],
                  linestyle=":", color="black", linewidth=1.0)
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("Direct RBF-FD value (mm)")
        axis.set_xlim(0, None)
        axis.set_ylim(0, None)
        style_axis(axis)
    axes[0].set_ylabel("Surrogate value (mm)")
    save_figure(fig, OUT / "Fig18_extremum_backsubstitution.pdf", tight=False)
    plt.close(fig)


def copy_existing() -> None:
    import shutil
    shutil.copyfile(
        RESULTS / "classical_modal_baselines" / "classical_baselines.pdf",
        OUT / "Fig16_classical_baselines.pdf",
    )
    shutil.copyfile(
        RESULTS / "cost_protocol_201" / "cumulative_cost_201.pdf",
        OUT / "Fig15_cumulative_cost_break_even.pdf",
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    method_overview()
    convergence()
    extremum_backsubstitution()
    copy_existing()
    print(f"Figures written to {OUT}")

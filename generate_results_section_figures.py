# -*- coding: utf-8 -*-
"""Generate the result-section figure set described in RESULTS_SECTION_WRITING_GUIDE.md.

The script is intentionally tolerant of incomplete experiments: figures whose
source files are unavailable are skipped and listed in ``figure_manifest.csv``.
This allows the plotting layer to be installed before all expensive training,
HF statistics and ablation runs have finished.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rbf_fd_full_order.examples.cases import MeshConfig, build_case_system
from rbf_fd_full_order.visualization import apply_journal_style, save_figure, style_axis
from rbf_fd_full_order.visualization.result_support import mesh_from_metadata, read_json

CASES = ("flat", "annular", "stiffened")
CASE_LABELS = {
    "flat": "Solid plate",
    "annular": "Circular perforated plate",
    "stiffened": "Stiffened plate",
}
CASE_COLORS = {"flat": "#000000", "annular": "#0070C0", "stiffened": "#C00000"}
RED = "#C00000"
BLACK = "#000000"
BLUE = "#0070C0"
GRAY = "#808080"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parametric-root", type=Path, default=Path("results") / "parametric_many_query")
    parser.add_argument("--ellipsoid-root", type=Path, default=Path("results") / "ellipsoidal_uq")
    parser.add_argument("--parameter-effect-root", type=Path, default=Path("results") / "parameter_effects")
    parser.add_argument("--ablation-root", type=Path, default=Path("results") / "parametric_ablation")
    parser.add_argument(
        "--verification-csv",
        type=Path,
        default=Path("results") / "verification" / "rbf_fd_verification.csv",
        help="Optional independent analytical/FEM verification table; see writing guide for schema.",
    )
    parser.add_argument("--out", type=Path, default=Path("results") / "manuscript_result_figures")
    parser.add_argument("--representative-case", choices=CASES, default="stiffened")
    parser.add_argument(
        "--calibration-case",
        choices=CASES,
        default=None,
        help="Case used in the calibration panel; defaults to --representative-case.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _save(fig, out: Path, stem: str) -> None:
    save_figure(fig, out / f"{stem}.png", tight=False)
    plt.close(fig)


def _manifest_row(name: str, status: str, source: str) -> dict:
    return {"figure": name, "status": status, "source": source}


def figure5_geometry(args) -> dict:
    fig, axes = plt.subplots(1, len(CASES), figsize=(6.42, 3.0), constrained_layout=True)
    for ax, case in zip(axes, CASES):
        metadata_path = args.parametric_root / case / "metadata.json"
        mesh = MeshConfig()
        if metadata_path.exists():
            mesh = mesh_from_metadata(read_json(metadata_path))
        system = build_case_system(case, mesh)
        nodes = np.asarray(system.nodes, dtype=float)
        ax.scatter(nodes[:, 0], nodes[:, 1], s=2.1, facecolors="none", edgecolors=BLACK, linewidths=0.3)
        xmin, ymin = nodes.min(axis=0)
        xmax, ymax = nodes.max(axis=0)
        if case == "annular":
            angle = np.linspace(0.0, 2.0 * np.pi, 361)
            outer_radius = float(system.metadata["outer_radius"])
            inner_radius = float(system.metadata["inner_radius"])
            for radius in (inner_radius, outer_radius):
                ax.plot(radius * np.cos(angle), radius * np.sin(angle), color=BLACK, lw=0.8)
        else:
            ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color=BLACK, lw=0.8)
        if case == "stiffened":
            for fraction in (0.2, 0.4, 0.6, 0.8):
                ax.axvline(fraction, color=GRAY, lw=0.55, alpha=0.8)
                ax.axhline(fraction, color=GRAY, lw=0.55, alpha=0.8)
        ax.set_title(CASE_LABELS[case])
        ax.set_xlabel("x")
        if case == "flat":
            ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        style_axis(ax)
    _save(fig, args.out, "Fig05_structural_configurations")
    return _manifest_row("Fig05", "written", "case geometry builders")


def figure6_verification(args) -> dict:
    path = args.verification_csv
    if not path.exists():
        return _manifest_row("Fig06", "skipped", f"missing {path}")
    rows = _read_csv(path)
    required = {"case", "node_count", "mode", "rbf_frequency_hz", "reference_frequency_hz"}
    if not rows or not required.issubset(rows[0]):
        return _manifest_row("Fig06", "skipped", "verification CSV schema incomplete")
    fig, axes = plt.subplots(1, 2, figsize=(6.42, 2.65), constrained_layout=True)
    ax_conv, ax_err = axes
    for case in CASES:
        case_rows = [row for row in rows if row["case"] == case]
        if not case_rows:
            continue
        node_counts = sorted({int(float(row["node_count"])) for row in case_rows})
        for mode in sorted({int(float(row["mode"])) for row in case_rows})[:4]:
            mode_rows = sorted(
                [row for row in case_rows if int(float(row["mode"])) == mode],
                key=lambda row: float(row["node_count"]),
            )
            ax_conv.plot(
                _float(mode_rows, "node_count"),
                _float(mode_rows, "rbf_frequency_hz"),
                marker="o",
                ms=3,
                lw=0.8,
                color=CASE_COLORS[case],
                alpha=0.45 + 0.12 * min(mode, 4),
            )
        finest = max(node_counts)
        final_rows = [row for row in case_rows if int(float(row["node_count"])) == finest]
        error = np.abs(_float(final_rows, "rbf_frequency_hz") - _float(final_rows, "reference_frequency_hz")) / np.maximum(
            np.abs(_float(final_rows, "reference_frequency_hz")), 1e-30
        )
        ax_err.plot(
            [int(float(row["mode"])) for row in final_rows],
            100.0 * error,
            marker="o",
            lw=1.0,
            color=CASE_COLORS[case],
            label=CASE_LABELS[case],
        )
    ax_conv.set_xlabel("node count")
    ax_conv.set_ylabel("natural frequency (Hz)")
    ax_conv.set_title("RBF-FD convergence")
    ax_err.set_xlabel("mode")
    ax_err.set_ylabel("relative frequency error (%)")
    ax_err.set_title("Independent-reference error")
    ax_err.legend()
    for ax in axes:
        style_axis(ax)
    _save(fig, args.out, "Fig06_rbffd_independent_verification")
    return _manifest_row("Fig06", "written", str(path))


def figure7_parameter_effects(args) -> dict:
    case = args.representative_case
    path = args.parameter_effect_root / case / "parameter_effect_frf.csv"
    if not path.exists():
        return _manifest_row("Fig07", "skipped", f"missing {path}")
    rows = _read_csv(path)
    parameters = ["young_scale", "density_scale", "pressure_scale", "loss_factor"]
    titles = [r"$s_E$", r"$s_\rho$", r"$s_p$", r"$\eta$"]
    fig, axes = plt.subplots(2, 2, figsize=(6.42, 4.75), constrained_layout=True)
    palette = (BLUE, GRAY, BLACK, GRAY, RED)
    styles = ("--", ":", "-", "-.", "--")
    for ax, parameter, title in zip(axes.ravel(), parameters, titles):
        subset = [row for row in rows if row["parameter"] == parameter]
        levels = sorted({float(row["parameter_value"]) for row in subset})
        for idx, value in enumerate(levels):
            curve_rows = sorted(
                [row for row in subset if np.isclose(float(row["parameter_value"]), value)],
                key=lambda row: float(row["frequency_hz"]),
            )
            ax.plot(
                _float(curve_rows, "frequency_hz"),
                _float(curve_rows, "response_mm"),
                color=palette[min(idx, 4)],
                linestyle=styles[min(idx, 4)],
                lw=1.2 if idx == len(levels) // 2 else 0.9,
                label=f"{value:.4g}",
            )
        ax.set_title(title)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel(r"$|w|$ (mm)")
        ax.legend(fontsize=8, ncol=2, loc="lower right")
        style_axis(ax)
    _save(fig, args.out, "Fig07_single_parameter_physical_effects")
    return _manifest_row("Fig07", "written", str(path))


def figure8_unseen_errors(args) -> dict:
    paths = [args.ellipsoid_root / case / "ellipsoidal_unseen_test_metrics.csv" for case in CASES]
    if not all(path.exists() for path in paths):
        return _manifest_row("Fig08", "skipped", "missing ellipsoidal unseen-test metrics")
    metrics = [
        ("observation_curve_relative_l2", "FRF relative error"),
        ("full_response_relative_l2", "full-response relative error"),
        ("peak_frequency_error_hz", "peak-frequency error (Hz)"),
        ("peak_amplitude_relative_error", "peak-amplitude relative error"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.42, 4.6), constrained_layout=True)
    for ax, (key, ylabel) in zip(axes.ravel(), metrics):
        data = []
        labels = []
        for case, path in zip(CASES, paths):
            rows = _read_csv(path)
            data.append(_float(rows, key))
            labels.append(f"{CASE_LABELS[case]}\n($n={len(rows)}$)")
        bp = ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=False)
        for element in ("boxes", "whiskers", "caps", "medians"):
            for artist in bp[element]:
                artist.set_color(BLACK if element != "medians" else RED)
        ax.set_ylabel(ylabel)
        if key == "peak_frequency_error_hz" and all(np.allclose(values, 0.0) for values in data):
            ax.text(0.5, 0.62, "all curves: 0 Hz on the 1 Hz grid", transform=ax.transAxes, ha="center", fontsize=8)
        ax.tick_params(axis="x", labelrotation=15)
        style_axis(ax)
    _save(fig, args.out, "Fig08_unseen_parameter_error_distributions")
    return _manifest_row("Fig08", "written", "ellipsoidal unseen-test metrics")


def figure9_representative_frfs(args) -> dict:
    paths = [args.ellipsoid_root / case / "ellipsoidal_uq_curves.npz" for case in CASES]
    metric_paths = [args.ellipsoid_root / case / "ellipsoidal_unseen_test_metrics.csv" for case in CASES]
    if not all(path.exists() for path in paths + metric_paths):
        return _manifest_row("Fig09", "skipped", "missing unseen-test curve archives")
    fig, axes = plt.subplots(len(CASES), 2, figsize=(6.42, 4.7), constrained_layout=True)
    for row_index, case in enumerate(CASES):
        archive = np.load(paths[row_index])
        frequency = archive["frequency_hz"]
        hf = archive["test_hf_curves_mm"]
        pred = archive["test_surrogate_curves_mm"]
        metrics = _read_csv(metric_paths[row_index])
        errors = _float(metrics, "observation_curve_relative_l2")
        idx = int(np.argsort(errors)[len(errors) // 2])
        ax_full, ax_zoom = axes[row_index]
        ax_full.plot(frequency, hf[idx], color=BLACK, lw=1.25, label="RBF-FD HF")
        ax_full.plot(frequency, pred[idx], color=RED, lw=1.05, linestyle="--", label="surrogate")
        peak_idx = int(np.argmax(hf[idx]))
        peak_f = frequency[peak_idx]
        half = max(3.0, 0.04 * max(float(peak_f), 1.0))
        mask = (frequency >= peak_f - half) & (frequency <= peak_f + half)
        ax_zoom.plot(frequency[mask], hf[idx][mask], color=BLACK, lw=1.25)
        ax_zoom.plot(frequency[mask], pred[idx][mask], color=RED, lw=1.05, linestyle="--")
        ax_full.set_ylabel(f"{CASE_LABELS[case]}\n$|w|$ (mm)")
        ax_full.set_xlabel("frequency (Hz)")
        ax_zoom.set_xlabel("frequency (Hz)")
        if row_index == 0:
            ax_full.set_title("Complete FRF")
            ax_zoom.set_title("Dominant-resonance zoom")
            ax_full.legend(loc="best")
        style_axis(ax_full)
        style_axis(ax_zoom)
    _save(fig, args.out, "Fig09_representative_unseen_frfs")
    return _manifest_row("Fig09", "written", "ellipsoidal unseen-test archives")


def figure10_ablation(args) -> dict:
    variants = ("direct_data", "direct_physics", "pole_data", "full")
    label_map = {
        "direct_data": "Direct-data",
        "direct_physics": "Direct-physics",
        "pole_data": "Pole-data",
        "full": "Full",
    }
    records = {}
    for variant in variants:
        records[variant] = {}
        for case in CASES:
            path = args.ablation_root / variant / case / "high_fidelity_statistical_comparison.json"
            if not path.exists():
                return _manifest_row("Fig10", "skipped", f"missing ablation output {path}")
            records[variant][case] = read_json(path)
    metrics = [
        ("mean_unseen_full_field_error", "mean unseen full-field error"),
        ("mean_peak_frequency_error_hz", "mean peak-frequency error (Hz)"),
        ("upper_boundary_relative_l2", "upper-boundary relative error"),
        ("unseen_test_curve_coverage", "curve-wise coverage"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.42, 4.7), constrained_layout=True)
    x = np.arange(len(variants))
    width = 0.22
    for ax, (key, ylabel) in zip(axes.ravel(), metrics):
        for case_index, case in enumerate(CASES):
            values = [float(records[variant][case][key]) for variant in variants]
            ax.bar(
                x + (case_index - 0.5 * (len(CASES) - 1)) * width,
                values,
                width=width,
                edgecolor=CASE_COLORS[case],
                facecolor="none",
                linewidth=1.0,
                label=CASE_LABELS[case],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([label_map[v] for v in variants], rotation=18)
        ax.set_ylabel(ylabel)
        style_axis(ax)
    axes[0, 0].legend(fontsize=8)
    _save(fig, args.out, "Fig10_four_model_ablation")
    return _manifest_row("Fig10", "written", str(args.ablation_root))


def figure11_ellipsoid_envelopes(args) -> dict:
    paths = [args.ellipsoid_root / case / "ellipsoidal_envelope.csv" for case in CASES]
    archives = [args.ellipsoid_root / case / "ellipsoidal_uq_curves.npz" for case in CASES]
    if not all(path.exists() for path in paths + archives):
        return _manifest_row("Fig11", "skipped", "missing ellipsoidal envelope outputs")
    fig, axes = plt.subplots(len(CASES), 1, figsize=(6.42, 4.6), constrained_layout=True)
    for ax, case, path, archive_path in zip(axes, CASES, paths, archives):
        rows = _read_csv(path)
        f = _float(rows, "frequency_hz")
        archive = np.load(archive_path)
        curves = archive["surrogate_curves_mm"]
        for curve in curves[: min(curves.shape[0], 80)]:
            ax.plot(f, curve, color=GRAY, lw=0.35, alpha=0.18)
        ax.plot(f, _float(rows, "hf_lower_mm"), color=BLACK, lw=1.0, linestyle="--", label="HF sample bounds")
        ax.plot(f, _float(rows, "hf_upper_mm"), color=BLACK, lw=1.0, linestyle="--")
        ax.plot(f, _float(rows, "surrogate_lower_mm"), color=RED, lw=1.2, label="surrogate ellipsoidal bounds")
        ax.plot(f, _float(rows, "surrogate_upper_mm"), color=RED, lw=1.2)
        ax.set_ylabel(f"{CASE_LABELS[case]}\n$|w|$ (mm)")
        ax.set_xlabel("frequency (Hz)")
        if case == "flat":
            ax.legend(fontsize=8, ncol=2)
        style_axis(ax)
    _save(fig, args.out, "Fig11_ellipsoidal_frf_envelopes")
    return _manifest_row("Fig11", "written", str(args.ellipsoid_root))


def figure12_resonance_ranges(args) -> dict:
    paths = [args.ellipsoid_root / case / "ellipsoidal_resonance_ranges.csv" for case in CASES]
    if not all(path.exists() for path in paths):
        return _manifest_row("Fig12", "skipped", "missing resonance range tables")
    fig, axes = plt.subplots(1, 2, figsize=(6.42, 2.8), constrained_layout=True)
    for case_index, (case, path) in enumerate(zip(CASES, paths)):
        rows = _read_csv(path)
        if not rows:
            continue
        mode = np.asarray([int(row["mode"]) for row in rows])
        f_lo, f_up = _float(rows, "frequency_lower_hz"), _float(rows, "frequency_upper_hz")
        a_lo, a_up = _float(rows, "amplitude_lower_mm"), _float(rows, "amplitude_upper_mm")
        offset = (case_index - 0.5 * (len(CASES) - 1)) * 0.16
        axes[0].errorbar(
            mode + offset,
            0.5 * (f_lo + f_up),
            yerr=0.5 * (f_up - f_lo),
            fmt="o",
            ms=4,
            capsize=2.5,
            color=CASE_COLORS[case],
            label=CASE_LABELS[case],
        )
        axes[1].errorbar(
            mode + offset,
            0.5 * (a_lo + a_up),
            yerr=0.5 * (a_up - a_lo),
            fmt="o",
            ms=4,
            capsize=2.5,
            color=CASE_COLORS[case],
        )
    axes[0].set_xlabel("mode")
    axes[0].set_ylabel("resonance frequency (Hz)")
    axes[0].set_title(r"$\mathcal{F}_r$")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("mode")
    axes[1].set_ylabel("peak amplitude (mm, log scale)")
    axes[1].set_yscale("log")
    axes[1].set_title(r"$\mathcal{A}_r$")
    for ax in axes:
        style_axis(ax)
    _save(fig, args.out, "Fig12_resonance_frequency_amplitude_ranges")
    return _manifest_row("Fig12", "written", "ellipsoidal resonance tables")


def figure13_dependence_convergence(args) -> dict:
    case = args.representative_case
    corr_path = args.ellipsoid_root / case / "ellipsoidal_correlation_envelopes.csv"
    conv_path = args.ellipsoid_root / case / "ellipsoidal_sample_convergence.csv"
    if not corr_path.exists() or not conv_path.exists():
        return _manifest_row("Fig13", "skipped", "missing correlation/convergence studies")
    corr_rows = _read_csv(corr_path)
    conv_rows = _read_csv(conv_path)
    fig, axes = plt.subplots(1, 2, figsize=(6.42, 2.8), constrained_layout=True)
    correlations = sorted({float(row["e_rho_correlation"]) for row in corr_rows})
    corr_colors = (BLUE, BLACK, RED, GRAY)
    for idx, corr in enumerate(correlations):
        rows = sorted(
            [row for row in corr_rows if np.isclose(float(row["e_rho_correlation"]), corr)],
            key=lambda row: float(row["frequency_hz"]),
        )
        f = _float(rows, "frequency_hz")
        axes[0].plot(f, _float(rows, "width_mm"), color=corr_colors[min(idx, 3)], lw=1.0, label=fr"$\rho_{{E,\rho}}={corr:g}$")
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("envelope width (mm)")
    axes[0].set_title("Effect of ellipsoid orientation")
    axes[0].legend(fontsize=8)
    n = _float(conv_rows, "sample_count")
    # The full-sample row is the reference and is exactly zero by definition;
    # exclude it from the log-valued error curves so it does not compress all
    # of the informative pre-asymptotic values by many orders of magnitude.
    full_count = float(np.max(n))
    mask = n < full_count
    axes[1].plot(n[mask], _float(conv_rows, "lower_relative_l2_vs_full")[mask], marker="o", color=BLUE, lw=1.0, label="lower")
    axes[1].plot(n[mask], _float(conv_rows, "upper_relative_l2_vs_full")[mask], marker="s", color=RED, lw=1.0, label="upper")
    axes[1].plot(n[mask], _float(conv_rows, "width_relative_l2_vs_full")[mask], marker="^", color=BLACK, lw=1.0, label="width")
    axes[1].axvline(full_count, color=GRAY, linestyle="--", lw=0.8)
    axes[1].text(full_count, 0.012, "full-sample reference", rotation=90, ha="right", va="bottom", fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$N_{\mathrm{UQ}}$")
    axes[1].set_ylabel("relative difference vs full sample")
    axes[1].set_title("Sample-estimated envelope convergence")
    axes[1].legend(fontsize=8)
    for ax in axes:
        style_axis(ax)
    _save(fig, args.out, "Fig13_ellipsoid_dependence_and_sample_convergence")
    return _manifest_row("Fig13", "written", f"{corr_path}; {conv_path}")


def figure14_calibration(args) -> dict:
    case_paths = {
        case: (
            args.ellipsoid_root / case / "ellipsoidal_calibration_example.csv",
            args.ellipsoid_root / case / "ellipsoidal_calibration_reliability.csv",
        )
        for case in CASES
    }
    if not all(path.exists() for paths in case_paths.values() for path in paths):
        return _manifest_row("Fig14", "skipped", "missing ellipsoidal calibration outputs")
    fig, axes = plt.subplots(2, 2, figsize=(6.42, 5.25), constrained_layout=True)
    sources = []
    for row_index, case in enumerate(CASES):
        example_path, reliability_path = case_paths[case]
        sources.extend((str(example_path), str(reliability_path)))
        example = _read_csv(example_path)
        reliability = _read_csv(reliability_path)
        band_ax, coverage_ax = axes[row_index]
        f = _float(example, "frequency_hz")
        band_ax.fill_between(
            f,
            _float(example, "calibrated_lower_mm"),
            _float(example, "calibrated_upper_mm"),
            color=BLUE,
            alpha=0.18,
            label=r"$\mathcal{I}_{\rm surr}$",
        )
        band_ax.plot(f, _float(example, "ensemble_mean_mm"), color=RED, lw=1.1, label="ensemble mean")
        band_ax.plot(f, _float(example, "hf_mm"), color=BLACK, lw=1.0, linestyle="--", label="HF")
        band_ax.set_xlabel("frequency (Hz)")
        band_ax.set_ylabel(r"$|w|$ (mm)")
        band_ax.set_title(f"{CASE_LABELS[case]}: representative 95% band")
        band_ax.legend(fontsize=7.2)

        nominal = _float(reliability, "nominal_curve_coverage")
        empirical = _float(reliability, "test_empirical_curve_coverage")
        coverage_ax.plot([0.70, 1.0], [0.70, 1.0], color=GRAY, linestyle="--", lw=0.9)
        coverage_ax.plot(nominal, empirical, color=CASE_COLORS[case], marker="o", lw=1.1)
        for x, y, record in zip(nominal, empirical, reliability):
            label = f'{int(float(record["covered_test_curves"]))}/{int(float(record["test_curve_count"]))}'
            coverage_ax.annotate(label, (x, y), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=7.2)
        coverage_ax.set_xlim(0.75, 1.0)
        coverage_ax.set_ylim(0.70, 1.01)
        coverage_ax.set_xlabel("nominal curve-wise coverage")
        coverage_ax.set_ylabel("empirical test coverage")
        coverage_ax.set_title(f"{CASE_LABELS[case]}: calibration reliability")
        style_axis(band_ax)
        style_axis(coverage_ax)
    _save(fig, args.out, "Fig14_surrogate_error_calibration")
    return _manifest_row("Fig14", "written", "; ".join(sources))


def figure15_cost(args) -> dict:
    costs = {}
    for case in CASES:
        path = args.parametric_root / case / "cost_and_break_even.json"
        if path.exists():
            costs[case] = read_json(path)
    if not costs:
        return _manifest_row("Fig15", "skipped", f"no cost files under {args.parametric_root}")
    max_be = max([cost["break_even_parameter_queries"] or 1 for cost in costs.values()])
    n = np.arange(0, max(100, int(max_be * 2.5)) + 1)
    fig, ax = plt.subplots(figsize=(6.42, 3.4), constrained_layout=True)
    for case in costs:
        cost = costs[case]
        hf = n * float(cost["direct_full_sweep_per_parameter_s"])
        surrogate = float(cost["offline_total_s"]) + n * float(cost["operator_ensemble_full_sweep_per_parameter_s"])
        color = CASE_COLORS[case]
        ax.plot(n, hf / 3600.0, color=color, lw=1.0, label=f"{CASE_LABELS[case]} HF")
        ax.plot(n, surrogate / 3600.0, color=color, lw=1.2, linestyle="--", label=f"{CASE_LABELS[case]} surrogate")
        if cost["break_even_parameter_queries"] is not None:
            be = int(cost["break_even_parameter_queries"])
            ax.axvline(be, color=color, lw=0.55, alpha=0.35)
    ax.set_xlabel("number of complete parameter queries")
    ax.set_ylabel("cumulative cost (h)")
    ax.set_title("Measured offline-online break-even")
    ax.legend(fontsize=8, ncol=2)
    style_axis(ax)
    _save(fig, args.out, "Fig15_cumulative_cost_break_even")
    missing = [case for case in CASES if case not in costs]
    source = str(args.parametric_root)
    if missing:
        source += f"; omitted cases without completed cost output: {', '.join(missing)}"
    return _manifest_row("Fig15", "written", source)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    apply_journal_style()
    builders = (
        figure5_geometry,
        figure6_verification,
        figure7_parameter_effects,
        figure8_unseen_errors,
        figure9_representative_frfs,
        figure10_ablation,
        figure11_ellipsoid_envelopes,
        figure12_resonance_ranges,
        figure13_dependence_convergence,
        figure14_calibration,
        figure15_cost,
    )
    manifest = []
    for builder in builders:
        try:
            manifest.append(builder(args))
        except Exception as exc:
            manifest.append(_manifest_row(builder.__name__, "failed", repr(exc)))
    with (args.out / "figure_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("figure", "status", "source"))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Wrote/checked result figures in {args.out}")
    for row in manifest:
        print(row)


if __name__ == "__main__":
    main()

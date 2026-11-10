# -*- coding: utf-8 -*-
"""Regenerate review-relevant figures with the retained current checkpoints.

Figures written (same names as the manuscript's result assets):
    Fig07_single_parameter_physical_effects.pdf   legend below panels
    Fig08_unseen_parameter_error_distributions.pdf  current 40-curve metrics
                                                    + 8-curve full-response
    Fig09_representative_unseen_frfs.pdf          median + worst-case curves
    Fig11_ellipsoidal_frf_envelopes.pdf           1000-curve current envelope
    Fig12_resonance_frequency_amplitude_ranges.pdf tracker + network ranges,
                                                    extrapolation shading
    Fig13_ellipsoid_dependence_and_sample_convergence.pdf
    Fig14_surrogate_error_calibration.pdf         current conformal bands

All surrogate predictions are computed on CPU with two worker threads; direct
RBF-FD reference curves are reused from the stored archives (matched 8-point
sets and the 40+40 calibration/test archive).  No new direct solves are
performed by this script except none: full-field spot-check references are
read from the phase-7 archives.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
import torch

from rbf_fd_full_order import (
    ParameterBounds,
    predict_parametric_members,
    predict_parametric_observation_members,
)
from rbf_fd_full_order.uncertainty import (
    CurveConformalCalibrator,
    EllipsoidalConvexSet,
    correlation_matrix,
    empirical_curve_coverage,
)
from rbf_fd_full_order.visualization import apply_journal_style, save_figure, style_axis
from rbf_fd_full_order.visualization.result_support import load_parametric_case


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"
OUT = ROOT / "manuscript" / "current" / "result_assets" / "figures" / "results_section"
CASES = ("flat", "stiffened")
CASE_LABELS = {"flat": "Unstiffened plate", "stiffened": "Stiffened plate"}
CASE_COLORS = {"flat": "#000000", "stiffened": "#C00000"}
RED = "#C00000"
BLACK = "#000000"
BLUE = "#0070C0"
GRAY = "#808080"


def _bounds() -> ParameterBounds:
    return ParameterBounds(
        lower=np.asarray([0.85, 0.90, 0.80, 0.005], dtype=float),
        upper=np.asarray([1.15, 1.10, 1.20, 0.020], dtype=float),
    ).validated()


def _ellipsoid(bounds: ParameterBounds, e_rho: float) -> EllipsoidalConvexSet:
    return EllipsoidalConvexSet(
        center=np.asarray(bounds.center, dtype=float),
        scales=0.5 * (np.asarray(bounds.upper, dtype=float) - np.asarray(bounds.lower, dtype=float)),
        correlation=correlation_matrix(bounds.dimension, e_rho=float(e_rho)),
    ).validated()


def _load(case: str, device: str = "cpu"):
    case_dir = RESULTS / f"parametric_many_query_{case}_5000_best" / case
    return load_parametric_case(case_dir, device=device)


def _obs(result, frequencies: np.ndarray, parameters: np.ndarray, observation_index: int) -> np.ndarray:
    members = predict_parametric_observation_members(result, frequencies, parameters, observation_index)
    return np.mean(np.abs(members), axis=0) * 1.0e3


def _parabolic_peak(frequencies: np.ndarray, curve: np.ndarray) -> float:
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


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - reference) / max(float(np.linalg.norm(reference)), 1.0e-30))


def _uq1000(case: str, result, observation_index: int) -> dict[str, np.ndarray]:
    cache = RESULTS / "unseen_accuracy_current" / f"{case}_current_uq1000.npz"
    bounds = _bounds()
    ellipsoid = _ellipsoid(bounds, 0.0)
    parameters = ellipsoid.sample(1000, 712, boundary_fraction=0.5)
    frequencies = np.linspace(0.0, 200.0, 201)
    if cache.exists():
        data = np.load(cache)
        if int(data["observation_index"]) == observation_index:
            return {
                "frequencies": data["frequencies"],
                "parameters": data["parameters"],
                "curves": data["curves"],
            }
    curves = _obs(result, frequencies, parameters, observation_index)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        frequencies=frequencies,
        parameters=parameters,
        curves=curves,
        observation_index=np.asarray(observation_index),
    )
    return {"frequencies": frequencies, "parameters": parameters, "curves": curves}


def fig07_parameter_effects() -> None:
    path = RESULTS / "parameter_effects" / "stiffened" / "parameter_effect_frf.csv"
    if not path.exists():
        print("fig07 skipped: missing parameter-effect CSV", flush=True)
        return
    rows = list(__import__("csv").DictReader(path.open("r", encoding="utf-8", newline="")))
    parameters = ["young_scale", "density_scale", "pressure_scale", "loss_factor"]
    titles = [r"$s_E$", r"$s_\rho$", r"$s_p$", r"$\eta$"]
    palette = (BLUE, GRAY, BLACK, GRAY, RED)
    styles = ("--", ":", "-", "-.", "--")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for ax, parameter, title in zip(axes.ravel(), parameters, titles):
        subset = [row for row in rows if row["parameter"] == parameter]
        levels = sorted({float(row["parameter_value"]) for row in subset})
        for idx, value in enumerate(levels):
            curve_rows = sorted(
                [row for row in subset if np.isclose(float(row["parameter_value"]), value)],
                key=lambda row: float(row["frequency_hz"]),
            )
            ax.plot(
                [float(row["frequency_hz"]) for row in curve_rows],
                [float(row["response_mm"]) for row in curve_rows],
                color=palette[min(idx, 4)],
                linestyle=styles[min(idx, 4)],
                lw=1.1,
                label=f"{value:.4g}",
            )
        ax.set_title(title)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel(r"$|w|$ (mm)")
        ax.legend(fontsize=7, ncol=min(len(levels), 3), loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), frameon=False)
        style_axis(ax)
    save_figure(fig, OUT / "Fig07_single_parameter_physical_effects.pdf", tight=False)
    plt.close(fig)
    print("fig07 written", flush=True)


def fig08_unseen_errors() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    panels = [
        ("observation", "Observation-FRF relative error"),
        ("full", "Full-response relative error (8-curve spot check)"),
        ("peak_frequency", "Peak-frequency error (Hz, 1 Hz grid)"),
        ("peak_amplitude", "Peak-amplitude relative error"),
    ]
    for ax, (key, ylabel) in zip(axes.ravel(), panels):
        data = []
        labels = []
        for case in CASES:
            metadata, _, _, _, result, observation_index = _load(case)
            archive = np.load(RESULTS / "ellipsoidal_uq" / case / "ellipsoidal_calibration_40x40.npz")
            frequencies = np.asarray(archive["frequencies_hz"], dtype=float)
            parameters = np.asarray(archive["test_parameters"], dtype=float)
            reference = np.asarray(archive["test_hf_mm"], dtype=float)
            predicted = _obs(result, frequencies, parameters, observation_index)
            if key == "observation":
                values = np.asarray(
                    [_relative_l2(predicted[i], reference[i]) for i in range(parameters.shape[0])]
                )
            elif key == "peak_frequency":
                values = np.asarray(
                    [
                        abs(float(frequencies[np.argmax(predicted[i])]) - float(frequencies[np.argmax(reference[i])]))
                        for i in range(parameters.shape[0])
                    ]
                )
            elif key == "peak_amplitude":
                values = np.asarray(
                    [
                        abs(float(np.max(predicted[i]) - np.max(reference[i])))
                        / max(float(np.max(reference[i])), 1.0e-30)
                        for i in range(parameters.shape[0])
                    ]
                )
            else:
                spot = RESULTS / "phase7" / f"{case}_test_reference.npz"
                if not spot.exists():
                    values = np.asarray([np.nan])
                    data.append(values)
                    labels.append(f"{CASE_LABELS[case]} (pending)")
                    continue
                else:
                    spot_data = np.load(spot)
                    spot_params = np.asarray(spot_data["test_parameters"], dtype=float)
                    spot_ref = np.asarray(spot_data["test_reference_full"], dtype=complex)
                    pred_full = np.mean(
                        predict_parametric_members(result, frequencies, spot_params), axis=0
                    )
                    values = np.asarray(
                        [_relative_l2(pred_full[i], spot_ref[i]) for i in range(spot_params.shape[0])]
                    )
            data.append(values)
            labels.append(f"{CASE_LABELS[case]}\n($n={values.size}$)")
            finite = [values for values in data if values.size and not np.all(np.isnan(values))]
            bp = ax.boxplot(finite, tick_labels=labels[: len(finite)], showfliers=True, patch_artist=False)
        for element in ("boxes", "whiskers", "caps", "medians"):
            for artist in bp[element]:
                artist.set_color(BLACK if element != "medians" else RED)
        ax.set_ylabel(ylabel)
        if key == "peak_frequency":
            ax.text(0.5, 0.62, "all curves: 0 Hz on the 1 Hz grid",
                    transform=ax.transAxes, ha="center", fontsize=8)
        ax.tick_params(axis="x", labelrotation=12)
        style_axis(ax)
    save_figure(fig, OUT / "Fig08_unseen_parameter_error_distributions.pdf", tight=False)
    plt.close(fig)
    print("fig08 written", flush=True)


def fig09_representative_frfs() -> None:
    fig, axes = plt.subplots(len(CASES), 2, figsize=(7.2, 5.2), constrained_layout=True)
    for row_index, case in enumerate(CASES):
        metadata, _, _, _, result, observation_index = _load(case)
        archive = np.load(RESULTS / "ellipsoidal_uq" / case / "ellipsoidal_calibration_40x40.npz")
        frequencies = np.asarray(archive["frequencies_hz"], dtype=float)
        parameters = np.asarray(archive["test_parameters"], dtype=float)
        reference = np.asarray(archive["test_hf_mm"], dtype=float)
        predicted = _obs(result, frequencies, parameters, observation_index)
        errors = np.asarray(
            [_relative_l2(predicted[i], reference[i]) for i in range(parameters.shape[0])]
        )
        median_idx = int(np.argsort(errors)[len(errors) // 2])
        worst_idx = int(np.argsort(errors)[-1])
        ax_full, ax_zoom = axes[row_index]
        for idx, color, style, label in (
            (median_idx, BLACK, "-", "direct (median)"),
            (worst_idx, BLACK, ":", "direct (worst)"),
        ):
            ax_full.plot(frequencies, reference[idx], color=color, lw=1.2, linestyle=style, label=label)
            ax_full.plot(frequencies, predicted[idx], color=RED, lw=1.0,
                         linestyle="--" if style == "-" else "-.", alpha=0.85)
        peak_idx = int(np.argmax(reference[median_idx]))
        peak_f = frequencies[peak_idx]
        half = max(3.0, 0.04 * float(peak_f))
        mask = (frequencies >= peak_f - half) & (frequencies <= peak_f + half)
        ax_zoom.plot(frequencies[mask], reference[median_idx][mask], color=BLACK, lw=1.3)
        ax_zoom.plot(frequencies[mask], predicted[median_idx][mask], color=RED, lw=1.1, linestyle="--")
        ax_full.set_ylabel(f"{CASE_LABELS[case]}\n$|w|$ (mm)")
        ax_full.set_xlabel("frequency (Hz)")
        ax_zoom.set_xlabel("frequency (Hz)")
        if row_index == 0:
            ax_full.set_title("Complete FRF")
            ax_zoom.set_title("Dominant-resonance zoom")
            ax_full.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.16),
                           ncol=2, frameon=False)
        style_axis(ax_full)
        style_axis(ax_zoom)
    save_figure(fig, OUT / "Fig09_representative_unseen_frfs.pdf", tight=False)
    plt.close(fig)
    print("fig09 written", flush=True)


def fig11_ellipsoid_envelopes() -> None:
    fig, axes = plt.subplots(len(CASES), 1, figsize=(7.2, 5.0), constrained_layout=True)
    for ax, case in zip(axes, CASES):
        _, _, _, _, result, observation_index = _load(case)
        data = _uq1000(case, result, observation_index)
        frequencies = data["frequencies"]
        curves = data["curves"]
        lower = np.min(curves, axis=0)
        upper = np.max(curves, axis=0)
        archive = np.load(RESULTS / "ellipsoidal_uq" / case / "ellipsoidal_uq_curves.npz")
        hf = np.asarray(archive["hf_curves_mm"], dtype=float)
        hf_lower = np.min(hf, axis=0)
        hf_upper = np.max(hf, axis=0)
        for curve in curves[: min(curves.shape[0], 80)]:
            ax.plot(frequencies, curve, color=GRAY, lw=0.35, alpha=0.18)
        ax.plot(frequencies, hf_lower, color=BLACK, lw=1.0, linestyle="--", label="8 direct RBF-FD bounds")
        ax.plot(frequencies, hf_upper, color=BLACK, lw=1.0, linestyle="--")
        ax.plot(frequencies, lower, color=RED, lw=1.2, label="1000-curve surrogate bounds")
        ax.plot(frequencies, upper, color=RED, lw=1.2)
        ax.set_ylabel(f"{CASE_LABELS[case]}\n$|w|$ (mm)")
        ax.set_xlabel("frequency (Hz)")
        if case == "flat":
            ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
        style_axis(ax)
    save_figure(fig, OUT / "Fig11_ellipsoidal_frf_envelopes.pdf", tight=False)
    plt.close(fig)
    print("fig11 written", flush=True)


def fig12_resonance_ranges() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    for case in CASES:
        _, _, _, _, result, observation_index = _load(case)
        data = _uq1000(case, result, observation_index)
        frequencies = data["frequencies"]
        curves = data["curves"]
        parameters = data["parameters"]
        peak_freqs = np.asarray([_parabolic_peak(frequencies, curves[i]) for i in range(curves.shape[0])])
        peak_amps = np.asarray([float(np.max(curves[i])) for i in range(curves.shape[0])])
        offset = 0.16 if case == "flat" else -0.16
        x = 1 + offset
        axes[0].errorbar(
            [x], [0.5 * (float(np.min(peak_freqs)) + float(np.max(peak_freqs)))],
            yerr=[0.5 * (float(np.max(peak_freqs)) - float(np.min(peak_freqs)))],
            fmt="o", ms=4, capsize=2.5, color=CASE_COLORS[case], label=CASE_LABELS[case],
        )
        axes[1].errorbar(
            [x], [0.5 * (float(np.min(peak_amps)) + float(np.max(peak_amps)))],
            yerr=[0.5 * (float(np.max(peak_amps)) - float(np.min(peak_amps)))],
            fmt="o", ms=4, capsize=2.5, color=CASE_COLORS[case],
        )
        if case == "stiffened":
            eigenvalues, _ = spla.eigsh(
                result.family.nominal.K, k=12, M=result.family.nominal.M,
                sigma=0.0, which="LM",
            )
            f0 = np.sqrt(np.sort(eigenvalues)) / (2.0 * np.pi)
            s_e = parameters[:, 0]
            s_rho = parameters[:, 1]
            for mode in (2, 3, 4):
                f_nominal = f0[mode - 1]
                tracker = f_nominal * np.sqrt(s_e / s_rho)
                local = np.linspace(float(np.min(tracker)) - 6.0, float(np.max(tracker)) + 6.0, 61)
                local_curves = _obs(result, local, parameters, observation_index)
                local_amps = np.asarray([float(np.max(local_curves[i])) for i in range(parameters.shape[0])])
                xm = mode + (-0.16 if case == "flat" else 0.16)
                axes[0].errorbar(
                    [xm], [0.5 * (float(np.min(tracker)) + float(np.max(tracker)))],
                    yerr=[0.5 * (float(np.max(tracker)) - float(np.min(tracker)))],
                    fmt="s", ms=4, capsize=2.5, color=CASE_COLORS[case],
                )
                axes[1].errorbar(
                    [xm], [0.5 * (float(np.min(local_amps)) + float(np.max(local_amps)))],
                    yerr=[0.5 * (float(np.max(local_amps)) - float(np.min(local_amps)))],
                    fmt="s", ms=4, capsize=2.5, color=CASE_COLORS[case],
                )
    axes[0].axvspan(2.5, 4.5, color=GRAY, alpha=0.10)
    axes[1].axvspan(2.5, 4.5, color=GRAY, alpha=0.10)
    axes[0].set_xlabel("mode")
    axes[0].set_ylabel("resonance frequency (Hz)")
    axes[0].set_title(r"$\mathcal{F}_r$")
    axes[0].legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.9)
    axes[1].set_xlabel("mode")
    axes[1].set_ylabel("peak amplitude (mm, log scale)")
    axes[1].set_yscale("log")
    axes[1].set_title(r"$\mathcal{A}_r$")
    ymax = axes[0].get_ylim()[1]
    axes[0].text(3.5, 0.96 * ymax, ">200 Hz extrapolation", ha="center", fontsize=7, color="#404040")
    for ax in axes:
        style_axis(ax)
    save_figure(fig, OUT / "Fig12_resonance_frequency_amplitude_ranges.pdf", tight=False)
    plt.close(fig)
    print("fig12 written", flush=True)


def fig13_dependence_convergence() -> None:
    case = "stiffened"
    _, _, _, _, result, observation_index = _load(case)
    bounds = _bounds()
    frequencies = np.linspace(0.0, 200.0, 201)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    corr_colors = (BLUE, BLACK, RED, GRAY)
    for corr_index, corr in enumerate((-0.5, 0.0, 0.5)):
        ellipsoid = _ellipsoid(bounds, corr)
        parameters = ellipsoid.sample(500, 701 + 1000 + corr_index, boundary_fraction=0.5)
        curves = _obs(result, frequencies, parameters, observation_index)
        width = np.max(curves, axis=0) - np.min(curves, axis=0)
        axes[0].plot(frequencies, width, color=corr_colors[corr_index], lw=1.0,
                     label=fr"$\rho_{{E,\rho}}={corr:g}$")
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("envelope width (mm)")
    axes[0].set_title("Effect of ellipsoid orientation")
    axes[0].legend(fontsize=8)

    data = np.load(RESULTS / "unseen_accuracy_current" / "stiffened_uq_large_convergence_curves.npz")
    curves = data["curves"]
    frequencies = data["frequencies"]
    counts = np.asarray([25, 50, 100, 200, 300, 500, 750, 1000, 2000, 5000, 10000])
    extremum = json.loads(
        (RESULTS / "ellipsoid_extremum_search" / "stiffened" / "extremum_summary.json").read_text("utf-8")
    )
    f1_index = int(np.argmin(np.abs(frequencies - float(extremum["target_frequency_hz"]))))
    search_lower = float(extremum["refined_targets"]["envelope_lower_at_f1"]["surrogate_value"])
    search_upper = float(extremum["refined_targets"]["envelope_upper_at_f1"]["surrogate_value"])
    lower_gaps = []
    upper_gaps = []
    for count in counts:
        lower = float(np.min(curves[:count, f1_index]))
        upper = float(np.max(curves[:count, f1_index]))
        lower_gaps.append(abs(lower - search_lower) / search_lower)
        upper_gaps.append(abs(search_upper - upper) / search_upper)
    axes[1].semilogx(counts, lower_gaps, marker="o", color=BLUE, lw=1.0, label="lower bound")
    axes[1].semilogx(counts, upper_gaps, marker="s", color=RED, lw=1.0, label="upper envelope")
    axes[1].set_xlabel("sample count $N_{\\mathrm{UQ}}$")
    axes[1].set_ylabel("gap to constrained-search estimate")
    axes[1].set_title("Sample convergence")
    axes[1].legend(fontsize=8)
    for ax in axes:
        style_axis(ax)
    save_figure(fig, OUT / "Fig13_ellipsoid_dependence_and_sample_convergence.pdf", tight=False)
    plt.close(fig)
    print("fig13 written", flush=True)


def fig14_calibration() -> None:
    fig, axes = plt.subplots(len(CASES), 2, figsize=(7.2, 5.0), constrained_layout=True)
    for row_index, case in enumerate(CASES):
        _, _, _, _, result, observation_index = _load(case)
        archive = np.load(RESULTS / "ellipsoidal_uq" / case / "ellipsoidal_calibration_40x40.npz")
        frequencies = np.asarray(archive["frequencies_hz"], dtype=float)
        calibration_parameters = np.asarray(archive["calibration_parameters"], dtype=float)
        test_parameters = np.asarray(archive["test_parameters"], dtype=float)
        calibration_hf = np.asarray(archive["calibration_hf_mm"], dtype=float)
        test_hf = np.asarray(archive["test_hf_mm"], dtype=float)
        calibration_members = predict_parametric_observation_members(
            result, frequencies, calibration_parameters, observation_index
        )
        calibration_members = np.abs(calibration_members) * 1.0e3
        test_members = predict_parametric_observation_members(
            result, frequencies, test_parameters, observation_index
        )
        test_members = np.abs(test_members) * 1.0e3
        test_mean = np.mean(test_members, axis=0)
        errors = np.asarray(
            [_relative_l2(test_mean[i], test_hf[i]) for i in range(test_parameters.shape[0])]
        )
        representative = int(np.argsort(errors)[len(errors) // 2])
        ax_band, ax_coverage = axes[row_index]
        calibrator = CurveConformalCalibrator.fit(
            calibration_members, calibration_hf, confidence=0.95, relative_scale_floor=0.02
        )
        mean, spread, lower, upper = calibrator.band(test_members[:, representative, :])
        peak_idx = int(np.argmax(test_hf[representative]))
        peak_f = frequencies[peak_idx]
        half = max(4.0, 0.05 * float(peak_f))
        mask = (frequencies >= peak_f - half) & (frequencies <= peak_f + half)
        ax_band.fill_between(frequencies[mask], lower[mask], upper[mask], color=BLUE, alpha=0.28,
                             label="95% conformal band")
        ax_band.plot(frequencies[mask], mean[mask], color=RED, lw=1.1, label="ensemble mean")
        ax_band.plot(frequencies[mask], test_hf[representative][mask], color=BLACK, lw=1.2,
                     linestyle="--", label="direct RBF-FD")
        ax_band.set_ylabel(f"{CASE_LABELS[case]}\n$|w|$ (mm)")
        ax_band.set_xlabel("frequency (Hz)")
        ax_band.set_title("Dominant-resonance calibration")
        ax_band.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.9)

        for confidence, color in ((0.80, BLUE), (0.90, BLACK), (0.95, RED)):
            fit = CurveConformalCalibrator.fit(
                calibration_members, calibration_hf, confidence=confidence, relative_scale_floor=0.02
            )
            _, _, lo, up = fit.band(test_members)
            inside = (test_hf >= lo) & (test_hf <= up)
            fraction = np.mean(inside, axis=0)
            covered = int(round(empirical_curve_coverage(test_hf, lo, up) * test_parameters.shape[0]))
            ax_coverage.plot(frequencies, fraction, color=color, lw=1.1,
                             label=f"{int(confidence*100)}%: {covered}/40")
        ax_coverage.axhline(0.95, color=GRAY, linestyle=":", lw=1.0)
        ax_coverage.set_ylim(0.0, 1.02)
        ax_coverage.set_xlabel("frequency (Hz)")
        ax_coverage.set_ylabel("covered-curve fraction")
        ax_coverage.set_title("Curve-wise coverage")
        ax_coverage.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
        style_axis(ax_band)
        style_axis(ax_coverage)
    save_figure(fig, OUT / "Fig14_surrogate_error_calibration.pdf", tight=False)
    plt.close(fig)
    print("fig14 written", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    apply_journal_style()
    fig07_parameter_effects()
    fig08_unseen_errors()
    fig09_representative_frfs()
    fig11_ellipsoid_envelopes()
    fig12_resonance_ranges()
    fig13_dependence_convergence()
    fig14_calibration()
    print("All review figures regenerated.", flush=True)

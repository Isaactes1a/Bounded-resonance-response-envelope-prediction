# -*- coding: utf-8 -*-
"""Classical modal / MOR / interpolation baselines against the RBF-FD surrogate.

The baselines are evaluated on the retained 40 independent ellipsoidal test
curves (201-point grid) stored in
``results/ellipsoidal_uq/{flat,stiffened}/ellipsoidal_calibration_40x40.npz``.
All methods predict the observation-point FRF (transverse displacement
amplitude at constrained coordinate 420 in mm):

    pole_only      : exact rational branch with the production mode count
                     (4 active modes for the unstiffened plate, 1 for the
                     stiffened plate), no neural background.
    ms8 / ms16     : modal superposition with 8 / 16 modes from a wider
                     nominal modal band (0--800 Hz).
    ma_sc4/ma_sc8  : mode acceleration with residual flexibility (static
                     correction based on A(0)) using 4/1 and 8 modes.
    nn_aligned     : nearest-calibration-curve baseline after exact
                     frequency alignment f*sqrt(rho/E) and amplitude
                     normalization u*E/p.
    interp_aligned : same alignment with linear interpolation in the loss
                     factor eta between the two bracketing calibration
                     curves.
    pod_mlp        : retained pole-off POD-MLP ablation checkpoint
                     (1500 epochs x 3 members).
    proposed       : retained production surrogate checkpoint
                     (5000 epochs x 5 members).

The direct RBF-FD timing is measured on the same 201-point grid so that the
online-cost comparison includes a classical reference rather than only a
slow direct solver.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from rbf_fd_full_order import (
    ModalResonanceConfig,
    ModalResonanceInterface,
    ParametricModalRationalInterface,
    ParameterBounds,
    build_modal_resonance_interface,
    build_parametric_frequency_family,
    load_parametric_operator,
    predict_parametric_observation_members,
    prepare_constrained_operators,
)
from rbf_fd_full_order.examples.cases import MeshConfig, build_case_system
from rbf_fd_full_order.visualization import apply_journal_style, save_figure


CASE_LABELS = {"flat": "Unstiffened plate", "stiffened": "Stiffened plate"}
OBSERVATION_INDEX = 420
METHOD_ORDER = (
    "pole_only",
    "ms8",
    "ms16",
    "ma_sc4",
    "ma_sc8",
    "nn_aligned",
    "interp_aligned",
    "pod_mlp",
    "proposed",
)
METHOD_LABELS = {
    "pole_only": "Pole-only (4/1)",
    "ms8": "Modal superp. 8",
    "ms16": "Modal superp. 16",
    "ma_sc4": "Static-corrected modal (4/1)",
    "ma_sc8": "Static-corrected modal (8)",
    "nn_aligned": "Aligned NN",
    "interp_aligned": "Aligned interp.",
    "pod_mlp": "POD-MLP (pole-off)",
    "proposed": "Proposed surrogate",
}
PRODUCTION_MODES = {"flat": 4, "stiffened": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=("flat", "stiffened"), default=("flat", "stiffened"))
    parser.add_argument("--out", type=Path, default=Path("results") / "classical_modal_baselines")
    parser.add_argument("--uq-root", type=Path, default=Path("results") / "ellipsoidal_uq")
    parser.add_argument("--ablation-root", type=Path, default=Path("results") / "parametric_ablation_compact")
    parser.add_argument("--production-root", type=Path, default=Path("results"))
    parser.add_argument("--max-test-curves", type=int, default=40)
    parser.add_argument("--direct-timing-runs", type=int, default=3)
    parser.add_argument("--skip-direct-timing", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _bounds() -> ParameterBounds:
    return ParameterBounds.default().validated()


def _mesh() -> MeshConfig:
    return MeshConfig()


def _modal_interface(
    operators,
    bounds: ParameterBounds,
    *,
    n_modes: int,
    frequency_max_hz: float,
) -> tuple[ModalResonanceInterface, float]:
    config = ModalResonanceConfig(
        enabled=True,
        n_modes=n_modes,
        frequency_min_hz=0.0,
        frequency_max_hz=frequency_max_hz,
        relative_half_width=0.04,
        minimum_half_width_hz=0.75,
        transition_fraction=0.20,
        exclusion_threshold=0.50,
        exclude_labeled=True,
        exclude_physics=True,
        minimum_participation_ratio=1.0e-6,
        use_sparse_solver=True,
    )
    started = time.perf_counter()
    reference = build_modal_resonance_interface(
        operators,
        config,
        loss_factor=float(bounds.center[3]),
    )
    elapsed = time.perf_counter() - started
    return reference, elapsed


def _production_band_max_hz(bounds: ParameterBounds) -> float:
    ratio_min = float(bounds.lower[0] / bounds.upper[1])
    return 200.0 / math.sqrt(ratio_min)


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


def _metrics(predicted_mm: np.ndarray, reference_mm: np.ndarray, frequencies: np.ndarray) -> dict[str, float]:
    """Observation-FRF metrics over complete curves (percentages where stated)."""
    predicted = np.asarray(predicted_mm, dtype=float)
    reference = np.asarray(reference_mm, dtype=float)
    if predicted.ndim == 1:
        predicted = predicted[None, :]
    if reference.ndim == 1:
        reference = reference[None, :]
    epsilon = 1.0e-30
    curve_errors = [
        float(np.linalg.norm(predicted[index] - reference[index]) / max(float(np.linalg.norm(reference[index])), epsilon))
        for index in range(predicted.shape[0])
    ]
    interp_errors = []
    amplitude_errors = []
    for index in range(predicted.shape[0]):
        ref_peak = _parabolic_peak(frequencies, reference[index])
        pred_peak = _parabolic_peak(frequencies, predicted[index])
        interp_errors.append(abs(pred_peak - ref_peak))
        ref_amplitude = float(np.max(reference[index]))
        pred_amplitude = float(np.max(predicted[index]))
        amplitude_errors.append(abs(pred_amplitude - ref_amplitude) / max(ref_amplitude, epsilon))
    curve_errors = np.asarray(curve_errors, dtype=float)
    return {
        "curve_relative_l2_median_percent": float(100.0 * np.median(curve_errors)),
        "curve_relative_l2_p95_percent": float(100.0 * np.percentile(curve_errors, 95.0)),
        "curve_relative_l2_max_percent": float(100.0 * np.max(curve_errors)),
        "peak_frequency_error_interp_mean_hz": float(np.mean(interp_errors)),
        "peak_frequency_error_interp_max_hz": float(np.max(interp_errors)),
        "peak_amplitude_relative_error_mean_percent": float(100.0 * np.mean(amplitude_errors)),
        "peak_amplitude_relative_error_max_percent": float(100.0 * np.max(amplitude_errors)),
    }


def _aligned_interpolation(
    calibration_mu: np.ndarray,
    calibration_mm: np.ndarray,
    test_mu: np.ndarray,
    frequencies: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Frequency-aligned and amplitude-normalized calibration baselines.

    For the proportional family ``u(f; mu) = (s_p/s_E) v(f sqrt(s_rho/s_E), eta)``.
    Calibration curves are transformed to the reference function ``v(g, eta)``,
    interpolated in ``g``, blended in ``eta``, and mapped back to the test
    parameter point.
    """
    sE_c, srho_c, sp_c, eta_c = calibration_mu.T
    sE_t, srho_t, sp_t, eta_t = test_mu.T
    g_c = frequencies[:, None] * np.sqrt(srho_c / sE_c)[None, :]
    v_c = calibration_mm.T / (sp_c / sE_c)[None, :]
    n_test = test_mu.shape[0]
    n_f = frequencies.size
    results = np.empty((n_test, n_f), dtype=float)
    for index in range(n_test):
        g = frequencies * np.sqrt(srho_t[index] / sE_t[index])
        if mode == "nn_aligned":
            nearest = int(np.argmin(np.abs(eta_c - eta_t[index])))
            v = np.interp(g, g_c[:, nearest], v_c[:, nearest], left=v_c[0, nearest], right=v_c[-1, nearest])
            # Analytic dominant-resonance amplitude correction: peak ~ 1/eta.
            v = v * (eta_c[nearest] / eta_t[index])
        else:  # interp_aligned
            order = np.argsort(np.abs(eta_c - eta_t[index]))[:2]
            first, second = int(order[0]), int(order[1])
            e1, e2 = float(eta_c[first]), float(eta_c[second])
            v1 = np.interp(g, g_c[:, first], v_c[:, first], left=v_c[0, first], right=v_c[-1, first])
            v2 = np.interp(g, g_c[:, second], v_c[:, second], left=v_c[0, second], right=v_c[-1, second])
            v1 = v1 * (e1 / eta_t[index])
            v2 = v2 * (e2 / eta_t[index])
            if abs(e2 - e1) < 1.0e-14:
                v = v1
            else:
                weight = (eta_t[index] - e1) / (e2 - e1)
                v = (1.0 - weight) * v1 + weight * v2
        results[index] = (sp_t[index] / sE_t[index]) * v
    return results


def _modal_observation(
    interface: ParametricModalRationalInterface,
    frequencies: np.ndarray,
    parameters: np.ndarray,
    observation_index: int,
) -> np.ndarray:
    reference = interface.reference
    if reference.n_modes == 0:
        return np.zeros((parameters.shape[0], frequencies.size), dtype=complex)
    omega_sq = (2.0 * np.pi * frequencies)[None, :, None] ** 2
    young = parameters[:, 0:1, None]
    density = parameters[:, 1:2, None]
    pressure = parameters[:, 2:3, None]
    eta = parameters[:, 3:4, None]
    denominator = (1.0 + 1.0j * eta) * young * reference.eigenvalues[None, None, :] - density * omega_sq
    coefficients = pressure * reference.modal_forces[None, None, :] / denominator
    return np.asarray(
        coefficients @ reference.mode_shapes[observation_index, :],
        dtype=complex,
    )


def _ma_sc_observation(
    interface: ParametricModalRationalInterface,
    frequencies: np.ndarray,
    parameters: np.ndarray,
    observation_index: int,
    static_observation: np.ndarray,
) -> np.ndarray:
    """Mode acceleration with residual flexibility (static correction at A(0))."""
    reference = interface.reference
    if reference.n_modes == 0:
        return np.zeros((parameters.shape[0], frequencies.size), dtype=complex)
    young = parameters[:, 0]
    density = parameters[:, 1]
    pressure = parameters[:, 2]
    eta = parameters[:, 3]
    omega_sq = (2.0 * np.pi * frequencies) ** 2
    eigenvalues = reference.eigenvalues
    modal_forces = reference.modal_forces
    phi_obs = reference.mode_shapes[observation_index, :]
    denominator = (
        (1.0 + 1.0j * eta)[:, None, None] * young[:, None, None] * eigenvalues[None, None, :]
        - density[:, None, None] * omega_sq[None, :, None]
    )
    coefficients = pressure[:, None, None] * modal_forces[None, None, :] / denominator
    modal = np.asarray(coefficients @ phi_obs, dtype=complex)
    static_modal = (
        pressure[:, None]
        * modal_forces
        / ((1.0 + 1.0j * eta)[:, None] * young[:, None] * eigenvalues[None, :])
    ) @ phi_obs
    static = (pressure / ((1.0 + 1.0j * eta) * young)) * static_observation
    return static[:, None] - static_modal[:, None] + modal


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)


def _timing(label: str, function, runs: int, warmup: int = 1) -> tuple[float, float]:
    for _ in range(warmup):
        function()
    values = []
    for _ in range(runs):
        started = time.perf_counter()
        function()
        values.append(time.perf_counter() - started)
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values))


def run_case(case: str, args: argparse.Namespace, device: str) -> dict[str, object]:
    out_dir = args.out / case
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = _bounds()
    mesh = _mesh()

    print(f"[{case}] building operators", flush=True)
    started_assembly = time.perf_counter()
    system = build_case_system(case, mesh)
    operators = prepare_constrained_operators(system)
    assembly_s = time.perf_counter() - started_assembly

    # The analysis band is a property of the model: take it from the calibration
    # archive so a case whose band was extended needs no edit here.
    archive_path = args.uq_root / case / "ellipsoidal_calibration_40x40.npz"
    archive = np.load(archive_path)
    frequencies = np.asarray(archive["frequencies_hz"], dtype=float)
    band_min_hz = float(frequencies[0])
    band_max_hz = float(frequencies[-1])

    family_direct = build_parametric_frequency_family(
        operators,
        bounds=bounds,
        modal_config=None,
        frequency_min_hz=band_min_hz,
        frequency_max_hz=band_max_hz,
    )
    calibration_mu = np.asarray(archive["calibration_parameters"], dtype=float)
    test_mu = np.asarray(archive["test_parameters"], dtype=float)
    calibration_mm = np.asarray(archive["calibration_hf_mm"], dtype=float)
    reference_mm = np.asarray(archive["test_hf_mm"], dtype=float)
    if reference_mm.shape[0] > args.max_test_curves:
        reference_mm = reference_mm[: args.max_test_curves]
        test_mu = test_mu[: args.max_test_curves]
    n_test = reference_mm.shape[0]
    print(f"[{case}] test curves = {n_test}, frequencies = {frequencies.size}", flush=True)

    # The production retained set is the one the model actually uses, so its band
    # is the case band, not a shift heuristic.
    production_max = band_max_hz
    print(f"[{case}] production modal band max = {production_max:.1f} Hz", flush=True)
    pole_interface, eigensolve_pole_s = _modal_interface(
        operators, bounds, n_modes=8, frequency_max_hz=production_max
    )
    pole_modal = ParametricModalRationalInterface(reference=pole_interface, bounds=bounds)
    ms8_interface, eigensolve_ms8_s = _modal_interface(operators, bounds, n_modes=8, frequency_max_hz=800.0)
    ms8_modal = ParametricModalRationalInterface(reference=ms8_interface, bounds=bounds)
    ms16_interface, eigensolve_ms16_s = _modal_interface(operators, bounds, n_modes=16, frequency_max_hz=800.0)
    ms16_modal = ParametricModalRationalInterface(reference=ms16_interface, bounds=bounds)
    print(
        f"[{case}] retained modes: production={pole_interface.n_modes}, "
        f"ms8={ms8_interface.n_modes}, ms16={ms16_interface.n_modes}",
        flush=True,
    )
    print(f"[{case}] modal frequencies (ms16): {np.round(ms16_interface.natural_frequencies_hz, 3).tolist()}", flush=True)

    started_static = time.perf_counter()
    static_solution = spla.spsolve(sp.csc_matrix(operators.K), operators.rhs)
    static_observation = float(np.real(np.asarray(static_solution).reshape(-1)[OBSERVATION_INDEX]))
    static_solve_s = time.perf_counter() - started_static

    rows: list[dict[str, object]] = []
    offline: dict[str, float] = {
        "operator_assembly_s": float(assembly_s),
        "modal_eigensolve_production_s": float(eigensolve_pole_s),
        "modal_eigensolve_ms8_s": float(eigensolve_ms8_s),
        "modal_eigensolve_ms16_s": float(eigensolve_ms16_s),
        "static_correction_solve_s": float(static_solve_s),
    }

    # 1) pole-only (exact rational branch, production mode count)
    predicted = np.abs(_modal_observation(pole_modal, frequencies, test_mu, OBSERVATION_INDEX)) * 1.0e3
    rows.append(
        {
            "case": case,
            "case_label": CASE_LABELS[case],
            "method": "pole_only",
            "retained_modes": int(pole_interface.n_modes),
            **_metrics(predicted, reference_mm, frequencies),
        }
    )

    # 2) modal superposition with 8 and 16 modes
    for label, modal in (("ms8", ms8_modal), ("ms16", ms16_modal)):
        predicted = np.abs(_modal_observation(modal, frequencies, test_mu, OBSERVATION_INDEX)) * 1.0e3
        rows.append(
            {
                "case": case,
                "case_label": CASE_LABELS[case],
                "method": label,
                "retained_modes": int(modal.reference.n_modes),
                **_metrics(predicted, reference_mm, frequencies),
            }
        )

    # 3) mode acceleration + residual flexibility (4/1 and 8 modes)
    for label, modal in (("ma_sc4", pole_modal), ("ma_sc8", ms8_modal)):
        predicted = np.abs(
            _ma_sc_observation(modal, frequencies, test_mu, OBSERVATION_INDEX, static_observation)
        ) * 1.0e3
        rows.append(
            {
                "case": case,
                "case_label": CASE_LABELS[case],
                "method": label,
                "retained_modes": int(modal.reference.n_modes),
                **_metrics(predicted, reference_mm, frequencies),
            }
        )

    # 4) frequency-aligned nearest / interpolated calibration baselines
    for mode in ("nn_aligned", "interp_aligned"):
        predicted = _aligned_interpolation(calibration_mu, calibration_mm, test_mu, frequencies, mode=mode)
        rows.append(
            {
                "case": case,
                "case_label": CASE_LABELS[case],
                "method": mode,
                "retained_modes": 0,
                **_metrics(predicted, reference_mm, frequencies),
            }
        )

    # 5) retained POD-MLP ablation checkpoint (pole-off, 1500 x 3)
    ablation_ckpt = args.ablation_root / case / "direct_data.pt"
    if ablation_ckpt.exists():
        pod_result = load_parametric_operator(ablation_ckpt, family_direct, device=device)
        members = predict_parametric_observation_members(
            pod_result, frequencies, test_mu, OBSERVATION_INDEX
        )
        predicted = np.mean(np.abs(members), axis=0) * 1.0e3
        rows.append(
            {
                "case": case,
                "case_label": CASE_LABELS[case],
                "method": "pod_mlp",
                "retained_modes": 0,
                "training_s": float(sum(member.training_s for member in pod_result.members)),
                **_metrics(predicted, reference_mm, frequencies),
            }
        )
    else:
        print(f"[{case}] WARNING: ablation checkpoint missing: {ablation_ckpt}", flush=True)

    # 6) retained production surrogate (5000 x 5)
    production_dir = args.production_root / f"parametric_many_query_{case}_5000_best" / case
    production_ckpt = production_dir / "parametric_operator.pt"
    if production_ckpt.exists():
        production_family = build_parametric_frequency_family(
            operators,
            bounds=bounds,
            modal_config=ModalResonanceConfig(
                enabled=True,
                n_modes=8,
                relative_half_width=0.04,
                minimum_half_width_hz=0.75,
                transition_fraction=0.20,
                exclusion_threshold=0.50,
                exclude_labeled=True,
                exclude_physics=True,
            ),
            frequency_min_hz=band_min_hz,
            frequency_max_hz=band_max_hz,
        )
        production_result = load_parametric_operator(production_ckpt, production_family, device=device)
        members = predict_parametric_observation_members(
            production_result, frequencies, test_mu, OBSERVATION_INDEX
        )
        predicted = np.mean(np.abs(members), axis=0) * 1.0e3
        rows.append(
            {
                "case": case,
                "case_label": CASE_LABELS[case],
                "method": "proposed",
                "retained_modes": int(production_result.family.modal.reference.n_modes),
                "training_s": float(sum(member.training_s for member in production_result.members)),
                **_metrics(predicted, reference_mm, frequencies),
            }
        )
    else:
        print(f"[{case}] WARNING: production checkpoint missing: {production_ckpt}", flush=True)

    # 7) online timing on one complete 201-point query
    timing: dict[str, dict[str, float]] = {}
    single_mu = test_mu[0:1]
    if not args.skip_direct_timing and args.direct_timing_runs > 0:
        mean, std = _timing(
            "direct",
            lambda: family_direct.solve_grid(frequencies, single_mu[0]),
            runs=args.direct_timing_runs,
            warmup=1,
        )
        timing["direct"] = {"mean_s": mean, "std_s": std}
        print(f"[{case}] direct 201-point query: {mean:.3f} s (+- {std:.3f})", flush=True)
    for label, modal in (
        ("pole_only", pole_modal),
        ("ms8", ms8_modal),
        ("ms16", ms16_modal),
        ("ma_sc4", pole_modal),
        ("ma_sc8", ms8_modal),
    ):
        mean, std = _timing(
            label,
            lambda m=modal: _modal_observation(m, frequencies, single_mu, OBSERVATION_INDEX),
            runs=5,
            warmup=1,
        )
        timing[label] = {"mean_s": mean, "std_s": std}
    for label, mode in (("nn_aligned", "nn_aligned"), ("interp_aligned", "interp_aligned")):
        mean, std = _timing(
            label,
            lambda m=mode: _aligned_interpolation(calibration_mu, calibration_mm, single_mu, frequencies, mode=m),
            runs=5,
            warmup=1,
        )
        timing[label] = {"mean_s": mean, "std_s": std}

    if ablation_ckpt.exists():
        mean, std = _timing(
            "pod_mlp",
            lambda: predict_parametric_observation_members(
                pod_result, frequencies, single_mu, OBSERVATION_INDEX
            ),
            runs=3,
            warmup=1,
        )
        timing["pod_mlp"] = {"mean_s": mean, "std_s": std}
    if production_ckpt.exists():
        mean, std = _timing(
            "proposed",
            lambda: predict_parametric_observation_members(
                production_result, frequencies, single_mu, OBSERVATION_INDEX
            ),
            runs=3,
            warmup=1,
        )
        timing["proposed"] = {"mean_s": mean, "std_s": std}

    summary = {
        "case": case,
        "case_label": CASE_LABELS[case],
        "observation_index": OBSERVATION_INDEX,
        "test_curve_count": int(n_test),
        "frequency_point_count": int(frequencies.size),
        "device": device,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": int(os.cpu_count() or 0),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "offline_s": offline,
        "timing_s": timing,
        "rows": rows,
    }
    _write_json(out_dir / "baseline_summary.json", summary)
    _write_csv(out_dir / "baseline_metrics.csv", rows)
    _write_json(args.out / f"{case}_timing.json", {"case": case, "offline_s": offline, "timing_s": timing})
    print(f"[{case}] wrote {out_dir / 'baseline_summary.json'}", flush=True)
    return summary


def make_figure(summaries: list[dict[str, object]], out: Path) -> None:
    apply_journal_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    colors = {
        "pole_only": "#8C8C8C",
        "ms8": "#A6A6A6",
        "ms16": "#BFBFBF",
        "ma_sc4": "#5B8FF9",
        "ma_sc8": "#9BB8F3",
        "nn_aligned": "#5AD8A6",
        "interp_aligned": "#2F9E77",
        "pod_mlp": "#EDB120",
        "proposed": "#D62728",
    }
    width = 0.085
    for case_index, summary in enumerate(summaries):
        rows = {row["method"]: row for row in summary["rows"]}
        for method_index, method in enumerate(METHOD_ORDER):
            if method not in rows:
                continue
            x = case_index + (method_index - len(METHOD_ORDER) / 2.0 + 0.5) * width
            axes[0].bar(x, float(rows[method]["curve_relative_l2_median_percent"]), width, color=colors[method])
            axes[1].bar(x, float(rows[method]["peak_frequency_error_interp_mean_hz"]), width, color=colors[method])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Median observation-FRF relative $L_2$ error (%)")
    axes[0].set_xticks(range(len(summaries)))
    axes[0].set_xticklabels([summary["case_label"] for summary in summaries])
    axes[0].grid(True, axis="y", which="both", alpha=0.3)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Peak-frequency error (Hz, interpolated mean)")
    axes[1].set_xticks(range(len(summaries)))
    axes[1].set_xticklabels([summary["case_label"] for summary in summaries])
    axes[1].grid(True, axis="y", which="both", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[method]) for method in METHOD_ORDER]
    figure.legend(
        handles,
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=3,
        frameon=False,
    )
    save_figure(figure, out / "classical_baselines.png")
    save_figure(figure, out / "classical_baselines.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    if args.device != "auto":
        device = args.device
    summaries = []
    for case in args.cases:
        summaries.append(run_case(case, args, device))
    make_figure(summaries, args.out)
    print(f"Wrote baseline summaries to {args.out}", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Equal-retained-mode test of a learned dynamic background against MASC.

The experiment is deliberately restricted to the production unstiffened-plate
surrogate and its four retained rational modes.  The candidate response DOF is
selected from nominal *direct RBF-FD and four-mode MASC responses only*; the
surrogate is not consulted during selection.  This prevents selecting a point
because the learned model happens to perform well there.

The script then compares, at exactly the same four retained modes:

1. direct constrained RBF-FD;
2. four-mode pole response plus zero-frequency residual flexibility (MASC4);
3. four-mode pole response plus the learned frequency-dependent background.

Outputs are written below ``results/equal_mode_dynamic_background``.  Direct
nominal and test solutions are cached so interrupted runs can be resumed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from rbf_fd_full_order import (
    predict_parametric_members,
    predict_parametric_observation_members,
)
from rbf_fd_full_order.visualization.result_support import load_parametric_case
from plotting.publication_style import apply_publication_style, format_axis, save_figure


SUCCESS_MEDIAN_RATIO = 0.25
SUCCESS_P95_RATIO = 0.50
SUCCESS_CURVE_WIN_RATE = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path("results") / "parametric_many_query_flat_5000_best" / "flat",
    )
    parser.add_argument(
        "--uq-archive",
        type=Path,
        default=Path("results") / "ellipsoidal_uq" / "flat" / "ellipsoidal_calibration_40x40.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "equal_mode_dynamic_background",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-curves", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frequency-min", type=float, default=0.0)
    parser.add_argument("--frequency-max", type=float, default=200.0)
    parser.add_argument("--frequency-step", type=float, default=1.0)
    parser.add_argument("--selection-min", type=float, default=130.0)
    parser.add_argument("--selection-max", type=float, default=188.0)
    parser.add_argument("--interior-margin", type=float, default=0.10)
    parser.add_argument("--fixed-x", type=float)
    parser.add_argument("--fixed-y", type=float)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def complex_relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(predicted) - np.asarray(reference)))
    denominator = max(float(np.linalg.norm(reference)), 1.0e-30)
    return numerator / denominator


def amplitude_relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return complex_relative_l2(np.abs(predicted), np.abs(reference))


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def solve_static_nominal(operators) -> np.ndarray:
    matrix = sp.csc_matrix(operators.K) if sp.issparse(operators.K) else np.asarray(operators.K)
    if sp.issparse(matrix):
        return np.asarray(spla.spsolve(matrix, operators.rhs), dtype=complex)
    return np.asarray(np.linalg.solve(matrix, operators.rhs), dtype=complex)


def masc_full(family, frequencies: np.ndarray, parameters: np.ndarray, static_nominal: np.ndarray) -> np.ndarray:
    mu = np.asarray(parameters, dtype=float).reshape(-1)
    modal = family.modal.modal_response(frequencies, mu)
    modal_zero = family.modal.modal_response(np.asarray([0.0]), mu)[0]
    young, _, pressure, eta = mu
    static = pressure / ((1.0 + 1.0j * eta) * young) * static_nominal
    return modal + (static - modal_zero)[None, :]


def masc_observation(
    family,
    frequencies: np.ndarray,
    parameters: np.ndarray,
    observation_index: int,
    static_nominal_value: complex,
) -> np.ndarray:
    mu = np.asarray(parameters, dtype=float)
    if mu.ndim == 1:
        mu = mu[None, :]
    reference = family.modal.reference
    omega_sq = (2.0 * np.pi * frequencies)[None, :, None] ** 2
    young = mu[:, 0:1, None]
    density = mu[:, 1:2, None]
    pressure = mu[:, 2:3, None]
    eta = mu[:, 3:4, None]
    denominator = (1.0 + 1.0j * eta) * young * reference.eigenvalues[None, None, :] - density * omega_sq
    coefficients = pressure * reference.modal_forces[None, None, :] / denominator
    phi_obs = reference.mode_shapes[int(observation_index), :]
    modal = np.asarray(coefficients @ phi_obs, dtype=complex)
    static_modal = (
        pressure[:, 0, 0, None]
        * reference.modal_forces[None, :]
        / ((1.0 + 1.0j * eta[:, 0, 0, None]) * young[:, 0, 0, None] * reference.eigenvalues[None, :])
    ) @ phi_obs
    static = (
        mu[:, 2] / ((1.0 + 1.0j * mu[:, 3]) * mu[:, 0]) * complex(static_nominal_value)
    )
    return modal + (static - static_modal)[:, None]


def displacement_constrained_indices(system, operators, margin: float) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = int(system.nodes.shape[0])
    if operators.free_dofs is None:
        raise RuntimeError("The square-plate experiment expects direct fixed-DOF elimination.")
    free = np.asarray(operators.free_dofs, dtype=int)
    mask = free < n_nodes
    constrained = np.flatnonzero(mask)
    physical_nodes = free[mask]
    xy = np.asarray(system.nodes[physical_nodes], dtype=float)
    interior = (
        (xy[:, 0] >= margin)
        & (xy[:, 0] <= 1.0 - margin)
        & (xy[:, 1] >= margin)
        & (xy[:, 1] <= 1.0 - margin)
    )
    return constrained[interior], physical_nodes[interior]


def direct_nominal_cache(
    cache: Path,
    family,
    frequencies: np.ndarray,
    nominal: np.ndarray,
    force: bool,
) -> np.ndarray:
    if cache.exists() and not force:
        data = np.load(cache)
        if np.allclose(data["frequencies_hz"], frequencies) and np.allclose(data["parameters"], nominal):
            return np.asarray(data["response"], dtype=complex)
    started = time.perf_counter()
    response = family.solve_grid(frequencies, nominal)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        frequencies_hz=frequencies,
        parameters=nominal,
        response=response,
        elapsed_s=np.asarray(time.perf_counter() - started),
    )
    return response


def select_candidate(
    frequencies: np.ndarray,
    direct: np.ndarray,
    masc: np.ndarray,
    displacement_indices: np.ndarray,
    physical_nodes: np.ndarray,
    system,
    selection_min: float,
    selection_max: float,
) -> tuple[int, int, list[dict[str, float]]]:
    band = (frequencies >= selection_min) & (frequencies <= selection_max)
    direct_band = direct[band][:, displacement_indices]
    masc_band = masc[band][:, displacement_indices]
    response_norm = np.linalg.norm(direct_band, axis=0)
    minimum_norm = float(np.quantile(response_norm, 0.35))
    relative_error = np.linalg.norm(masc_band - direct_band, axis=0) / np.maximum(response_norm, 1.0e-30)

    # Prefer a genuine interior dip, but choose solely from direct/MASC physics.
    amplitude = np.abs(direct_band)
    dip_depth = np.ones(amplitude.shape[1], dtype=float)
    dip_frequency = np.full(amplitude.shape[1], np.nan, dtype=float)
    band_frequencies = frequencies[band]
    for column in range(amplitude.shape[1]):
        curve = amplitude[:, column]
        if curve.size < 5:
            continue
        local = np.flatnonzero((curve[1:-1] < curve[:-2]) & (curve[1:-1] <= curve[2:])) + 1
        local = local[(local >= 2) & (local <= curve.size - 3)]
        if local.size == 0:
            continue
        idx = int(local[np.argmin(curve[local])])
        shoulder = max(float(np.max(curve[: idx + 1])), float(np.max(curve[idx:])), 1.0e-30)
        dip_depth[column] = float(curve[idx] / shoulder)
        dip_frequency[column] = float(band_frequencies[idx])

    eligible = response_norm >= minimum_norm
    # MASC error is the primary score; antiresonance depth is a mild tie-breaker.
    score = relative_error * (1.0 + np.clip(-np.log10(np.maximum(dip_depth, 1.0e-12)), 0.0, 4.0))
    score[~eligible] = -np.inf
    winner = int(np.argmax(score))
    constrained_index = int(displacement_indices[winner])
    physical_node = int(physical_nodes[winner])

    rows: list[dict[str, float]] = []
    order = np.argsort(score)[::-1]
    for rank, column in enumerate(order[: min(50, order.size)], start=1):
        node = int(physical_nodes[column])
        rows.append(
            {
                "rank": rank,
                "constrained_index": int(displacement_indices[column]),
                "physical_node": node,
                "x": float(system.nodes[node, 0]),
                "y": float(system.nodes[node, 1]),
                "direct_response_norm": float(response_norm[column]),
                "masc_complex_relative_l2": float(relative_error[column]),
                "antiresonance_depth_ratio": float(dip_depth[column]),
                "antiresonance_frequency_hz": float(dip_frequency[column]),
                "selection_score": float(score[column]),
            }
        )
    return constrained_index, physical_node, rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def direct_test_cache(
    cache: Path,
    family,
    frequencies: np.ndarray,
    parameters: np.ndarray,
    observation_index: int,
    force: bool,
    workers: int,
) -> np.ndarray:
    if cache.exists() and not force:
        data = np.load(cache)
        if (
            int(data["observation_index"]) == int(observation_index)
            and np.allclose(data["frequencies_hz"], frequencies)
            and np.allclose(data["parameters"], parameters)
        ):
            return np.asarray(data["response"], dtype=complex)
    response = np.full((parameters.shape[0], frequencies.size), np.nan + 1.0j * np.nan, dtype=complex)
    completed = np.zeros(parameters.shape[0], dtype=bool)
    if cache.exists() and not force:
        data = np.load(cache)
        if (
            int(data["observation_index"]) == int(observation_index)
            and np.allclose(data["frequencies_hz"], frequencies)
            and np.allclose(data["parameters"], parameters)
        ):
            response[:] = np.asarray(data["response"], dtype=complex)
            completed[:] = np.asarray(data.get("completed", np.all(np.isfinite(response), axis=1)), dtype=bool)

    # Reuse a smaller completed prefix, including the initial eight-curve pilot.
    if not force and not np.any(completed):
        for prior in sorted(cache.parent.glob("test_direct_candidate_n*.npz"), reverse=True):
            if prior == cache:
                continue
            data = np.load(prior)
            prior_parameters = np.asarray(data["parameters"], dtype=float)
            if (
                int(data["observation_index"]) == int(observation_index)
                and np.allclose(data["frequencies_hz"], frequencies)
                and prior_parameters.shape[0] <= parameters.shape[0]
                and np.allclose(prior_parameters, parameters[: prior_parameters.shape[0]])
            ):
                count = prior_parameters.shape[0]
                response[:count] = np.asarray(data["response"], dtype=complex)
                completed[:count] = np.asarray(
                    data.get("completed", np.all(np.isfinite(response[:count]), axis=1)), dtype=bool
                )
                break

    def solve_curve(mu_index: int) -> tuple[int, np.ndarray]:
        curve = np.empty(frequencies.size, dtype=complex)
        mu = parameters[mu_index]
        for frequency_index, frequency in enumerate(frequencies):
            curve[frequency_index] = family.solve(float(frequency), mu)[int(observation_index)]
        return mu_index, curve

    started = time.perf_counter()
    pending = np.flatnonzero(~completed).tolist()
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as executor:
        futures = {executor.submit(solve_curve, int(index)): int(index) for index in pending}
        for future in as_completed(futures):
            mu_index, curve = future.result()
            response[mu_index] = curve
            completed[mu_index] = True
            np.savez_compressed(
                cache,
                frequencies_hz=frequencies,
                parameters=parameters,
                observation_index=np.asarray(observation_index),
                response=response,
                completed=completed,
                elapsed_s=np.asarray(time.perf_counter() - started),
            )
            print(
                f"direct test curves {int(np.sum(completed))}/{parameters.shape[0]} "
                f"({time.perf_counter() - started:.1f} s current run)",
                flush=True,
            )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        frequencies_hz=frequencies,
        parameters=parameters,
        observation_index=np.asarray(observation_index),
        response=response,
        completed=completed,
        elapsed_s=np.asarray(time.perf_counter() - started),
    )
    return response


def make_figure(
    path: Path,
    frequencies: np.ndarray,
    direct: np.ndarray,
    masc: np.ndarray,
    surrogate: np.ndarray,
    pole: np.ndarray,
    candidate_xy: tuple[float, float],
    selection_min: float,
    selection_max: float,
) -> None:
    apply_publication_style()
    background_direct = direct - pole
    background_masc = masc - pole
    background_surrogate = surrogate - pole
    plot_max = min(float(frequencies[-1]), selection_max + 4.0)
    band = (frequencies >= selection_min) & (frequencies <= plot_max)
    scale = 1.0e3
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    axes[0, 0].plot(frequencies, np.abs(direct) * scale, color="black", lw=1.5, label="Direct RBF-FD")
    axes[0, 0].plot(frequencies, np.abs(masc) * scale, color="#0072B2", ls="--", label="MASC, 4 modes")
    axes[0, 0].plot(frequencies, np.abs(surrogate) * scale, color="#D55E00", ls="-.", label="Surrogate, 4 modes")
    axes[0, 0].set_title(f"Selected displacement response at ({candidate_xy[0]:.3f}, {candidate_xy[1]:.3f})")
    axes[0, 0].set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("Amplitude (mm)")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(frequencies[band], np.abs(direct[band]) * scale, color="black", lw=1.5)
    axes[0, 1].plot(frequencies[band], np.abs(masc[band]) * scale, color="#0072B2", ls="--")
    axes[0, 1].plot(frequencies[band], np.abs(surrogate[band]) * scale, color="#D55E00", ls="-.")
    axes[0, 1].set_title("High-frequency/antiresonance window")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Amplitude (mm)")

    axes[1, 0].plot(frequencies[band], np.real(background_direct[band]) * scale, color="black", lw=1.5, label="Direct background")
    axes[1, 0].plot(frequencies[band], np.real(background_masc[band]) * scale, color="#0072B2", ls="--", label="Static background")
    axes[1, 0].plot(frequencies[band], np.real(background_surrogate[band]) * scale, color="#D55E00", ls="-.", label="Learned background")
    axes[1, 0].set_title("Real part of truncated background")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Displacement (mm)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    phase_direct = np.unwrap(np.angle(direct[band]))
    phase_masc = np.unwrap(np.angle(masc[band]))
    phase_surrogate = np.unwrap(np.angle(surrogate[band]))
    axes[1, 1].plot(frequencies[band], phase_direct, color="black", lw=1.5, label="Direct RBF-FD")
    axes[1, 1].plot(frequencies[band], phase_masc, color="#0072B2", ls="--", label="MASC")
    axes[1, 1].plot(frequencies[band], phase_surrogate, color="#D55E00", ls="-.", label="Surrogate")
    axes[1, 1].set_title("Phase through the selected window")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Unwrapped phase (rad)")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.tick_params(which="both", direction="in", top=True, right=True)
        format_axis(axis, tight=False)
    save_figure(figure, path.parent, path.stem, dpi=600)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frequencies = np.arange(
        args.frequency_min,
        args.frequency_max + 0.5 * args.frequency_step,
        args.frequency_step,
        dtype=float,
    )
    metadata, system, operators, family, result, original_observation = load_parametric_case(
        args.case_dir, device=args.device
    )
    if family.modal is None or family.modal.n_modes != 4:
        raise RuntimeError(f"Expected exactly four production modes, found {0 if family.modal is None else family.modal.n_modes}.")
    nominal = np.asarray(family.bounds.center, dtype=float)
    static_nominal = solve_static_nominal(operators)

    direct_full = direct_nominal_cache(
        args.out / "nominal_direct_full.npz",
        family,
        frequencies,
        nominal,
        args.force_recompute,
    )
    pole_full = family.modal.modal_response(frequencies, nominal)
    masc_full_response = masc_full(family, frequencies, nominal, static_nominal)

    displacement_indices, physical_nodes = displacement_constrained_indices(
        system, operators, args.interior_margin
    )
    candidate_index, candidate_node, ranking = select_candidate(
        frequencies,
        direct_full,
        masc_full_response,
        displacement_indices,
        physical_nodes,
        system,
        args.selection_min,
        args.selection_max,
    )
    selection_kind = "direct-masc physics scan"
    if args.fixed_x is not None or args.fixed_y is not None:
        if args.fixed_x is None or args.fixed_y is None:
            raise ValueError("--fixed-x and --fixed-y must be supplied together.")
        target = np.asarray([args.fixed_x, args.fixed_y], dtype=float)
        candidate_position = int(
            np.argmin(np.sum((system.nodes[physical_nodes] - target[None, :]) ** 2, axis=1))
        )
        candidate_index = int(displacement_indices[candidate_position])
        candidate_node = int(physical_nodes[candidate_position])
        selection_kind = "fixed geometric point"
    write_csv(args.out / "candidate_ranking.csv", ranking)
    candidate_xy = tuple(float(value) for value in system.nodes[candidate_node])

    surrogate_nominal_members = predict_parametric_members(result, frequencies, nominal)
    surrogate_nominal = np.mean(surrogate_nominal_members[:, 0, :, candidate_index], axis=0)
    direct_nominal = direct_full[:, candidate_index]
    masc_nominal = masc_full_response[:, candidate_index]
    pole_nominal = pole_full[:, candidate_index]

    band = (frequencies >= args.selection_min) & (frequencies <= args.selection_max)
    # Field-wide nominal comparison is computed after the physics-only selection.
    surrogate_full_nominal = np.mean(surrogate_nominal_members[:, 0, :, :], axis=0)
    field_masc_errors = np.asarray(
        [
            complex_relative_l2(masc_full_response[band, idx], direct_full[band, idx])
            for idx in displacement_indices
        ]
    )
    field_surrogate_errors = np.asarray(
        [
            complex_relative_l2(surrogate_full_nominal[band, idx], direct_full[band, idx])
            for idx in displacement_indices
        ]
    )

    archive = np.load(args.uq_archive)
    test_parameters = np.asarray(archive["test_parameters"], dtype=float)[: max(int(args.test_curves), 1)]
    test_frequencies = frequencies[band]
    direct_test = direct_test_cache(
        args.out / f"test_direct_observation_{candidate_index}_n{test_parameters.shape[0]}.npz",
        family,
        test_frequencies,
        test_parameters,
        candidate_index,
        args.force_recompute,
        args.workers,
    )
    masc_test = masc_observation(
        family,
        test_frequencies,
        test_parameters,
        candidate_index,
        static_nominal[candidate_index],
    )
    surrogate_test_members = predict_parametric_observation_members(
        result,
        test_frequencies,
        test_parameters,
        candidate_index,
    )
    surrogate_test = np.mean(surrogate_test_members, axis=0)

    rows: list[dict[str, object]] = []
    masc_complex = []
    surrogate_complex = []
    masc_amplitude = []
    surrogate_amplitude = []
    for index in range(test_parameters.shape[0]):
        mc = complex_relative_l2(masc_test[index], direct_test[index])
        sc = complex_relative_l2(surrogate_test[index], direct_test[index])
        ma = amplitude_relative_l2(masc_test[index], direct_test[index])
        sa = amplitude_relative_l2(surrogate_test[index], direct_test[index])
        masc_complex.append(mc)
        surrogate_complex.append(sc)
        masc_amplitude.append(ma)
        surrogate_amplitude.append(sa)
        rows.append(
            {
                "curve": index,
                "young_scale": float(test_parameters[index, 0]),
                "density_scale": float(test_parameters[index, 1]),
                "pressure_scale": float(test_parameters[index, 2]),
                "loss_factor": float(test_parameters[index, 3]),
                "masc_complex_relative_l2": mc,
                "surrogate_complex_relative_l2": sc,
                "complex_error_ratio_surrogate_over_masc": sc / max(mc, 1.0e-30),
                "masc_amplitude_relative_l2": ma,
                "surrogate_amplitude_relative_l2": sa,
                "surrogate_better_complex": bool(sc < mc),
            }
        )
    write_csv(args.out / "independent_test_metrics.csv", rows)

    masc_complex = np.asarray(masc_complex)
    surrogate_complex = np.asarray(surrogate_complex)
    masc_amplitude = np.asarray(masc_amplitude)
    surrogate_amplitude = np.asarray(surrogate_amplitude)
    median_ratio = float(np.median(surrogate_complex) / max(float(np.median(masc_complex)), 1.0e-30))
    p95_ratio = float(np.quantile(surrogate_complex, 0.95) / max(float(np.quantile(masc_complex, 0.95)), 1.0e-30))
    curve_win_rate = float(np.mean(surrogate_complex < masc_complex))
    success = bool(
        median_ratio <= SUCCESS_MEDIAN_RATIO
        and p95_ratio <= SUCCESS_P95_RATIO
        and curve_win_rate >= SUCCESS_CURVE_WIN_RATE
    )

    summary = {
        "experiment": "equal-retained-mode dynamic truncated background",
        "case": metadata["case"],
        "retained_modes_both_methods": int(family.modal.n_modes),
        "retained_natural_frequencies_hz": family.modal.reference.natural_frequencies_hz.tolist(),
        "candidate_selection_uses_surrogate": False,
        "candidate_selection_kind": selection_kind,
        "candidate": {
            "constrained_index": candidate_index,
            "physical_node": candidate_node,
            "x": candidate_xy[0],
            "y": candidate_xy[1],
            "original_manuscript_observation_index": int(original_observation),
            "selection_frequency_window_hz": [args.selection_min, args.selection_max],
        },
        "nominal_candidate_errors": {
            "masc_complex_relative_l2": complex_relative_l2(masc_nominal[band], direct_nominal[band]),
            "surrogate_complex_relative_l2": complex_relative_l2(surrogate_nominal[band], direct_nominal[band]),
            "masc_amplitude_relative_l2": amplitude_relative_l2(masc_nominal[band], direct_nominal[band]),
            "surrogate_amplitude_relative_l2": amplitude_relative_l2(surrogate_nominal[band], direct_nominal[band]),
        },
        "nominal_antiresonance": {},
        "nominal_all_interior_displacement_dofs": {
            "count": int(displacement_indices.size),
            "surrogate_better_fraction": float(np.mean(field_surrogate_errors < field_masc_errors)),
            "masc_complex_relative_l2": quantiles(field_masc_errors),
            "surrogate_complex_relative_l2": quantiles(field_surrogate_errors),
            "median_error_ratio_surrogate_over_masc": float(
                np.median(field_surrogate_errors) / max(float(np.median(field_masc_errors)), 1.0e-30)
            ),
        },
        "independent_test": {
            "curve_count": int(test_parameters.shape[0]),
            "frequency_window_hz": [float(test_frequencies[0]), float(test_frequencies[-1])],
            "frequency_points": int(test_frequencies.size),
            "masc_complex_relative_l2": quantiles(masc_complex),
            "surrogate_complex_relative_l2": quantiles(surrogate_complex),
            "masc_amplitude_relative_l2": quantiles(masc_amplitude),
            "surrogate_amplitude_relative_l2": quantiles(surrogate_amplitude),
            "median_ratio_surrogate_over_masc": median_ratio,
            "p95_ratio_surrogate_over_masc": p95_ratio,
            "surrogate_curve_win_rate": curve_win_rate,
        },
        "predeclared_success_thresholds": {
            "median_ratio_max": SUCCESS_MEDIAN_RATIO,
            "p95_ratio_max": SUCCESS_P95_RATIO,
            "curve_win_rate_min": SUCCESS_CURVE_WIN_RATE,
        },
        "supports_retaining_compressed_masc_comparison": success,
    }
    anti_band = (frequencies >= 180.0) & (frequencies <= 192.0)
    for label, curve in (("direct", direct_nominal), ("masc", masc_nominal), ("surrogate", surrogate_nominal)):
        local_index = int(np.argmin(np.abs(curve[anti_band])))
        summary["nominal_antiresonance"][label] = {
            "frequency_hz": float(frequencies[anti_band][local_index]),
            "amplitude_mm": float(np.abs(curve[anti_band][local_index]) * 1.0e3),
        }
    with (args.out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    np.savez_compressed(
        args.out / "selected_case_curves.npz",
        frequencies_hz=frequencies,
        direct=direct_nominal,
        masc=masc_nominal,
        surrogate=surrogate_nominal,
        pole=pole_nominal,
        candidate_index=np.asarray(candidate_index),
        candidate_xy=np.asarray(candidate_xy),
    )
    make_figure(
        args.out / "equal_mode_dynamic_background.png",
        frequencies,
        direct_nominal,
        masc_nominal,
        surrogate_nominal,
        pole_nominal,
        candidate_xy,
        args.selection_min,
        args.selection_max,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not success:
        sys.exit(2)


if __name__ == "__main__":
    main()

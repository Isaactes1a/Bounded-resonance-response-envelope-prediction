"""Frozen-checkpoint local frequency verification, kept outside the manuscript.

Reference values are new calls to ParametricFrequencyFamily.solve, including
its existing diagonal regularization.  All states belong to the diagonal
ellipsoid.  This is deterministic solution/representation verification, not
experimental validation or a global response-envelope certificate.
"""
from __future__ import annotations

import os

# Apply before importing BLAS, SciPy or torch; the independent baseline run uses
# the same thread budget and eigsh starting-vector rule.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "2"

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import scipy
from scipy.optimize import brentq, minimize, minimize_scalar
import scipy.sparse.linalg as spla
import torch

from rbf_fd_full_order import predict_parametric_members, predict_parametric_observation_members
from rbf_fd_full_order.core.modal_resonance import _generalized_modes
from rbf_fd_full_order.visualization import apply_journal_style, save_figure
from rbf_fd_full_order.visualization.result_support import load_parametric_case


CODE = Path(__file__).resolve().parent
DEFAULT_OUT = CODE / "results" / "review_validation_20261103" / "local"
SOURCES = {
    "flat": CODE / "results" / "parametric_many_query_flat_5000_best" / "flat",
    "stiffened": CODE / "results" / "parametric_many_query_stiffened_6mode_0_600" / "stiffened",
}
EIGSH_SEED = 20261103


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def deterministic_eigsh():
    original = spla.eigsh

    def fixed_start(matrix, *args, **kwargs):
        if kwargs.get("v0") is None:
            vector = np.random.default_rng(EIGSH_SEED).standard_normal(matrix.shape[0])
            kwargs["v0"] = vector / np.linalg.norm(vector)
        return original(matrix, *args, **kwargs)

    spla.eigsh = fixed_start
    try:
        yield
    finally:
        spla.eigsh = original


def matrix_hash(operators):
    value = hashlib.sha256()
    for matrix in (operators.K, operators.M):
        if hasattr(matrix, "tocsr"):
            matrix = matrix.tocsr()
            for array in (matrix.data, matrix.indices, matrix.indptr):
                value.update(np.ascontiguousarray(array).tobytes())
        else:
            value.update(np.ascontiguousarray(matrix).tobytes())
    value.update(np.ascontiguousarray(operators.rhs).tobytes())
    return value.hexdigest()


def ellipsoid_state(bounds, xi):
    xi = np.asarray(xi, dtype=float)
    if float(xi @ xi) > 1.0 + 1.e-9:
        raise ValueError("Requested state lies outside the unit ball")
    return bounds.center + 0.5 * (bounds.upper - bounds.lower) * xi


def states_for(case, bounds):
    states = [
        {"name": "nominal", "mu": bounds.center.tolist(), "source": "ellipsoid center"},
        {"name": "eta_min", "mu": ellipsoid_state(bounds, [0, 0, 0, -1]).tolist(),
         "source": "minimum eta feasible on its own ellipsoid axis"},
    ]
    previous = CODE / "results" / "ellipsoid_extremum_search" / case / "extremum_summary.json"
    old = json.loads(previous.read_text(encoding="utf-8"))
    for target in ("peak_amplitude", "envelope_upper_at_f1"):
        states.append({"name": "stored_" + target,
                       "mu": old["refined_targets"][target]["mu"],
                       "source": str(previous),
                       "source_note": "Reuse coordinates only; all response values are newly evaluated with the specified production checkpoint"})
    if case == "stiffened":
        for eta_xi in (0.0, -0.5, -0.6):
            radius = np.sqrt(1.0 - eta_xi**2)

            def ratio(x):
                # SLSQP may temporarily evaluate outside its inequality set.
                # Only the final constrained result is mapped as a valid state.
                xi = np.asarray([x[0], x[1], 0.0, eta_xi])
                mu = bounds.center + 0.5*(bounds.upper-bounds.lower)*xi
                return np.sqrt(mu[0] / mu[1])

            fit = minimize(ratio, np.asarray([-0.8, 0.6]) * radius,
                           method="SLSQP", bounds=[(-radius, radius)] * 2,
                           constraints={"type": "ineq", "fun": lambda x: radius**2 - x @ x},
                           options={"ftol": 1.e-13, "maxiter": 200})
            if not fit.success:
                raise RuntimeError(f"Feasible entry-state construction failed: {fit.message}")
            # Roundoff projection preserves the prescribed eta coordinate.
            pair = fit.x * min(1.0, radius / np.linalg.norm(fit.x))
            xi = [float(pair[0]), float(pair[1]), 0.0, eta_xi]
            states.append({"name": f"entry_eta_xi_{abs(eta_xi):.1f}".replace(".", "p"),
                           "mu": ellipsoid_state(bounds, xi).tolist(),
                           "source": "SLSQP minimum sqrt(sE/srho) at fixed eta coordinate, inside the unit ball"})
    for state in states:
        mu = np.asarray(state["mu"])
        xi = bounds.normalize(mu)
        state.update(xi=xi.tolist(), ellipsoid_radius_squared=float(xi @ xi),
                     frequency_scale=float(np.sqrt(mu[0] / mu[1])))
        if state["ellipsoid_radius_squared"] > 1.0 + 1.e-8:
            raise ValueError(f"Infeasible stored or constructed state: {state['name']}")
    return states


def modal_evidence(operators, system, production, center_index, transfer_index, out):
    with deterministic_eigsh():
        eigenvalues, phi = _generalized_modes(operators, 40, True)
    eigenvalues, phi = eigenvalues[:40], phi[:, :40]
    for j in range(phi.shape[1]):
        phi[:, j] /= np.sqrt(float(phi[:, j] @ (operators.M @ phi[:, j])))
    gamma = np.asarray(phi.T @ operators.rhs, dtype=complex)
    frequency = np.sqrt(eigenvalues) / (2 * np.pi)
    n_nodes = system.nodes.shape[0]
    expanded_phi = operators.expand(phi.T).T
    w_phi = expanded_phi[:n_nodes]
    residual = np.linalg.norm(operators.K @ phi - (operators.M @ phi) * eigenvalues, axis=0)
    residual /= np.linalg.norm(operators.K @ phi, axis=0) + np.linalg.norm((operators.M @ phi) * eigenvalues, axis=0)
    np.savez_compressed(out / "nominal_eigensystems.npz", wide_eigenvalues=eigenvalues,
                        wide_mode_shapes=phi, wide_modal_forces=gamma, wide_frequency_hz=frequency,
                        wide_relative_eigen_residual=residual,
                        production_eigenvalues=production.eigenvalues,
                        production_mode_shapes=production.mode_shapes,
                        production_modal_forces=production.modal_forces,
                        production_frequency_hz=production.natural_frequencies_hz,
                        nodes=system.nodes, free_dofs=operators.free_dofs)
    # Near-degenerate numerical clusters: 0.01 Hz absolute or 2e-5 relative.
    # Gamma norm and Phi_cluster @ Gamma_cluster do not depend on arbitrary
    # orthonormal rotation of a degenerate eigenspace. Individual Gamma does.
    clusters = []
    for j, value in enumerate(frequency):
        if not clusters or value - frequency[clusters[-1][-1]] > max(0.01, 2.e-5 * value):
            clusters.append([j])
        else:
            clusters[-1].append(j)
    peak_gamma = max(np.linalg.norm(gamma[ids]) for ids in clusters)
    rows = []
    individual = []
    for j in range(frequency.size):
        individual.append({"mode": j + 1, "frequency_hz": float(frequency[j]),
                           "gamma_real": float(gamma[j].real), "gamma_imag": float(gamma[j].imag),
                           "eigen_residual_relative": float(residual[j]),
                           "center_residue_real": float((phi[center_index, j] * gamma[j]).real),
                           "transfer_residue_real": float((phi[transfer_index, j] * gamma[j]).real)})
    for ids in clusters:
        residue = phi[:, ids] @ gamma[ids]
        w_residue = w_phi[:, ids] @ gamma[ids]
        node = int(np.argmax(np.abs(w_residue)))
        force_norm = float(np.linalg.norm(gamma[ids]))
        rows.append({"cluster": len(rows), "modes": ";".join(str(j + 1) for j in ids),
                     "count": len(ids), "frequency_min_hz": float(frequency[ids].min()),
                     "frequency_mean_hz": float(frequency[ids].mean()),
                     "frequency_max_hz": float(frequency[ids].max()),
                     "modal_force_norm": force_norm, "force_relative_to_largest_cluster": force_norm / peak_gamma,
                     "negligible_at_relative_1e6": bool(force_norm / peak_gamma < 1.e-6),
                     "center_residue_abs": float(abs(residue[center_index])),
                     "transfer_residue_abs": float(abs(residue[transfer_index])),
                     "max_transverse_residue_abs": float(abs(w_residue[node])),
                     "max_residue_node": node, "max_residue_x": float(system.nodes[node, 0]),
                     "max_residue_y": float(system.nodes[node, 1]),
                     "max_relative_eigen_residual": float(residual[ids].max()),
                     "within_production_count": bool(max(ids) < production.n_modes)})
    write_csv(out / "modal_individual.csv", individual)
    write_csv(out / "modal_clusters.csv", rows)
    return rows


class DirectCache:
    def __init__(self, family, mu, path, identity):
        self.family, self.mu, self.path = family, np.asarray(mu), path
        self.identity = identity
        self.values = {}
        self.new_solves, self.cache_hits, self.solve_seconds = 0, 0, 0.0
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["identity"]) != identity or not np.array_equal(archive["mu"], self.mu):
                    raise ValueError(f"Cache identity mismatch: {path}")
                self.values = {float(f): u for f, u in zip(archive["frequency_hz"], archive["response"])}

    def __call__(self, frequency):
        key = round(float(frequency), 10)
        if key in self.values:
            self.cache_hits += 1
            return self.values[key]
        start = time.perf_counter()
        response = self.family.solve(key, self.mu)
        self.solve_seconds += time.perf_counter() - start
        if not np.all(np.isfinite(response)):
            raise FloatingPointError(f"Nonfinite direct result at {key}")
        self.values[key] = response
        self.new_solves += 1
        if self.new_solves % 20 == 0:
            self.save()
            print(f"  {self.path.stem}: {self.new_solves} new direct solves, {self.solve_seconds:.1f}s", flush=True)
        return response

    def save(self):
        keys = sorted(self.values)
        temporary = self.path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, identity=np.asarray(self.identity), mu=self.mu,
                            frequency_hz=np.asarray(keys), response=np.asarray([self.values[k] for k in keys]))
        temporary.replace(self.path)


def local_plot(path, frequencies, direct, learned, observation):
    apply_journal_style()
    plt.rcParams.update({"xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    axes[0].plot(frequencies, abs(direct[:, observation]) * 1000, "k-", lw=1.5, label="Direct")
    axes[0].plot(frequencies, abs(learned[:, observation]) * 1000, color="#C00000", lw=1.5,
                 marker="o", ms=4, markevery=max(1, frequencies.size // 7), label="Frozen surrogate")
    axes[0].set_ylabel(r"$|w|$ (mm)")
    phase = np.unwrap(np.angle(direct[:, observation]))
    other = phase + np.angle(learned[:, observation] * np.conj(direct[:, observation]))
    axes[1].plot(frequencies, phase, "k-", lw=1.5)
    axes[1].plot(frequencies, other, color="#C00000", lw=1.5, marker="o", ms=4,
                 markevery=max(1, frequencies.size // 7))
    axes[1].set_ylabel("Phase (rad)")
    for i, axis in enumerate(axes):
        axis.set_xlabel("Frequency (Hz)")
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.tick_params(which="both", direction="in", top=True, right=True)
        axis.text(-0.15, 1.04, f"({chr(97+i)})", transform=axis.transAxes,
                  fontsize=10, fontweight="bold", va="bottom")
        if 0 < np.max(np.abs(axis.get_ylim())) < 0.1:
            axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
    save_figure(figure, path)
    plt.close(figure)


def parabolic_log_peak(frequency, amplitude):
    index = int(np.argmax(amplitude))
    if index in (0, amplitude.size - 1):
        return float(frequency[index])
    y = np.log(np.maximum(amplitude[index-1:index+2], 1.e-30))
    denominator = y[0] - 2 * y[1] + y[2]
    return float(frequency[index] + 0.5 * (y[0] - y[2]) / denominator)


def local_check(case, state, target, bracket, observation, cache, result, operators, system, out):
    name = state["name"] + "__" + target
    mu = np.asarray(state["mu"])
    sign = 1.0 if "dip" in target else -1.0
    direct_scalar = lambda f: abs(cache(f)[observation])

    def learned_scalar(f):
        value = predict_parametric_observation_members(result, np.asarray([f]), mu, observation)
        return float(abs(value.mean(axis=0)[0, 0]))

    start = time.perf_counter()
    print(f"[{case}] {name}: {bracket}, observation={observation}", flush=True)
    coarse = minimize_scalar(lambda f: sign * direct_scalar(f), bounds=bracket,
                             method="bounded", options={"xatol": 1.e-4, "maxiter": 70})
    fine = minimize_scalar(lambda f: sign * direct_scalar(f), bounds=bracket,
                           method="bounded", options={"xatol": 1.e-7, "maxiter": 100})
    predicted = minimize_scalar(lambda f: sign * learned_scalar(f), bounds=bracket,
                                method="bounded", options={"xatol": 1.e-7, "maxiter": 100})
    if not (coarse.success and fine.success and predicted.success):
        raise RuntimeError(f"Local optimizer failed for {name}")
    direct_f, learned_f = float(fine.x), float(predicted.x)
    direct_amp, learned_amp = direct_scalar(direct_f), learned_scalar(learned_f)
    width = None
    half_power = [None, None]
    if sign < 0:
        level = direct_amp / np.sqrt(2.0)
        if direct_scalar(bracket[0]) < level and direct_scalar(bracket[1]) < level:
            left = brentq(lambda f: direct_scalar(f)-level, bracket[0], direct_f, xtol=1.e-7)
            right = brentq(lambda f: direct_scalar(f)-level, direct_f, bracket[1], xtol=1.e-7)
            half_power, width = [float(left), float(right)], float(right-left)
    # Resolved local error windows are sampled with NEW direct solves. They are
    # not interpolations of the manuscript's 1 Hz curves.
    if width is not None:
        lo, hi = max(bracket[0], direct_f - 1.5*width), min(bracket[1], direct_f + 1.5*width)
    else:
        lo, hi = bracket
    frequencies = np.unique(np.r_[np.linspace(lo, hi, 31), direct_f, learned_f])
    direct = np.asarray([cache(f) for f in frequencies])
    learned = predict_parametric_members(result, frequencies, mu).mean(axis=0)[0]
    difference = learned[:, observation] - direct[:, observation]
    w_direct = operators.expand(direct)[:, :system.nodes.shape[0]]
    w_learned = operators.expand(learned)[:, :system.nodes.shape[0]]
    field_relative = np.linalg.norm(learned-direct, axis=1) / np.linalg.norm(direct, axis=1)
    w_relative = np.linalg.norm(w_learned-w_direct, axis=1) / np.linalg.norm(w_direct, axis=1)
    phase_error = np.angle(learned[:, observation] * np.conj(direct[:, observation]))
    local_scale = float(np.max(abs(direct[:, observation])))
    # Five direct samples check that the refined extremum is locally stable;
    # the frequency search itself is repeated at 1000x smaller tolerance.
    step = max(1.e-4, (width if width else hi-lo) * 0.01)
    probe_f = direct_f + step * np.arange(-2, 3)
    probe_f = probe_f[(probe_f >= bracket[0]) & (probe_f <= bracket[1])]
    probe_amp = np.asarray([direct_scalar(f) for f in probe_f])
    old_frequency = np.round(direct_f) + np.arange(-2, 3, dtype=float)
    old_frequency = old_frequency[(old_frequency >= bracket[0]) & (old_frequency <= bracket[1])]
    old_amplitude = np.asarray([direct_scalar(f) for f in old_frequency])
    row = {"case": case, "state": state["name"], "target": target,
           "observation_constrained_index": int(observation),
           "bracket_min_hz": float(bracket[0]), "bracket_max_hz": float(bracket[1]),
           "ellipsoid_radius_squared": state["ellipsoid_radius_squared"],
           "direct_frequency_hz": direct_f, "surrogate_frequency_hz": learned_f,
           "frequency_error_hz": abs(learned_f-direct_f),
           "direct_extremum_mm": direct_amp*1000, "surrogate_extremum_mm": learned_amp*1000,
           "extremum_amplitude_error_percent": 100*abs(learned_amp-direct_amp)/direct_amp,
           "surrogate_at_direct_extremum_mm": learned_scalar(direct_f)*1000,
           "half_power_width_hz": width,
           "reference_refinement_frequency_change_hz": abs(direct_f-float(coarse.x)),
           "reference_refinement_amplitude_change_mm": abs(direct_amp-direct_scalar(coarse.x))*1000,
           "bounded_search_floating_resolution_hz": float(np.sqrt(np.finfo(float).eps)*abs(direct_f)+1.e-7/3),
           "reference_probe_step_hz": step,
           "reference_is_interior": bool(min(direct_f-bracket[0], bracket[1]-direct_f) > 1.e-4),
           "local_complex_relative_l2_percent": 100*float(np.linalg.norm(difference)/np.linalg.norm(direct[:, observation])),
           "local_amplitude_relative_l2_percent": 100*float(np.linalg.norm(abs(learned[:, observation])-abs(direct[:, observation]))/np.linalg.norm(abs(direct[:, observation]))),
           "local_max_complex_error_over_max_reference_percent": 100*float(np.max(abs(difference)))/local_scale,
           "phase_wrapped_rmse_rad": float(np.sqrt(np.mean(phase_error**2))),
           "phase_wrapped_max_rad": float(np.max(abs(phase_error))),
           "mixed_field_relative_l2_max_percent": 100*float(np.max(field_relative)),
           "transverse_field_relative_l2_max_percent": 100*float(np.max(w_relative)),
           "local_frequency_points": int(frequencies.size),
           "wall_seconds": time.perf_counter()-start}
    if sign < 0 and old_frequency.size > 2:
        row.update(grid_1hz_peak_frequency_hz=float(old_frequency[np.argmax(old_amplitude)]),
                   grid_1hz_peak_amplitude_mm=float(np.max(old_amplitude))*1000,
                   grid_1hz_parabolic_log_peak_hz=parabolic_log_peak(old_frequency, old_amplitude),
                   grid_1hz_peak_amplitude_underestimate_percent=100*(1-float(np.max(old_amplitude))/direct_amp))
    cache.save()
    np.savez_compressed(out / (name + ".npz"), frequency_hz=frequencies, mu=mu,
                        direct=direct, surrogate=learned, probe_frequency_hz=probe_f,
                        probe_direct_amplitude=probe_amp, grid_1hz_frequency_hz=old_frequency,
                        grid_1hz_direct_amplitude=old_amplitude,
                        half_power_frequency_hz=np.asarray(half_power, dtype=float))
    write_json(out / (name + ".json"), row)
    local_plot(out / name, frequencies, direct, learned, observation)
    print(f"  completed f_dir={direct_f:.9f}, f_NN={learned_f:.9f}, width={width}, new_solves={cache.new_solves}", flush=True)
    return row


def run_case(case, args):
    out = args.out / case
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with deterministic_eigsh():
        metadata, system, operators, family, result, center = load_parametric_case(SOURCES[case], device="cpu")
    production = family.modal.reference
    target_node = int(np.argmin(np.linalg.norm(system.nodes[:, :2]-np.asarray([7/30, 7/30]), axis=1)))
    matches = np.flatnonzero(np.asarray(operators.free_dofs) == target_node)
    if matches.size != 1:
        raise ValueError("Fixed transfer coordinate is not a unique unconstrained displacement DOF")
    transfer = int(matches[0])
    identity = matrix_hash(operators)
    clusters = modal_evidence(operators, system, production, center, transfer, out)
    states = states_for(case, family.bounds)
    write_json(out / "states.json", states)
    write_json(out / "provenance.json", {
        "case": case, "checkpoint": str(SOURCES[case] / "parametric_operator.pt"),
        "checkpoint_sha256": digest(SOURCES[case] / "parametric_operator.pt"),
        "metadata_sha256": digest(SOURCES[case] / "metadata.json"),
        "operator_sha256": identity, "production_modes": production.n_modes,
        "production_frequency_hz": production.natural_frequencies_hz.tolist(),
        "modal_note": "Deterministically reconstructed eigenpairs; original training eigenpairs were not stored in the checkpoint. This is not a bitwise restoration of the training pole branch.",
        "eigsh_start": "default_rng(20261103).standard_normal(n), Euclidean normalized; regenerated for every call",
        "wide_modes": 40, "cluster_frequency_tolerance": "max(0.01 Hz, 2e-5 times frequency)",
        "center_constrained_index": center, "transfer_constrained_index": transfer,
        "transfer_physical_node": target_node, "transfer_xy": system.nodes[target_node, :2].tolist(),
        "diagonal_regularization": operators.diagonal_regularization,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "torch": torch.__version__, "threads": 2,
        "scope": "Deterministic in-ellipsoid local checks of the fixed linear discrete family; no continuum/external validation or global certificate"})
    first_frequency = float(production.natural_frequencies_hz[0])
    rows, edge_rows = [], []
    for state in states:
        if args.states and state["name"] not in args.states:
            continue
        mu = np.asarray(state["mu"])
        cache = DirectCache(family, mu, out / (state["name"] + "_direct_cache.npz"), identity)
        scale = state["frequency_scale"]
        if not state["name"].startswith("entry_"):
            first = first_frequency * scale
            half = max(1.0, 4*mu[3]*first)
            rows.append(local_check(case, state, "first_peak", [first-half, first+half], center,
                                    cache, result, operators, system, out))
            if state["name"] in ("nominal", "eta_min"):
                dip = (183.0 if case == "flat" else 432.0) * scale
                rows.append(local_check(case, state, "transfer_dip", [dip-12*scale, dip+12*scale], transfer,
                                        cache, result, operators, system, out))
        if case == "stiffened" and state["name"] in ("nominal", "eta_min"):
            second = float(production.natural_frequencies_hz[-1]) * scale
            half = max(4.0, 4*mu[3]*second)
            rows.append(local_check(case, state, "second_peak", [second-half, min(600.0, second+half)], center,
                                    cache, result, operators, system, out))
        if state["name"].startswith("entry_"):
            for cluster in clusters:
                nominal_f = cluster["frequency_mean_hz"]
                if nominal_f <= 600 or cluster["within_production_count"] or nominal_f*scale > 600:
                    continue
                nominal_f = float(nominal_f)
                actual = nominal_f*scale
                frequencies = actual + np.linspace(-4.0, 4.0, 17)
                frequencies = frequencies[(frequencies >= 0) & (frequencies <= 600)]
                direct = np.asarray([cache(f) for f in frequencies])
                learned = predict_parametric_members(result, frequencies, mu).mean(axis=0)[0]
                wd, wl = operators.expand(direct)[:, :system.nodes.shape[0]], operators.expand(learned)[:, :system.nodes.shape[0]]
                edge = {"state": state["name"], "cluster_modes": cluster["modes"],
                        "nominal_frequency_hz": nominal_f, "scaled_frequency_hz": actual,
                        "force_relative_to_largest_cluster": cluster["force_relative_to_largest_cluster"],
                        "negligible_under_fixed_uniform_load": cluster["negligible_at_relative_1e6"],
                        "direct_center_max_mm": float(np.max(abs(direct[:, center])))*1000,
                        "direct_transfer_max_mm": float(np.max(abs(direct[:, transfer])))*1000,
                        "direct_transverse_field_max_mm": float(np.max(abs(wd)))*1000,
                        "center_complex_relative_l2_percent": 100*float(np.linalg.norm(learned[:, center]-direct[:, center])/np.linalg.norm(direct[:, center])),
                        "transfer_complex_relative_l2_percent": 100*float(np.linalg.norm(learned[:, transfer]-direct[:, transfer])/np.linalg.norm(direct[:, transfer])),
                        "transverse_field_relative_l2_percent": 100*float(np.linalg.norm(wl-wd)/np.linalg.norm(wd)),
                        "observation": "Raw direct response around the entered nominal cluster; no peak is attributed to a negligible modal-force cluster"}
                edge_rows.append(edge)
                name = state["name"] + "__entry_cluster_" + cluster["modes"].replace(";", "_")
                np.savez_compressed(out / (name + ".npz"), frequency_hz=frequencies, mu=mu,
                                    direct=direct, surrogate=learned)
                local_plot(out / name, frequencies, direct, learned, center)
                print(f"[{case}] {name}: scaled {actual:.3f} Hz, relative force {cluster['force_relative_to_largest_cluster']:.3e}", flush=True)
                # The first omitted entering cluster is the predeclared edge test.
                break
        cache.save()
        write_csv(out / "local_extrema.csv", rows)
        write_csv(out / "entry_checks.csv", edge_rows)
        write_json(out / "progress.json", {"last_completed_state": state["name"], "extrema_checks": len(rows),
                                           "entry_checks": len(edge_rows), "elapsed_seconds": time.perf_counter()-start})
    summary = {"case": case, "extrema": rows, "entry": edge_rows, "modal_clusters": clusters,
               "wall_seconds": time.perf_counter()-start}
    write_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--states", nargs="+", default=None, help="Optional named subset for staged execution")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for case in args.cases:
        run_case(case, args)


if __name__ == "__main__":
    main()

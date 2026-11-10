# -*- coding: utf-8 -*-
"""Route-B pilot: non-proportional stiffener-stiffness case (changing modal basis).

The pilot uses a square plate with a single embedded stiffener line whose
stiffness scale ``s_b`` is an independent uncertain parameter:

    K(s_b) = K_plate + s_b * K_stiffener,
    M(s_rho) = s_rho * M0,
    f(s_p) = s_p * f0,
    eta uncertain.

Because ``s_b`` scales only the beam contribution, the mode shapes and modal
participation factors change with the parameters; the case therefore probes
the claim that a proportional pole branch cannot be assumed.  The pilot:

    1. quantifies the modal-basis change with MAC(s_b=0.5, s_b=1.5);
    2. evaluates classical modal baselines with nominal and with
       parameter-dependent modes (modal superposition and mode acceleration
       with static correction);
    3. trains two pole-free neural variants (direct_data, direct_physics) on a
       reduced budget and reports observation-FRF accuracy, showing whether
       the equilibrium-residual loss helps when no analytic pole branch is
       available.

Resource policy: one worker thread, ndiv = 20 mesh, 8 training realizations,
six test realizations, 300 epochs and a single ensemble member.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from rbf_fd_full_order import (
    ParameterBounds,
    ParametricOperatorConfig,
    predict_parametric_members,
    prepare_constrained_operators,
    save_parametric_operator,
)
from rbf_fd_full_order.core.frequency_system import ConstrainedOperators
from rbf_fd_full_order.core.parametric_family import ParametricFrequencyFamily
from rbf_fd_full_order.core.rbf_fd_operator import (
    beam_section,
    embedded_beam_line_ops,
    square_nodes_and_quadrature,
    square_simple_support_fixed_dofs,
    weak_plate_matrices,
)
from rbf_fd_full_order.examples.cases import MeshConfig
from rbf_fd_full_order.training.parametric_operator import (
    ParametricOperatorResult,
    _frequency_bounds,
    _train_one_member,
    build_parametric_training_dataset,
)


class ScaledStiffenerFamily(ParametricFrequencyFamily):
    """Four-parameter family with non-proportional stiffener stiffness."""

    def __init__(
        self,
        nominal: ConstrainedOperators,
        bounds: ParameterBounds,
        kp: sp.csr_matrix,
        kb: sp.csr_matrix,
        m0: sp.csr_matrix,
        f0: np.ndarray,
    ) -> None:
        super().__init__(nominal=nominal, bounds=bounds, modal=None)
        self.kp = sp.csr_matrix(kp)
        self.kb = sp.csr_matrix(kb)
        self.m0 = sp.csr_matrix(m0)
        self.f0 = np.asarray(f0, dtype=complex).reshape(-1)

    def _dynamic(self, frequency_hz: float, parameters):
        mu = self.validate_parameters(parameters, allow_extrapolation=True).reshape(-1)
        s_b, s_rho, _, eta = mu
        omega_sq = (2.0 * np.pi * float(frequency_hz)) ** 2
        stiffness = (self.kp + s_b * self.kb).tocsr()
        dynamic = ((1.0 + 1.0j * eta) * stiffness - omega_sq * s_rho * self.m0).astype(complex)
        if self.nominal.diagonal_regularization <= 0.0:
            return dynamic
        diagonal_scale = max(float(np.max(np.abs(dynamic.diagonal()))), 1.0)
        shift = self.nominal.diagonal_regularization * diagonal_scale
        return (dynamic + shift * sp.eye(self.n_free, dtype=complex, format="csr")).tocsr()

    def rhs(self, parameters):
        mu = self.validate_parameters(parameters, allow_extrapolation=True).reshape(-1)
        return np.asarray(mu[2] * self.f0, dtype=complex)


def _bounds() -> ParameterBounds:
    return ParameterBounds(
        lower=np.asarray([0.5, 0.95, 0.90, 0.005], dtype=float),
        upper=np.asarray([1.5, 1.05, 1.10, 0.020], dtype=float),
    ).validated()


def _build_matrices(ndiv: int = 20):
    nodes, qpts, qw = square_nodes_and_quadrature(ndiv)
    kp, mp, f, info = weak_plate_matrices(
        nodes, qpts, qw, stencil_size=MeshConfig().plate_stencil,
        poly_degree=MeshConfig().plate_poly_degree,
        condition_limit=MeshConfig().rbf_condition_limit,
        thickness=0.010,
        pressure=1000.0,
    )
    n = nodes.shape[0]
    sec = beam_section()
    kb = sp.csr_matrix(kp.shape)
    mb = sp.csr_matrix(mp.shape)
    axis, fixed = "x", 0.5
    selector, derivative, line_w, _ = embedded_beam_line_ops(
        nodes, axis, fixed, ndiv=ndiv, stencil_size=7
    )
    w_ei = sp.diags(line_w * sec["ei"], format="csr")
    w_ga = sp.diags(line_w * sec["kga"], format="csr")
    w_m = sp.diags(line_w * sec["rho_a"], format="csr")
    w_i = sp.diags(line_w * sec["rho_i"], format="csr")
    kww = derivative.T @ w_ga @ derivative
    kwt = -derivative.T @ w_ga @ selector
    ktt = derivative.T @ w_ei @ derivative + selector.T @ w_ga @ selector
    mw = selector.T @ w_m @ selector
    mt = selector.T @ w_i @ selector
    zero = sp.csr_matrix((n, n))
    kb = sp.bmat([[kww, kwt, zero], [kwt.T, ktt, zero], [zero, zero, zero]], format="csr")
    mb = sp.bmat([[mw, zero, zero], [zero, mt, zero], [zero, zero, zero]], format="csr")
    fixed_dofs = square_simple_support_fixed_dofs(nodes)
    return nodes, kp, kb, mp, mb, f, fixed_dofs


def _constrained(
    kp: sp.csr_matrix, kb: sp.csr_matrix, mp: sp.csr_matrix, mb: sp.csr_matrix,
    f: np.ndarray, fixed_dofs: np.ndarray, source,
) -> tuple[ConstrainedOperators, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    full_dimension = kp.shape[0]
    mask = np.ones(full_dimension, dtype=bool)
    mask[np.asarray(fixed_dofs, dtype=int)] = False
    free = np.flatnonzero(mask)
    kp_c = kp[free][:, free].tocsr()
    kb_c = kb[free][:, free].tocsr()
    mp_c = mp[free][:, free].tocsr()
    mb_c = mb[free][:, free].tocsr()
    f_c = np.asarray(f[free], dtype=complex)
    nominal = ConstrainedOperators(
        source=source,
        K=(kp_c + kb_c).tocsr(),
        M=(mp_c + mb_c).tocsr(),
        rhs=f_c,
        reduction_mode="fixed_dofs",
        full_dimension=full_dimension,
        free_dofs=free,
        diagonal_regularization=1.0e-12,
    )
    return nominal, kp_c, kb_c, (mp_c + mb_c).tocsr(), f_c


def _transverse_observation_index(free_dofs: np.ndarray, node_count: int, nominal_response: np.ndarray) -> int:
    candidates = np.flatnonzero(np.asarray(free_dofs) < int(node_count))
    local = int(np.argmax(np.max(np.abs(nominal_response[:, candidates]), axis=0)))
    return int(candidates[local])


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - reference) / max(float(np.linalg.norm(reference)), 1.0e-30))


def _modal_projection(family: ScaledStiffenerFamily, frequencies, modes, eigenvalues, parameters):
    """Modal-superposition observation with parameter-dependent eigenpairs."""
    s_b, s_rho, s_p, eta = parameters
    stiffness = (family.kp + s_b * family.kb).tocsr()
    load = s_p * family.f0
    modal_forces = modes.T @ load
    omega_sq = (2.0 * np.pi * frequencies)[:, None] ** 2
    denominator = (1.0 + 1.0j * eta) * eigenvalues[None, :] - omega_sq * s_rho
    coefficients = modal_forces[None, :] / denominator
    return np.asarray(coefficients @ modes.T, dtype=complex)


def _nominal_modes(family: ScaledStiffenerFamily, n_modes: int):
    k0 = (family.kp + family.kb).tocsr()
    eigenvalues, mode_shapes = spla.eigsh(k0, k=n_modes, M=family.m0, sigma=0.0, which="LM")
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    mode_shapes = mode_shapes[:, order]
    return eigenvalues, mode_shapes


def _parameter_modes(family: ScaledStiffenerFamily, parameters: np.ndarray, n_modes: int):
    s_b = float(parameters[0])
    k = (family.kp + s_b * family.kb).tocsr()
    eigenvalues, mode_shapes = spla.eigsh(k, k=n_modes, M=family.m0, sigma=0.0, which="LM")
    order = np.argsort(eigenvalues)
    return eigenvalues[order], mode_shapes[:, order]


def _static_corrected(family: ScaledStiffenerFamily, frequencies, parameters, n_modes: int):
    s_b, s_rho, s_p, eta = parameters
    k = (family.kp + s_b * family.kb).tocsr()
    load = s_p * family.f0
    a0 = ((1.0 + 1.0j * eta) * k).tocsc()
    static = spla.spsolve(a0, load)
    eigenvalues, mode_shapes = _parameter_modes(family, parameters, n_modes)
    modal_forces = mode_shapes.T @ load
    omega_sq = (2.0 * np.pi * frequencies)[:, None] ** 2
    denominator = (1.0 + 1.0j * eta) * eigenvalues[None, :] - omega_sq * s_rho
    coefficients = modal_forces[None, :] / denominator
    residual = static - np.asarray(mode_shapes @ (modal_forces / ((1.0 + 1.0j * eta) * eigenvalues)), dtype=complex)
    return np.asarray(coefficients @ mode_shapes.T, dtype=complex) + residual[None, :]


def _mac(shape_a: np.ndarray, shape_b: np.ndarray) -> float:
    a = shape_a / max(float(np.linalg.norm(shape_a)), 1.0e-30)
    b = shape_b / max(float(np.linalg.norm(shape_b)), 1.0e-30)
    return float(abs(np.vdot(a, b)) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results") / "route_b_pilot")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=6)
    parser.add_argument("--n-labeled", type=int, default=24)
    parser.add_argument("--n-physics", type=int, default=51)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    bounds = _bounds()

    print("[routeB] assembling single-stiffener system", flush=True)
    nodes, kp, kb, mp, mb, f, fixed_dofs = _build_matrices(ndiv=20)
    source = type("Source", (), {"metadata": {"fixed_dofs": fixed_dofs}})()
    nominal, kp_c, kb_c, m0_c, f_c = _constrained(kp, kb, mp, mb, f, fixed_dofs, source)
    family = ScaledStiffenerFamily(nominal, bounds, kp_c, kb_c, m0_c, f_c)
    n_free = family.n_free
    print(f"[routeB] n_free = {n_free}", flush=True)

    frequencies = np.linspace(0.0, 200.0, 201)
    labeled = np.linspace(0.0, 200.0, int(args.n_labeled))
    physics = np.unique(np.r_[labeled, np.linspace(0.0, 200.0, int(args.n_physics))])
    train_mu = bounds.sample_latin_hypercube(int(args.n_train), 21 + 101)
    test_mu = bounds.sample_latin_hypercube(int(args.n_test), 21 + 303)

    print("[routeB] direct test reference", flush=True)
    reference = np.asarray(family.solve_grid(frequencies, test_mu), dtype=complex)
    nominal_response = np.asarray(family.solve_grid(frequencies, bounds.center), dtype=complex)
    observation_index = _transverse_observation_index(
        np.asarray(nominal.free_dofs), nodes.shape[0], nominal_response
    )
    print(f"[routeB] observation index = {observation_index}", flush=True)

    # Modal-basis change: MAC between s_b extremes.
    mu_low = np.asarray([0.5, 1.0, 1.0, 0.0125])
    mu_high = np.asarray([1.5, 1.0, 1.0, 0.0125])
    lam_low, phi_low = _parameter_modes(family, mu_low, 8)
    lam_high, phi_high = _parameter_modes(family, mu_high, 8)
    mac_rows = [
        {
            "mode": index + 1,
            "f_low_hz": float(np.sqrt(lam_low[index]) / (2.0 * np.pi)),
            "f_high_hz": float(np.sqrt(lam_high[index]) / (2.0 * np.pi)),
            "mac_low_vs_high": _mac(phi_low[:, index], phi_high[:, index]),
        }
        for index in range(8)
    ]

    # Classical baselines on the test set.
    lam_nom, phi_nom = _nominal_modes(family, 16)
    f_nom = np.sqrt(lam_nom) / (2.0 * np.pi)
    active_nominal = int(np.sum(f_nom <= 200.0))
    baselines = {
        "ms_nominal": [],
        "pole_only_nominal": [],
        "ms_param": [],
        "ma_sc_param": [],
    }
    for index, mu in enumerate(test_mu):
        ms_nom = _modal_projection(family, frequencies, phi_nom[:, :16], lam_nom[:16], mu)
        baselines["ms_nominal"].append(ms_nom)
        pole = _modal_projection(family, frequencies, phi_nom[:, :active_nominal], lam_nom[:active_nominal], mu)
        baselines["pole_only_nominal"].append(pole)
        lam_p, phi_p = _parameter_modes(family, mu, 16)
        baselines["ms_param"].append(_modal_projection(family, frequencies, phi_p, lam_p, mu))
        baselines["ma_sc_param"].append(_static_corrected(family, frequencies, mu, 8))

    ref_obs = np.abs(reference[..., observation_index]) * 1.0e3
    baseline_rows = []
    for name, predictions in baselines.items():
        predictions = np.asarray(predictions)
        pred_obs = np.abs(predictions[..., observation_index]) * 1.0e3
        errors = np.asarray([_relative_l2(pred_obs[i], ref_obs[i]) for i in range(pred_obs.shape[0])])
        baseline_rows.append(
            {
                "method": name,
                "median_observation_relative_l2_percent": 100.0 * float(np.median(errors)),
                "max_observation_relative_l2_percent": 100.0 * float(np.max(errors)),
            }
        )
        print(f"[routeB] {name}: median obs error = {100*np.median(errors):.4f}%", flush=True)

    # Neural variants without a pole branch.
    dataset = build_parametric_training_dataset(
        family, train_mu, labeled, physics, representation="pod", pod_modes=8
    )
    frequency_min, frequency_range = _frequency_bounds(labeled, physics)
    neural_rows = []
    for variant, physics_weight in (("direct_data", 0.0), ("direct_physics", 0.02)):
        config = ParametricOperatorConfig(
            representation="pod", epochs=int(args.epochs), learning_rate=2.0e-4,
            width=96, depth=3, fourier_modes=4, pod_modes=8, ensemble_size=1,
            physics_weight=physics_weight, warmup_epochs=max(int(args.epochs) // 4, 1),
            log_every=max(int(args.epochs) // 20, 1), device=args.device,
            modal_resonance=None,
        )
        print(f"[routeB] training {variant}", flush=True)
        started = time.perf_counter()
        members = [
            _train_one_member(family, dataset, config, frequency_min, frequency_range, seed=21 + 1009 * index)
            for index in range(config.ensemble_size)
        ]
        result = ParametricOperatorResult(
            family=family, members=members, config=config, bounds=bounds,
            frequency_min=frequency_min, frequency_range=frequency_range,
            n_dofs=family.n_free, response_scaler=dataset.response_scaler,
            coefficient_scale=dataset.coefficient_scale, pod_basis=dataset.pod_basis,
            training_parameters=np.asarray(train_mu, dtype=float),
            labeled_frequencies_hz=labeled, physics_frequencies_hz=physics,
            high_fidelity_generation_s=dataset.high_fidelity_generation_s,
            high_fidelity_solve_count=dataset.high_fidelity_solve_count,
            ensemble_training_s=float(sum(member.training_s for member in members)),
        )
        checkpoint = args.out / f"{variant}.pt"
        save_parametric_operator(result, checkpoint)
        training_s = time.perf_counter() - started
        predicted = np.mean(predict_parametric_members(result, frequencies, test_mu), axis=0)
        pred_obs = np.abs(predicted[..., observation_index]) * 1.0e3
        errors = np.asarray([_relative_l2(pred_obs[i], ref_obs[i]) for i in range(pred_obs.shape[0])])
        row = {
            "method": variant,
            "median_observation_relative_l2_percent": 100.0 * float(np.median(errors)),
            "max_observation_relative_l2_percent": 100.0 * float(np.max(errors)),
            "training_s": round(float(training_s), 2),
        }
        neural_rows.append(row)
        print(
            f"[routeB] {variant}: median obs error = {100*np.median(errors):.4f}% "
            f"[{training_s:.0f} s]",
            flush=True,
        )

    summary = {
        "case": "single_stiffener_sb",
        "bounds": {"lower": bounds.lower.tolist(), "upper": bounds.upper.tolist()},
        "n_free": int(n_free),
        "observation_index": int(observation_index),
        "mac_low_vs_high": mac_rows,
        "baselines": baseline_rows,
        "neural": neural_rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Route-B pilot written to {args.out}", flush=True)


if __name__ == "__main__":
    main()

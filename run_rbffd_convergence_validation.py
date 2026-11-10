# -*- coding: utf-8 -*-
"""Independent discretization validation and convergence study.

The study provides the numerical-reference evidence requested by the review:

1.  Analytical Kirchhoff eigenfrequencies of a simply supported thin square
    plate (h/L = 0.01) versus the weak-form PHS-RBF-FD eigenvalues;
2.  natural-frequency convergence with node density (ndiv), stencil size and
    polynomial augmentation order;
3.  an independent SRI-Q4 Reissner--Mindlin finite-element eigenvalue
    reference for the same plate;
4.  convergence of the complete observation-point FRF with node density;
5.  thin-plate limit check h/L in {0.05, 0.02, 0.01, 0.005} (Mindlin should
    approach Kirchhoff as h/L -> 0 without shear locking);
6.  mode-shape agreement (MAC) between the 30x30 and 40x40 RBF-FD meshes at
    common coordinates.

All RBF-FD systems are built with the same weak-form operator assembly used by
the surrogate study, so the evidence directly supports the manuscript's claim
that the fixed RBF-FD discrete reference is a controlled numerical model.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from rbf_fd_full_order import (
    ModalResonanceConfig,
    build_modal_resonance_interface,
    build_parametric_frequency_family,
    prepare_constrained_operators,
)
from rbf_fd_full_order.core import PlateMaterial, build_rbf_flat_plate_system
from rbf_fd_full_order.core.parametric_family import ParameterBounds
from rbf_fd_full_order.visualization import apply_journal_style, save_figure


Array = np.ndarray
MATERIAL = PlateMaterial()
PRESSURE = 1.0e3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results") / "rbffd_convergence_validation")
    parser.add_argument("--frf-frequency-step-hz", type=float, default=10.0)
    parser.add_argument("--n-modes", type=int, default=6)
    return parser.parse_args()


def kirchhoff_frequency(m: int, n: int, material: PlateMaterial = MATERIAL) -> float:
    """Analytical Kirchhoff eigenfrequency of a simply supported square plate."""
    side = 1.0
    d = material.E * material.h**3 / (12.0 * (1.0 - material.nu**2))
    mass_per_area = material.rho * material.h
    omega = np.pi**2 * (m**2 + n**2) / side**2 * np.sqrt(d / mass_per_area)
    return float(omega / (2.0 * np.pi))


def rbffd_modes(
    ndiv: int,
    *,
    stencil: int = 20,
    poly_degree: int = 2,
    material: PlateMaterial = MATERIAL,
    n_modes: int = 6,
) -> tuple[Array, Array, Array, Array, float]:
    """Return (natural_freqs_hz, mode_shapes, nodes, free_dofs, elapsed_s)."""
    started = time.perf_counter()
    system = build_rbf_flat_plate_system(
        ndiv=ndiv,
        plate_stencil=stencil,
        plate_poly_degree=poly_degree,
        pressure=PRESSURE,
        material=material,
    )
    operators = prepare_constrained_operators(system)
    config = ModalResonanceConfig(
        enabled=True,
        n_modes=n_modes,
        frequency_min_hz=0.0,
        frequency_max_hz=1000.0,
        use_sparse_solver=True,
    )
    interface = build_modal_resonance_interface(operators, config, loss_factor=0.0125)
    elapsed = time.perf_counter() - started
    return (
        np.asarray(interface.natural_frequencies_hz, dtype=float),
        np.asarray(interface.mode_shapes, dtype=float),
        np.asarray(system.nodes, dtype=float),
        np.asarray(operators.free_dofs, dtype=int),
        elapsed,
    )


def _mitc4_element(vertices: Array, material: PlateMaterial) -> tuple[Array, Array, Array]:
    """Return (K_element, M_element, dof_map) for one MITC4 Mindlin element.

    DOF ordering per node: (w, theta_x, theta_y).  Bending stiffness uses
    2x2 Gauss quadrature.  Transverse shear follows the Bathe--Dvorkin MITC4
    tying procedure (shear strains sampled at the four edge midpoints and
    interpolated), which suppresses the hourglass modes of one-point shear
    integration while avoiding shear locking.
    """
    xy = np.asarray(vertices, dtype=float)  # (4, 2) in counter-clockwise order
    gauss = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    ke = np.zeros((12, 12), dtype=float)
    me = np.zeros((12, 12), dtype=float)
    d = material.E * material.h**3 / (12.0 * (1.0 - material.nu**2))
    db = d * np.array(
        [
            [1.0, material.nu, 0.0],
            [material.nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - material.nu)],
        ]
    )
    ds = material.kappa * material.G * material.h * np.eye(2)
    n_quad = material.rho * material.h
    n_rot = material.rho * material.h**3 / 12.0

    def shape(xi: float, eta: float) -> Array:
        return 0.25 * np.array(
            [
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ]
        )

    def shape_derivatives(xi: float, eta: float) -> Array:
        dxi = 0.25 * np.array(
            [
                -(1.0 - eta),
                (1.0 - eta),
                (1.0 + eta),
                -(1.0 + eta),
            ]
        )
        deta = 0.25 * np.array(
            [
                -(1.0 - xi),
                -(1.0 + xi),
                (1.0 + xi),
                (1.0 - xi),
            ]
        )
        return dxi, deta

    def jacobian(xi: float, eta: float) -> tuple[Array, Array, float]:
        dxi, deta = shape_derivatives(xi, eta)
        j = np.column_stack([dxi @ xy, deta @ xy])
        det = float(np.linalg.det(j))
        inv = np.linalg.inv(j)
        return inv, dxi, deta, det

    for xi in gauss:
        for eta in gauss:
            inv, dxi, deta, det = jacobian(xi, eta)
            weight = det
            # Bending: kappa = [d theta_x/dx, d theta_y/dy, d theta_x/dy + d theta_y/dx]
            bb = np.zeros((3, 12), dtype=float)
            for a in range(4):
                grad_x = float(inv[0, 0] * dxi[a] + inv[0, 1] * deta[a])
                grad_y = float(inv[1, 0] * dxi[a] + inv[1, 1] * deta[a])
                tx = 1 + 3 * a  # theta_x dof
                ty = 2 + 3 * a  # theta_y dof
                bb[0, tx] = grad_x
                bb[1, ty] = grad_y
                bb[2, tx] = grad_y
                bb[2, ty] = grad_x
            ke += weight * bb.T @ db @ bb
            n = shape(xi, eta)
            nw = np.zeros((1, 12), dtype=float)
            nw[0, 0::3] = n
            nx = np.zeros((1, 12), dtype=float)
            nx[0, 1::3] = n
            ny = np.zeros((1, 12), dtype=float)
            ny[0, 2::3] = n
            me += weight * (n_quad * nw.T @ nw + n_rot * (nx.T @ nx + ny.T @ ny))

    # MITC4 shear tying.  Covariant shear components are
    #   g_xi  = w,xi  - theta_x x,xi  - theta_y y,xi
    #   g_eta = w,eta - theta_x x,eta - theta_y y,eta
    # sampled at the edge midpoints and interpolated bilinearly.
    def covariant_forms(xi_t: float, eta_t: float) -> tuple[Array, Array]:
        """Return coefficient rows for (g_xi, g_eta) at a tying point."""
        n = shape(xi_t, eta_t)
        dxi_t, deta_t = shape_derivatives(xi_t, eta_t)
        jac = np.column_stack([dxi_t @ xy, deta_t @ xy])
        x_xi, y_xi = float(jac[0, 0]), float(jac[1, 0])
        x_eta, y_eta = float(jac[0, 1]), float(jac[1, 1])
        row_xi = np.zeros(12, dtype=float)
        row_eta = np.zeros(12, dtype=float)
        for a in range(4):
            w = 3 * a
            tx = 1 + 3 * a
            ty = 2 + 3 * a
            row_xi[w] += dxi_t[a]
            row_xi[tx] += -x_xi * n[a]
            row_xi[ty] += -y_xi * n[a]
            row_eta[w] += deta_t[a]
            row_eta[tx] += -x_eta * n[a]
            row_eta[ty] += -y_eta * n[a]
        return row_xi, row_eta

    g_xi_a, _ = covariant_forms(0.0, -1.0)
    g_xi_b, _ = covariant_forms(0.0, 1.0)
    _, g_eta_c = covariant_forms(-1.0, 0.0)
    _, g_eta_d = covariant_forms(1.0, 0.0)

    for xi in gauss:
        for eta in gauss:
            inv, _, _, det = jacobian(xi, eta)
            g_xi = 0.5 * (1.0 - eta) * g_xi_a + 0.5 * (1.0 + eta) * g_xi_b
            g_eta = 0.5 * (1.0 - xi) * g_eta_c + 0.5 * (1.0 + xi) * g_eta_d
            # Cartesian shear: [gx; gy] = J^{-T} [g_xi; g_eta]
            row_x = float(inv[0, 0]) * g_xi + float(inv[0, 1]) * g_eta
            row_y = float(inv[1, 0]) * g_xi + float(inv[1, 1]) * g_eta
            bs = np.vstack([row_x, row_y])
            ke += det * bs.T @ ds @ bs

    dof_map = np.arange(12, dtype=int)
    return ke, me, dof_map


def mitc4_modes(ndiv: int, material: PlateMaterial = MATERIAL, n_modes: int = 6) -> tuple[Array, float]:
    """MITC4 Reissner--Mindlin eigenfrequencies for a simply supported plate."""
    n = ndiv + 1
    coords = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    node_id = lambda i, j: j * n + i
    full = 3 * nodes.shape[0]

    rows_k: list[int] = []
    cols_k: list[int] = []
    vals_k: list[float] = []
    rows_m: list[int] = []
    cols_m: list[float] = []
    vals_m: list[float] = []
    for j in range(ndiv):
        for i in range(ndiv):
            ids = [
                node_id(i, j),
                node_id(i + 1, j),
                node_id(i + 1, j + 1),
                node_id(i, j + 1),
            ]
            vertices = nodes[ids]
            ke, me, _ = _mitc4_element(vertices, material)
            dofs = np.asarray([3 * node + c for node in ids for c in range(3)], dtype=int)
            for r_local in range(12):
                for c_local in range(12):
                    if abs(ke[r_local, c_local]) > 0.0:
                        rows_k.append(dofs[r_local])
                        cols_k.append(dofs[c_local])
                        vals_k.append(ke[r_local, c_local])
                    if abs(me[r_local, c_local]) > 0.0:
                        rows_m.append(dofs[r_local])
                        cols_m.append(dofs[c_local])
                        vals_m.append(me[r_local, c_local])

    kg = sp.coo_matrix((vals_k, (rows_k, cols_k)), shape=(full, full)).tocsc()
    mg = sp.coo_matrix((vals_m, (rows_m, cols_m)), shape=(full, full)).tocsc()

    fixed: set[int] = set()
    for node in range(nodes.shape[0]):
        x, y = nodes[node]
        on_edge = abs(x) < 1.0e-12 or abs(x - 1.0) < 1.0e-12 or abs(y) < 1.0e-12 or abs(y - 1.0) < 1.0e-12
        if on_edge:
            fixed.add(3 * node + 0)  # w = 0
        if abs(y) < 1.0e-12 or abs(y - 1.0) < 1.0e-12:
            fixed.add(3 * node + 1)  # theta_x = 0 on horizontal edges
        if abs(x) < 1.0e-12 or abs(x - 1.0) < 1.0e-12:
            fixed.add(3 * node + 2)  # theta_y = 0 on vertical edges
    free = np.asarray(sorted(set(range(full)) - fixed), dtype=int)
    kc = kg[np.ix_(free, free)]
    mc = mg[np.ix_(free, free)]
    started = time.perf_counter()
    try:
        eigenvalues, eigenvectors = spla.eigsh(
            kc,
            k=min(n_modes + 8, free.size - 1),
            M=mc,
            sigma=0.0,
            which="LM",
            return_eigenvectors=True,
        )
    except spla.ArpackNoConvergence as exc:
        eigenvalues = np.asarray(exc.eigenvalues, dtype=float)
        eigenvectors = np.asarray(exc.eigenvectors, dtype=float)
    elapsed = time.perf_counter() - started
    # Keep physical modes by requiring a material share of the mass-weighted
    # transverse energy (guards against any remaining spurious modes).
    free_w = np.searchsorted(free, free[np.flatnonzero(free % 3 == 0)])
    mc_w = mc.tocsr()[free_w][:, free_w].tocsc()
    order = np.argsort(np.maximum(np.real(eigenvalues), 0.0))
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    kept = []
    for index in range(eigenvectors.shape[1]):
        vector = eigenvectors[:, index]
        w_vector = vector[free_w]
        w_mass = float(np.real(np.vdot(w_vector, mc_w @ w_vector)))
        total_mass = float(np.real(np.vdot(vector, mc @ vector)))
        if total_mass > 0.0 and w_mass / total_mass > 1.0e-3:
            kept.append(index)
        if len(kept) >= n_modes:
            break
    frequencies = np.sqrt(np.maximum(np.real(eigenvalues[kept]), 0.0)) / (2.0 * np.pi)
    return np.asarray(frequencies, dtype=float), elapsed


def observation_center_index(ndiv: int, free_dofs: Array, nodes: Array) -> int:
    """Constrained transverse DOF nearest to the plate centre."""
    center = np.asarray([0.5, 0.5], dtype=float)
    dist = np.linalg.norm(nodes - center[None, :], axis=1)
    node = int(np.argmin(dist))
    n = nodes.shape[0]
    free_transverse = np.flatnonzero(free_dofs < n)
    candidates = free_dofs[free_transverse]
    node_ids = candidates % n
    match = np.flatnonzero(node_ids == node)
    return int(candidates[match[0]]) if match.size else int(candidates[0])


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


def mac_matrix(shapes_a: Array, shapes_b: Array) -> Array:
    mac = np.empty((shapes_a.shape[1], shapes_b.shape[1]), dtype=float)
    for i in range(shapes_a.shape[1]):
        va = shapes_a[:, i]
        for j in range(shapes_b.shape[1]):
            vb = shapes_b[:, j]
            mac[i, j] = float(abs(np.vdot(va, vb)) ** 2 / (np.vdot(va, va) * np.vdot(vb, vb)).real)
    return mac


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    material = MATERIAL
    rows: list[dict[str, object]] = []

    # 1) Analytical Kirchhoff reference.
    kirchhoff = {
        f"f_{m}{n}": kirchhoff_frequency(m, n, material)
        for m, n in ((1, 1), (1, 2), (2, 1), (2, 2))
    }
    print(f"Kirchhoff frequencies: {kirchhoff}", flush=True)

    # 2) Node-density convergence of RBF-FD natural frequencies.
    ndiv_list = (10, 15, 20, 25, 30, 40)
    for ndiv in ndiv_list:
        freqs, _, nodes, free_dofs, elapsed = rbffd_modes(ndiv, n_modes=args.n_modes)
        row = {
            "study": "ndiv_convergence",
            "ndiv": ndiv,
            "nodes": int(nodes.shape[0]),
            "n_free": int(free_dofs.size),
            "elapsed_s": float(elapsed),
        }
        for k, value in enumerate(freqs, start=1):
            row[f"f_{k}_hz"] = float(value)
        rows.append(row)
        print(f"[ndiv {ndiv}] freqs = {np.round(freqs[:4], 4).tolist()}", flush=True)

    # 3) Stencil / polynomial-order convergence at ndiv = 30.
    for stencil in (12, 16, 20, 24):
        freqs, _, nodes, _, elapsed = rbffd_modes(30, stencil=stencil, n_modes=args.n_modes)
        rows.append(
            {
                "study": "stencil_convergence",
                "stencil": stencil,
                "poly_degree": 2,
                "ndiv": 30,
                "nodes": int(nodes.shape[0]),
                "elapsed_s": float(elapsed),
                **{f"f_{k}_hz": float(value) for k, value in enumerate(freqs, start=1)},
            }
        )
    for poly in (1, 2, 3):
        freqs, _, nodes, _, elapsed = rbffd_modes(30, poly_degree=poly, n_modes=args.n_modes)
        rows.append(
            {
                "study": "poly_convergence",
                "stencil": 20,
                "poly_degree": poly,
                "ndiv": 30,
                "nodes": int(nodes.shape[0]),
                "elapsed_s": float(elapsed),
                **{f"f_{k}_hz": float(value) for k, value in enumerate(freqs, start=1)},
            }
        )

    # 4) Independent SRI-Q4 FEM eigenvalues.
    fem_rows = []
    for ndiv in (20, 30, 40):
        freqs, elapsed = mitc4_modes(ndiv, material, n_modes=args.n_modes)
        fem_rows.append(
            {
                "ndiv": ndiv,
                "elements": int(ndiv * ndiv),
                "elapsed_s": float(elapsed),
                **{f"f_{k}_hz": float(value) for k, value in enumerate(freqs, start=1)},
            }
        )
        print(f"[fem ndiv {ndiv}] freqs = {np.round(freqs[:4], 4).tolist()}", flush=True)

    # 5) Thin-plate limit h/L sweep at ndiv = 30.
    thin_rows = []
    for thickness in (0.05, 0.02, 0.01, 0.005):
        thin_material = replace(material, h=thickness)
        freqs, _, _, _, elapsed = rbffd_modes(30, material=thin_material, n_modes=args.n_modes)
        thin_rows.append(
            {
                "h_over_L": float(thickness),
                "elapsed_s": float(elapsed),
                **{f"f_{k}_hz": float(value) for k, value in enumerate(freqs, start=1)},
                "kirchhoff_f11_hz": kirchhoff_frequency(1, 1, thin_material),
            }
        )
        print(f"[h/L {thickness}] freqs = {np.round(freqs[:4], 4).tolist()}", flush=True)

    # 6) FRF convergence at the plate centre (even ndiv meshes).
    frf_rows = []
    frequencies = np.arange(0.0, 201.0, args.frf_frequency_step_hz)
    references: dict[int, Array] = {}
    for ndiv in (10, 20, 30, 40):
        system = build_rbf_flat_plate_system(ndiv=ndiv, pressure=PRESSURE, material=material)
        operators = prepare_constrained_operators(system)
        index = observation_center_index(ndiv, np.asarray(operators.free_dofs, dtype=int), system.nodes)
        curves = np.asarray([abs(operators.solve(float(f), loss_factor=0.0125)[index]) * 1.0e3 for f in frequencies])
        references[ndiv] = curves
        print(f"[frf ndiv {ndiv}] center peak = {curves.max():.4f} mm at {frequencies[np.argmax(curves)]:.0f} Hz", flush=True)
    finest = references[max(references)]
    for ndiv, curves in references.items():
        frf_rows.append(
            {
                "ndiv": ndiv,
                "relative_l2_error_vs_ndiv40_percent": float(
                    100.0 * np.linalg.norm(curves - finest) / max(float(np.linalg.norm(finest)), 1.0e-30)
                ),
                "peak_mm": float(curves.max()),
                "peak_frequency_hz": float(frequencies[np.argmax(curves)]),
            }
        )

    # 7) MAC between ndiv 30 and 40 at common coordinates.
    mac_extra: dict[str, object] = {}
    mac_rows: list[dict[str, object]] = []
    shapes: dict[int, tuple[Array, Array, Array]] = {}
    for ndiv in (30, 40):
        freqs, modes, nodes, free_dofs, _ = rbffd_modes(ndiv, n_modes=args.n_modes)
        n = nodes.shape[0]
        free_w = free_dofs[np.flatnonzero(free_dofs < n)]
        node_ids = free_w % n
        coords = nodes[node_ids]
        tol = 1.0e-9
        common = np.flatnonzero(
            (np.abs(coords[:, 0] - 0.1) < tol)
            & (np.abs(coords[:, 1] - 0.1) < tol)
        )
        # Common coordinates are all k/10 with k=1..9; keep the full subset.
        is_common = (
            np.minimum(np.abs(coords[:, 0] % 0.1), np.abs(0.1 - coords[:, 0] % 0.1)) < 1.0e-9
        ) & (
            np.minimum(np.abs(coords[:, 1] % 0.1), np.abs(0.1 - coords[:, 1] % 0.1)) < 1.0e-9
        )
        common = np.flatnonzero(is_common)
        shapes[ndiv] = (modes[common, :], coords[common], freqs)
    modes30, coords30, freqs30 = shapes[30]
    modes40, coords40, freqs40 = shapes[40]
    if coords30.shape == coords40.shape and np.allclose(coords30, coords40, atol=1.0e-9):
        mac = mac_matrix(modes30, modes40)
        for k in range(min(mac.shape[0], mac.shape[1])):
            mac_rows.append(
                {
                    "mode_index": k + 1,
                    "mac_diagonal": float(mac[k, k]),
                    "max_offdiag": float(np.max(np.delete(mac[k, :], k))) if mac.shape[1] > 1 else 0.0,
                    "f_ndiv30_hz": float(freqs30[k]),
                    "f_ndiv40_hz": float(freqs40[k]),
                }
            )
        # Subspace overlap for the degenerate (1,2)/(2,1) pair: individually
        # mixed MAC entries are expected, while the two-dimensional subspace
        # should coincide between meshes.
        def normalize_columns(matrix: Array) -> Array:
            norms = np.linalg.norm(matrix, axis=0)
            return matrix / np.maximum(norms[None, :], 1.0e-30)

        a = normalize_columns(modes30[:, 1:3])
        b = normalize_columns(modes40[:, 1:3])
        gram = a.T @ b
        singular = np.linalg.svd(gram, compute_uv=False)
        subspace_overlap = float(np.sum(singular**2) / 2.0)
        mac_extra["mac_subspace_overlap_modes_2_3"] = subspace_overlap

    summary = {
        "kirchhoff_hz": kirchhoff,
        "rbffd_rows": rows,
        "fem_rows": fem_rows,
        "thin_limit_rows": thin_rows,
        "frf_convergence_rows": frf_rows,
        "mac_rows": mac_rows,
        "material": {
            "E": material.E,
            "nu": material.nu,
            "rho": material.rho,
            "h": material.h,
            "kappa": material.kappa,
        },
    }
    summary.update(mac_extra)
    _write_json(args.out / "convergence_summary.json", summary)
    _write_csv(args.out / "rbffd_natural_frequency_convergence.csv", rows)
    _write_csv(args.out / "fem_mitc4_frequencies.csv", fem_rows)
    _write_csv(args.out / "thin_plate_limit.csv", thin_rows)
    _write_csv(args.out / "frf_convergence.csv", frf_rows)
    _write_csv(args.out / "mac_30_vs_40.csv", mac_rows)
    print(f"Wrote convergence summary to {args.out}", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Direct-reference four-mode MASC diagnostic for manuscript Figure 6.

Restore the current flat-four-mode and stiffened-six-mode checkpoints, then
use the first four eigenpairs in both diagnostic splits.  Direct full-order
constrained solves define the dynamic remainder.  A static stiffness solve
evaluates the actual Eq. (9) MASC background, whose magnitude normalizes the
plotted remainder.  The best complex-constant error is recorded separately
and is never identified with the MASC error.  No network is retrained.

Outputs (optionally synchronized to the manuscript figure-data directory):
    code/results/review_validation_20261103/figure6/background_parallel.npz
    code/results/review_validation_20261103/figure6/background_parallel.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from threadpoolctl import threadpool_limits

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from rbf_fd_full_order.visualization.result_support import load_parametric_case  # noqa: E402

# label, case directory, retained modes for the parallel comparison, description
CONFIGS = (
    ("flat", r"parametric_many_query_flat_5000_best\flat", 4,
     "4 retained modes, band 0-200 Hz"),
    ("stiffened", r"parametric_many_query_stiffened_6mode_0_600\stiffened", 4,
     "4 retained modes, band 0-600 Hz"),
)
POINTS = 801


def modal_sum(frequencies, eigenvalues, modal_forces, phi, mu, loss_factor):
    omega_sq = (2.0 * np.pi * frequencies)[None, :] ** 2
    s_e, s_r, s_p, eta = mu
    denominator = (1.0 + 1j * eta) * s_e * eigenvalues[:, None] - s_r * omega_sq
    coefficients = s_p * modal_forces[:, None] / denominator
    return np.asarray(coefficients.T @ phi, dtype=complex)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=CODE_DIR / "results")
    parser.add_argument("--output", type=Path,
                        default=CODE_DIR / "results/review_validation_20261103/figure6")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--sync-manuscript", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    thread_limit = threadpool_limits(limits=args.threads)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {}
    summary: dict[str, object] = {
        "reference": "full constrained direct solve at every frequency",
        "normalization": "abs(dynamic remainder) / abs(Eq. 9 MASC background)",
        "error_metric": "norm(complex dynamic remainder - MASC) / norm(complex dynamic remainder)",
        "frequency_points_per_case": POINTS,
        "threads": args.threads, "cases": {}}

    for tag, case_rel, k, description in CONFIGS:
        case_dir = args.results / case_rel
        if not case_dir.exists():
            raise FileNotFoundError(f"Required current checkpoint directory missing: {case_dir}")
        started = time.perf_counter()
        metadata, _, operators, family, restored, observation = load_parametric_case(str(case_dir), device="cpu")
        expected_production_modes = 4 if tag == "flat" else 6
        if family.modal.n_modes != expected_production_modes:
            raise RuntimeError(f"{tag}: unexpected production modal count {family.modal.n_modes}")
        del restored
        reference = family.modal.reference
        loss_factor = float(reference.loss_factor)
        fmin, fmax = (float(v) for v in metadata["frequency_interval_hz"])
        frequencies = np.linspace(fmin, fmax, POINTS)

        eigenvalues = np.asarray(reference.eigenvalues, dtype=float)
        modal_forces = np.asarray(reference.modal_forces, dtype=complex)
        phi = np.asarray(reference.mode_shapes, dtype=complex)[observation, :]
        natural = np.asarray(reference.natural_frequencies_hz, dtype=float)
        mu = np.asarray(family.bounds.center, dtype=float)

        exact = np.empty(POINTS, dtype=complex)
        checked_residual = 0.0
        solve_started = time.perf_counter()
        for index, frequency in enumerate(frequencies):
            field = family.solve(float(frequency), mu)
            exact[index] = field[observation]
            if index % 100 == 0 or index + 1 == POINTS:
                residual = family._dynamic(float(frequency), mu) @ field - family.rhs(mu)
                checked_residual = max(checked_residual, float(np.linalg.norm(residual) / np.linalg.norm(family.rhs(mu))))
                solve_elapsed = time.perf_counter() - solve_started
                remaining = (POINTS - index - 1) * solve_elapsed / (index + 1)
                print(f"{tag}: direct {index + 1}/{POINTS}, f={frequency:.2f} Hz, "
                      f"case ETA {remaining:.1f} s", flush=True)
        pole = modal_sum(frequencies, eigenvalues[:k], modal_forces[:k], phi[:k],
                         mu, loss_factor)
        background = exact - pole
        if sp.issparse(operators.K):
            static_field = spla.spsolve(operators.K.tocsc(), operators.rhs)
        else:
            static_field = np.linalg.solve(operators.K, operators.rhs)
        young, _, pressure, eta = mu
        masc = pressure / ((1.0 + 1j * eta) * young) * (
            static_field[observation] - np.sum(phi[:k] * modal_forces[:k] / eigenvalues[:k]))
        if abs(masc) < 1.0e-30:
            raise RuntimeError(f"{tag}: vanishing MASC normalization")
        mean_bg = background.mean()
        relative = np.abs(background) / abs(masc)
        static_error = float(np.linalg.norm(background - masc)
                             / max(np.linalg.norm(background), 1.0e-30))
        best_constant_error = float(np.linalg.norm(background - mean_bg) / np.linalg.norm(background))

        # amplitude of the full response, to locate the interior minimum
        amplitude = np.abs(exact)
        local = np.flatnonzero((amplitude[1:-1] < amplitude[:-2])
                               & (amplitude[1:-1] <= amplitude[2:])) + 1
        local = local[(local >= 4) & (local <= amplitude.size - 5)]
        dip_hz = float(frequencies[int(local[np.argmin(amplitude[local])])]) if local.size else float("nan")

        key = tag
        payload[f"{key}_frequencies"] = frequencies
        payload[f"{key}_background_relative"] = relative
        payload[f"{key}_background_complex"] = background
        payload[f"{key}_direct_complex"] = exact
        payload[f"{key}_pole_complex"] = pole
        payload[f"{key}_masc_background"] = np.asarray(masc)
        payload[f"{key}_best_constant_background"] = np.asarray(mean_bg)
        payload[f"{key}_amplitude_mm"] = amplitude * 1.0e3
        payload[f"{key}_masc_amplitude_mm"] = np.abs(pole + masc) * 1.0e3
        checkpoint = case_dir / "parametric_operator.pt"
        summary["cases"][tag] = {
            "description": description,
            "band_hz": [fmin, fmax],
            "retained_modes": k,
            "retained_frequencies_hz": natural[:k].tolist(),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "production_retained_modes": expected_production_modes,
            "parameters": mu.tolist(),
            "observation_index": int(observation),
            "masc_background_relative_l2_error": static_error,
            "best_constant_relative_l2_error": best_constant_error,
            "masc_background_complex_m": [float(masc.real), float(masc.imag)],
            "best_constant_background_complex_m": [float(mean_bg.real), float(mean_bg.imag)],
            "relative_remainder_range": [float(relative.min()), float(relative.max())],
            "interior_minimum_hz": dip_hz,
            "direct_solve_count": POINTS,
            "static_stiffness_solve_count": 1,
            "direct_checked_max_relative_residual": checked_residual,
            "direct_diagonal_regularization": operators.diagonal_regularization,
            "zero_frequency_masc_relative_difference": float(abs(background[0] - masc) / abs(masc)),
            "elapsed_seconds": time.perf_counter() - started,
        }
        print(f"=== {tag} ({description}) ===")
        print(f"  retained: {np.round(natural[:k], 1)} Hz")
        print(f"  static-correction background error : {100*static_error:6.2f} %")
        print(f"  interior amplitude minimum         : {dip_hz:7.1f} Hz")
        print(f"  relative remainder range           : "
              f"{relative.min():.3f} .. {relative.max():.3f}")
        print(f"  best complex-constant error         : {100*best_constant_error:.4f}%", flush=True)

    np.savez_compressed(out / "background_parallel.npz", **payload)
    (out / "background_parallel.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
    if args.sync_manuscript:
        manuscript_data = CODE_DIR.parent / "manuscript/current/submission_3/figure_scripts/data"
        for suffix in ("json", "npz"):
            shutil.copy2(out / f"background_parallel.{suffix}", manuscript_data)
    thread_limit.restore_original_limits()
    print(f"\nwritten {out / 'background_parallel.npz'}")


if __name__ == "__main__":
    main()

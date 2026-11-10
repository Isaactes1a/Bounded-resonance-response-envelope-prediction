# -*- coding: utf-8 -*-
"""Rebuild ``ellipsoidal_uq_curves.npz`` for a case whose band has changed.

That archive carries the 1000-state surrogate cloud (the first 200 curves are
saved), the eight direct high-fidelity sweeps used as the reference envelope,
and the eight unseen test curves.  ``prepare_figure_data.py`` copies it into
``figure_scripts/data/<case>_direct8_curves.npz``, which the envelope figure
reads as the "8 direct RBF-FD sweeps".

Band and point count come from the case metadata, so the archive follows the
model instead of assuming 0-200 Hz.  Sampling reproduces
``generate_ellipsoidal_uq_study.py``:

    uq   = ellipsoid.sample(n_uq,                seed + 11)
    hf   = ellipsoid.sample(n_hf_statistics,     seed + 22)
    test = ellipsoid.sample(n_test,              seed + 44)

The sixteen direct observation curves are distributed over a thread pool.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from rbf_fd_full_order import (  # noqa: E402
    EllipsoidalConvexSet,
    correlation_matrix,
    predict_parametric_observation_members,
)
from rbf_fd_full_order.visualization.result_support import load_parametric_case  # noqa: E402

DEFAULT_SEED = 701
BOUNDARY_FRACTION = 0.5


def observation_curves(result, frequencies, parameters, observation_index):
    members = predict_parametric_observation_members(
        result, frequencies, parameters, observation_index)
    return np.abs(np.mean(members, axis=0)) * 1.0e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir", type=Path,
        default=Path("results/parametric_many_query_stiffened_5000_best/stiffened"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/ellipsoidal_uq/stiffened"))
    parser.add_argument("--n-uq", type=int, default=1000)
    parser.add_argument("--n-hf-statistics", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument("--max-saved-curves", type=int, default=200)
    parser.add_argument("--sweep-points", type=int, default=601)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    metadata, _, _, family, result, observation_index = load_parametric_case(
        args.case_dir, device=args.device)
    case = str(metadata.get("case", args.case_dir.name))
    fmin, fmax = (float(value) for value in metadata["frequency_interval_hz"])
    frequencies = np.linspace(fmin, fmax, args.sweep_points)
    bounds = family.bounds
    ellipsoid = EllipsoidalConvexSet(
        center=np.asarray(bounds.center, dtype=float),
        scales=0.5 * (np.asarray(bounds.upper, dtype=float)
                      - np.asarray(bounds.lower, dtype=float)),
        correlation=correlation_matrix(bounds.dimension, e_rho=0.0),
    ).validated()

    uq_mu = ellipsoid.sample(args.n_uq, args.seed + 11,
                             boundary_fraction=BOUNDARY_FRACTION)
    hf_mu = ellipsoid.sample(args.n_hf_statistics, args.seed + 22,
                             boundary_fraction=BOUNDARY_FRACTION)
    test_mu = ellipsoid.sample(args.n_test, args.seed + 44,
                               boundary_fraction=BOUNDARY_FRACTION)
    all_direct = np.vstack((hf_mu, test_mu))

    print(f"case={case}  band={fmin:.0f}-{fmax:.0f} Hz  points={args.sweep_points}")
    print(f"surrogate cloud: {args.n_uq} curves "
          f"(saving {min(args.max_saved_curves, args.n_uq)})", flush=True)
    uq_mean_curves = observation_curves(result, frequencies, uq_mu, observation_index)

    print(f"direct sweeps: {all_direct.shape[0]} curves, {args.workers} threads",
          flush=True)

    def solve_one(parameter: np.ndarray) -> np.ndarray:
        response = family.solve_grid(frequencies, np.asarray(parameter)[None, :])
        return np.abs(response[0, :, observation_index]) * 1.0e3

    direct = np.empty((all_direct.shape[0], frequencies.size), dtype=float)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(solve_one, mu): index
                   for index, mu in enumerate(all_direct)}
        for future in as_completed(futures):
            direct[futures[future]] = future.result()
            done += 1
            print(f"  direct curve {done}/{all_direct.shape[0]}", flush=True)

    hf_curves = direct[: args.n_hf_statistics]
    test_hf_curves = direct[args.n_hf_statistics:]
    test_surrogate = observation_curves(result, frequencies, test_mu, observation_index)

    save_count = min(args.max_saved_curves, uq_mean_curves.shape[0])
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "ellipsoidal_uq_curves.npz"
    np.savez_compressed(
        target,
        frequency_hz=frequencies,
        parameters=uq_mu[:save_count],
        surrogate_curves_mm=uq_mean_curves[:save_count],
        hf_parameters=hf_mu,
        hf_curves_mm=hf_curves,
        test_parameters=test_mu,
        test_hf_curves_mm=test_hf_curves,
        test_surrogate_curves_mm=test_surrogate,
    )
    print(f"written {target}")


if __name__ == "__main__":
    main()

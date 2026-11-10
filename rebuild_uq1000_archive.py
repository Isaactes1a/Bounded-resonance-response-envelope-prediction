# -*- coding: utf-8 -*-
"""Rebuild the 1000-realisation UQ archive for a case whose band has changed.

``prepare_figure_data.py`` consumes
``unseen_accuracy_current/<case>_current_uq1000.npz``: 1000 admissible parameter
sets and their observation curves.  The curves come from the trained ensemble,
so this is cheap; only the band and point count depend on the case, and both are
read from the case metadata instead of being hard coded.

Reproduces the sampling of ``regenerate_review_figures_current.py``:

    parameters = ellipsoid.sample(1000, 712, boundary_fraction=0.5)
"""

from __future__ import annotations

import argparse
import sys
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

SAMPLE_SEED = 712
N_UQ = 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir", type=Path,
        default=Path("results/parametric_many_query_stiffened_5000_best/stiffened"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/unseen_accuracy_current"))
    parser.add_argument("--n-uq", type=int, default=N_UQ)
    parser.add_argument("--sweep-points", type=int, default=601)
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
    parameters = ellipsoid.sample(args.n_uq, SAMPLE_SEED, boundary_fraction=0.5)

    print(f"case={case}  band={fmin:.0f}-{fmax:.0f} Hz  points={args.sweep_points}")
    print(f"predicting {parameters.shape[0]} observation curves with the ensemble",
          flush=True)
    members = predict_parametric_observation_members(
        result, frequencies, parameters, observation_index)
    curves = np.abs(np.mean(members, axis=0)) * 1.0e3

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{case}_current_uq1000.npz"
    np.savez_compressed(
        target,
        frequencies=frequencies,
        parameters=parameters,
        curves=curves,
        observation_index=np.asarray(observation_index),
    )
    print(f"written {target}  curves={curves.shape}")


if __name__ == "__main__":
    main()

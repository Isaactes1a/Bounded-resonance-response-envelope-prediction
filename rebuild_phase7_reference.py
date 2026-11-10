# -*- coding: utf-8 -*-
"""Rebuild the eight full-field test references used by the figure pipeline.

``prepare_figure_data.py`` needs ``phase7/<case>_test_reference.npz``: eight
withheld parameter sets with their complete complex response over the case's
own frequency band.  This script regenerates that archive for a case whose band
has changed, reproducing the sampling used by ``run_phase7_studies.py``

    test_parameters = bounds.sample_latin_hypercube(n_test, 21 + 303)

and evaluating the retained-modal family over the case band.  The full-field
curves are distributed over a thread pool: each parameter set is independent
and the scipy sparse solve releases the GIL.
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

from rbf_fd_full_order.core.modal_resonance import ModalResonanceConfig  # noqa: E402
from rbf_fd_full_order.core.parametric_family import build_parametric_frequency_family  # noqa: E402
from rbf_fd_full_order.visualization.result_support import load_parametric_case  # noqa: E402

OBSERVATION_INDEX = 420
TEST_SEED = 21 + 303


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir", type=Path,
        default=Path("results/parametric_many_query_stiffened_5000_best/stiffened"))
    parser.add_argument("--out", type=Path, default=Path("results/phase7"))
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument("--sweep-points", type=int, default=601)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--archive-name", default=None)
    args = parser.parse_args()

    metadata, _, _, family, result, observation_index = load_parametric_case(
        args.case_dir, device="cpu")
    case = str(metadata.get("case", args.case_dir.name))
    fmin, fmax = (float(value) for value in metadata["frequency_interval_hz"])
    sweep = np.linspace(fmin, fmax, args.sweep_points)
    bounds = family.bounds
    test_mu = bounds.sample_latin_hypercube(int(args.n_test), TEST_SEED)

    print(f"case={case}  band={fmin:.0f}-{fmax:.0f} Hz  points={args.sweep_points}")
    print(f"building full-field references for {test_mu.shape[0]} parameter sets, "
          f"{args.workers} threads", flush=True)

    n_dofs = int(metadata.get("n_dofs", 0)) or int(family.modal.reference.n_dofs)
    reference = np.empty((test_mu.shape[0], sweep.size, n_dofs), dtype=complex)

    def solve_one(index: int) -> tuple[int, np.ndarray]:
        response = family.solve_grid(sweep, test_mu[index][None, :])
        return index, np.asarray(response[0], dtype=complex)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(solve_one, index): index
                   for index in range(test_mu.shape[0])}
        for future in as_completed(futures):
            index, curve = future.result()
            reference[index] = curve
            done += 1
            print(f"  curve {done}/{test_mu.shape[0]} done", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    name = args.archive_name or f"{case}_test_reference.npz"
    np.savez_compressed(
        args.out / name,
        frequencies_hz=sweep,
        test_parameters=test_mu,
        test_reference_full=reference,
        observation_index=np.asarray(OBSERVATION_INDEX),
    )
    print(f"written {args.out / name}  shape={reference.shape}")


if __name__ == "__main__":
    main()

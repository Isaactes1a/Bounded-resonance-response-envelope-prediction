# -*- coding: utf-8 -*-
"""Update Fig. 3 metrics on a retained 40-parameter plate test set."""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
from pathlib import Path

import numpy as np

from rbf_fd_full_order import predict_parametric_members
from rbf_fd_full_order.visualization.result_support import load_parametric_case


_FAMILY = None
_FREQUENCIES = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path("results/parametric_many_query_stiffened_5000_best/stiffened"),
    )
    parser.add_argument(
        "--calibration-archive",
        type=Path,
        default=Path("results/ellipsoidal_uq/stiffened/ellipsoidal_calibration_40x40.npz"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/ellipsoidal_uq/stiffened/ellipsoidal_unseen_test_metrics.csv"),
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def _init_worker(case_dir: str, frequencies: np.ndarray) -> None:
    global _FAMILY, _FREQUENCIES
    _, _, _, _FAMILY, _, _ = load_parametric_case(Path(case_dir), device="cpu")
    _FREQUENCIES = np.asarray(frequencies, dtype=float)


def _solve(parameter: np.ndarray) -> np.ndarray:
    return _FAMILY.solve_grid(_FREQUENCIES, np.asarray(parameter, dtype=float)[None, :])[0]


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - reference) / max(float(np.linalg.norm(reference)), 1.0e-30))


def main() -> None:
    args = parse_args()
    archive = np.load(args.calibration_archive)
    frequencies = np.asarray(archive["frequencies_hz"], dtype=float)
    parameters = np.asarray(archive["test_parameters"], dtype=float)
    stored_hf_obs = np.asarray(archive["test_hf_mm"], dtype=float)
    _, _, _, family, result, observation_index = load_parametric_case(args.case_dir, device="cpu")

    context = mp.get_context("spawn")
    rows = []
    with context.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(str(args.case_dir), frequencies),
    ) as pool:
        for index, direct in enumerate(pool.imap(_solve, parameters, chunksize=1)):
            members = predict_parametric_members(result, frequencies, parameters[index : index + 1])[:, 0]
            predicted = np.mean(members, axis=0)
            predicted_obs = np.mean(np.abs(members[..., observation_index]), axis=0) * 1.0e3
            direct_obs = np.abs(direct[:, observation_index]) * 1.0e3
            if not np.allclose(direct_obs, stored_hf_obs[index], rtol=1.0e-9, atol=1.0e-10):
                raise RuntimeError(f"Direct observation mismatch for test parameter {index}.")
            peak_hf = int(np.argmax(direct_obs))
            peak_pred = int(np.argmax(predicted_obs))
            row = {
                "parameter_index": index,
                **{name: float(value) for name, value in zip(family.bounds.names, parameters[index])},
                "observation_curve_relative_l2": _relative_l2(predicted_obs, direct_obs),
                "full_response_relative_l2": _relative_l2(predicted, direct),
                "peak_frequency_error_hz": abs(float(frequencies[peak_pred] - frequencies[peak_hf])),
                "peak_amplitude_relative_error": abs(float(np.max(predicted_obs) - np.max(direct_obs)))
                / max(float(np.max(direct_obs)), 1.0e-30),
            }
            rows.append(row)
            print(f"completed {index + 1}/{parameters.shape[0]}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    mp.freeze_support()
    main()

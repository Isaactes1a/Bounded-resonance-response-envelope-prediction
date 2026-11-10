# -*- coding: utf-8 -*-
"""Summarize phase-7 checkpoints (multi-seed and low-label) after training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rbf_fd_full_order import (
    ModalResonanceConfig,
    ParameterBounds,
    build_parametric_frequency_family,
    load_parametric_operator,
    predict_parametric_members,
    prepare_constrained_operators,
)
from rbf_fd_full_order.examples.cases import MeshConfig, build_case_system


def _bounds() -> ParameterBounds:
    return ParameterBounds(
        lower=np.asarray([0.85, 0.90, 0.80, 0.005], dtype=float),
        upper=np.asarray([1.15, 1.10, 1.20, 0.020], dtype=float),
    ).validated()


def _parabolic_peak(frequencies: np.ndarray, curve: np.ndarray) -> float:
    index = int(np.argmax(curve))
    if index <= 0 or index >= curve.size - 1:
        return float(frequencies[index])
    y = np.log10(np.maximum(curve[index - 1 : index + 2], 1.0e-30))
    denominator = y[0] - 2.0 * y[1] + y[2]
    if abs(denominator) < 1.0e-14:
        return float(frequencies[index])
    step = float(np.median(np.diff(frequencies)))
    return float(frequencies[index] + 0.5 * (y[0] - y[2]) / denominator * step)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results") / "phase7")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    bounds = _bounds()
    mesh = MeshConfig()
    rows = []
    for pattern, case in (("multiseed", "flat"), ("lowlabel", "stiffened")):
        system = build_case_system(case, mesh)
        operators = prepare_constrained_operators(system)
        family = build_parametric_frequency_family(
            operators, bounds=bounds,
            modal_config=ModalResonanceConfig(
                enabled=True, n_modes=8, relative_half_width=0.04,
                minimum_half_width_hz=0.75, transition_fraction=0.2,
                exclusion_threshold=0.5, exclude_labeled=True, exclude_physics=True,
            ),
            frequency_min_hz=0.0, frequency_max_hz=200.0,
        )
        archive = np.load(args.root / f"{case}_test_reference.npz")
        frequencies = np.asarray(archive["frequencies_hz"], dtype=float)
        parameters = np.asarray(archive["test_parameters"], dtype=float)
        reference = np.asarray(archive["test_reference_full"], dtype=complex)
        ref_obs = np.abs(reference[..., 420]) * 1.0e3
        for checkpoint in sorted((args.root / pattern).glob("*.pt")):
            result = load_parametric_operator(checkpoint, family, device=args.device)
            predicted = np.mean(predict_parametric_members(result, frequencies, parameters), axis=0)
            squared = np.sum(np.abs(predicted - reference) ** 2, axis=-1)
            reference_norm = np.sum(np.abs(reference) ** 2, axis=-1)
            mse = float(np.mean(squared / np.maximum(reference_norm, 1.0e-30)))
            pred_obs = np.abs(predicted[..., 420]) * 1.0e3
            obs_errors = np.asarray(
                [
                    np.linalg.norm(pred_obs[i] - ref_obs[i]) / max(np.linalg.norm(ref_obs[i]), 1.0e-30)
                    for i in range(parameters.shape[0])
                ]
            )
            peak_errors = []
            for i in range(parameters.shape[0]):
                peak_errors.append(
                    abs(_parabolic_peak(frequencies, pred_obs[i]) - _parabolic_peak(frequencies, ref_obs[i]))
                )
            rows.append(
                {
                    "group": pattern,
                    "checkpoint": checkpoint.stem,
                    "case": case,
                    "full_domain_relative_mse": mse,
                    "observation_median_relative_l2_percent": 100.0 * float(np.median(obs_errors)),
                    "observation_max_relative_l2_percent": 100.0 * float(np.max(obs_errors)),
                    "peak_frequency_interp_mean_hz": float(np.mean(peak_errors)),
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    (args.root / "phase7_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

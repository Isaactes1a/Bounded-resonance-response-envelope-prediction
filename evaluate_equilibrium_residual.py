# -*- coding: utf-8 -*-
"""Report the relative discrete dynamic-equilibrium residual on test curves.

For a set of trained checkpoints, the residual

    r(mu, f) = A(mu, f) u_pred(mu, f) - f(mu)

is evaluated with the same preassembled RBF-FD operators used by the direct
solver and the physics loss.  The metric is the per-evaluation relative
residual ``||r||_2 / ||f||_2`` aggregated as mean, p95 and max over all
realizations and frequency points, plus the mean per complete curve.

Supported checkpoint groups:
    production : results/parametric_many_query_{case}_5000_best/{case}
                 (40 ellipsoidal test curves on the 1 Hz grid of the case band)
    ablation   : results/parametric_ablation_stiffened_0_600/stiffened/{variant}.pt
                 for the stiffened panel and
                 results/parametric_ablation_compact/flat/{variant}.pt for the
                 unstiffened plate (8 LHS test curves on the case band)
    phase7     : results/phase7/{multiseed,lowlabel}/*.pt
                 (8 LHS test curves on the case band)

The script limits itself to one worker thread and batches evaluations so that
memory stays far below the reported 20% limit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from rbf_fd_full_order import (
    ParameterBounds,
    build_parametric_frequency_family,
    predict_parametric_members,
    prepare_constrained_operators,
)
from rbf_fd_full_order.examples.cases import MeshConfig, build_case_system
from rbf_fd_full_order.visualization.result_support import load_parametric_case


CASES = ("flat", "stiffened")
OBSERVATION_INDEX = 420
# Deployed checkpoints whose metadata carries the analysis band of each case.
CASE_METADATA = {
    "flat": Path("results") / "parametric_many_query_flat_5000_best" / "flat" / "metadata.json",
    "stiffened": Path("results") / "parametric_many_query_stiffened_5000_best" / "stiffened" / "metadata.json",
}
# Ablation checkpoints rebuilt on the current band, per case.
ABLATION_ROOTS = {
    "flat": Path("results") / "parametric_ablation_compact" / "flat",
    "stiffened": Path("results") / "parametric_ablation_stiffened_0_600" / "stiffened",
}


def _case_band(case: str) -> float:
    """Upper end of the analysis band, read from the deployed metadata."""
    path = CASE_METADATA.get(case)
    if path is None or not path.is_file():
        return 200.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    interval = payload.get("frequency_interval_hz")
    if not interval:
        return 200.0
    return float(interval[1])


def _band_grid(case: str) -> np.ndarray:
    fmax = _case_band(case)
    return np.linspace(0.0, fmax, int(round(fmax)) + 1)


def _bounds() -> ParameterBounds:
    return ParameterBounds(
        lower=np.asarray([0.85, 0.90, 0.80, 0.005], dtype=float),
        upper=np.asarray([1.15, 1.10, 1.20, 0.020], dtype=float),
    ).validated()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("production", "ablation", "phase7"), default="production")
    parser.add_argument("--out", type=Path, default=Path("results") / "equilibrium_residual")
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-frequencies", type=int, default=64)
    return parser.parse_args()


def _residual_metrics(
    result,
    frequencies: np.ndarray,
    parameters: np.ndarray,
    device: str,
    batch_frequencies: int,
) -> dict[str, object]:
    nominal = result.family.nominal
    k0 = sp.csr_matrix(nominal.K)
    m0 = sp.csr_matrix(nominal.M)
    f0 = np.asarray(nominal.rhs, dtype=complex).reshape(-1)
    values = []
    for start in range(0, frequencies.size, batch_frequencies):
        stop = min(start + batch_frequencies, frequencies.size)
        freq_batch = frequencies[start:stop]
        predicted = np.mean(predict_parametric_members(result, freq_batch, parameters), axis=0)
        # predicted: (n_mu, n_f, n_dof)
        for mu_index, mu in enumerate(parameters):
            s_e, s_rho, s_p, eta = mu
            omega_sq = (2.0 * np.pi * freq_batch) ** 2
            for freq_index in range(freq_batch.size):
                a_matrix = (1.0 + 1.0j * eta) * s_e * k0 - omega_sq[freq_index] * s_rho * m0
                load = s_p * f0
                residual = a_matrix @ predicted[mu_index, freq_index, :] - load
                values.append(
                    float(np.linalg.norm(residual) / max(float(np.linalg.norm(load)), 1.0e-30))
                )
    values = np.asarray(values, dtype=float)
    n_f = int(frequencies.size)
    curve_means = [
        float(np.mean(values[mu_index * n_f : (mu_index + 1) * n_f]))
        for mu_index in range(parameters.shape[0])
    ]
    return {
        "relative_residual_mean": float(np.mean(values)),
        "relative_residual_p95": float(np.quantile(values, 0.95)),
        "relative_residual_max": float(np.max(values)),
        "curve_mean_min": float(np.min(curve_means)),
        "curve_mean_max": float(np.max(curve_means)),
        "evaluation_count": int(values.size),
        "curve_count": int(parameters.shape[0]),
        "frequency_count": int(n_f),
    }


def _test_data(group: str, case: str, results_root: Path):
    if group == "production":
        archive = np.load(results_root / "ellipsoidal_uq" / case / "ellipsoidal_calibration_40x40.npz")
        frequencies = np.asarray(archive["frequencies_hz"], dtype=float)
        parameters = np.asarray(archive["test_parameters"], dtype=float)
    elif group == "phase7":
        archive = results_root / "phase7" / f"{case}_test_reference.npz"
        if archive.is_file():
            with np.load(archive) as data:
                frequencies = np.asarray(data["frequencies_hz"], dtype=float)
                parameters = np.asarray(data["test_parameters"], dtype=float)
        else:
            frequencies = _band_grid(case)
            parameters = _bounds().sample_latin_hypercube(8, 21 + 303)
    else:
        frequencies = _band_grid(case)
        parameters = _bounds().sample_latin_hypercube(8, 21 + 303)
    return frequencies, parameters


def run_case(group: str, case: str, args: argparse.Namespace) -> list[dict[str, object]]:
    results_root = Path("results")
    frequencies, parameters = _test_data(group, case, results_root)
    if group == "production":
        case_dir = results_root / f"parametric_many_query_{case}_5000_best" / case
        _, _, _, _, result, _ = load_parametric_case(case_dir, device=args.device)
        checkpoints = [(f"production_{case}", result)]
    elif group == "ablation":
        bounds = _bounds()
        mesh = MeshConfig()
        system = build_case_system(case, mesh)
        operators = prepare_constrained_operators(system)
        checkpoints = []
        from rbf_fd_full_order import ModalResonanceConfig, load_parametric_operator

        for variant in ("direct_data", "direct_physics", "pole_data", "full"):
            checkpoint = ABLATION_ROOTS.get(case, results_root / "parametric_ablation_compact" / case) / f"{variant}.pt"
            if not checkpoint.is_file():
                checkpoint = results_root / "parametric_ablation_compact" / case / f"{variant}.pt"
            if checkpoint.exists():
                modal_enabled = variant in ("pole_data", "full")
                family = build_parametric_frequency_family(
                    operators, bounds=bounds,
                    modal_config=(
                        ModalResonanceConfig(
                            enabled=True, n_modes=8, relative_half_width=0.04,
                            minimum_half_width_hz=0.75, transition_fraction=0.2,
                            exclusion_threshold=0.5, exclude_labeled=True, exclude_physics=True,
                        )
                        if modal_enabled
                        else None
                    ),
                    frequency_min_hz=0.0, frequency_max_hz=_case_band(case),
                )
                result = load_parametric_operator(checkpoint, family, device=args.device)
                checkpoints.append((f"{variant}", result))
    else:  # phase7
        bounds = _bounds()
        mesh = MeshConfig()
        system = build_case_system(case, mesh)
        operators = prepare_constrained_operators(system)
        checkpoints = []
        from rbf_fd_full_order import ModalResonanceConfig, load_parametric_operator

        archive = results_root / "phase7" / f"{case}_test_reference.npz"
        if archive.exists():
            data = np.load(archive)
            frequencies = np.asarray(data["frequencies_hz"], dtype=float)
            parameters = np.asarray(data["test_parameters"], dtype=float)
        patterns = ("multiseed",) if case == "flat" else ("lowlabel",)
        for pattern in patterns:
            for checkpoint in sorted((results_root / "phase7" / pattern).glob("*.pt")):
                if f"n" not in checkpoint.stem:
                    continue
                modal_enabled = ("pole_data" in checkpoint.stem) or ("full" in checkpoint.stem)
                family = build_parametric_frequency_family(
                    operators, bounds=bounds,
                    modal_config=(
                        ModalResonanceConfig(
                            enabled=True, n_modes=8, relative_half_width=0.04,
                            minimum_half_width_hz=0.75, transition_fraction=0.2,
                            exclusion_threshold=0.5, exclude_labeled=True, exclude_physics=True,
                        )
                        if modal_enabled
                        else None
                    ),
                    frequency_min_hz=0.0, frequency_max_hz=_case_band(case),
                )
                result = load_parametric_operator(checkpoint, family, device=args.device)
                checkpoints.append((f"{pattern}/{checkpoint.stem}", result))

    rows = []
    for label, result in checkpoints:
        metrics = _residual_metrics(
            result,
            frequencies,
            parameters,
            args.device,
            int(args.batch_frequencies),
        )
        rows.append({"case": case, "checkpoint": label, **metrics})
        print(json.dumps({"case": case, "checkpoint": label, **metrics}, ensure_ascii=False), flush=True)
    return rows


if __name__ == "__main__":
    args = _parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    cases = CASES if args.case == "all" else (args.case,)
    all_rows = []
    for case in cases:
        try:
            all_rows.extend(run_case(args.group, case, args))
        except Exception as exc:  # keep going
            print(f"[{case}] FAILED: {exc}", flush=True)
    if all_rows:
        (args.out / f"{args.group}_residual.json").write_text(
            json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )

# -*- coding: utf-8 -*-
"""Phase-7 studies: multi-seed ablation and low-label trend curves.

Resource policy for this script:
    - torch and BLAS are limited to one worker thread;
    - no dataset copies larger than a single full-field test reference;
    - physics and data tensors are built per member and released after use.

The multi-seed study trains ``pole_data`` and ``full`` variants on the flat
case for several training seeds at a reduced budget (400 epochs, one member),
so that the seed-to-seed variability of the physics-loss increment on the
complete model can be assessed without reproducing the 1500-epoch budget.

The low-label study trains the same two variants on the stiffened case with
8, 12 and 16 training realizations.  Both studies save the shared full-field
test reference and test parameters for later figure generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from rbf_fd_full_order import (
    ModalResonanceConfig,
    ParameterBounds,
    ParametricOperatorConfig,
    build_parametric_frequency_family,
    predict_parametric_members,
    prepare_constrained_operators,
    save_parametric_operator,
)
from rbf_fd_full_order.examples.cases import MeshConfig, build_case_system
from rbf_fd_full_order.training.parametric_operator import (
    ParametricOperatorResult,
    _frequency_bounds,
    _train_one_member,
    build_parametric_training_dataset,
)


VARIANTS = ("pole_data", "full")
OBSERVATION_INDEX = 420
# Deployed checkpoints whose metadata carries the analysis band of each case.
CASE_METADATA = {
    "flat": Path("results") / "parametric_many_query_flat_5000_best" / "flat" / "metadata.json",
    "stiffened": Path("results") / "parametric_many_query_stiffened_5000_best" / "stiffened" / "metadata.json",
}


def case_band(case: str, override: float | None = None) -> float:
    """Upper end of the analysis band, read from the deployed metadata."""
    if override is not None:
        return float(override)
    path = CASE_METADATA.get(case)
    if path is None or not path.is_file():
        return 200.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    interval = payload.get("frequency_interval_hz")
    if not interval:
        return 200.0
    return float(interval[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results") / "phase7")
    parser.add_argument("--only", choices=("all", "multiseed", "lowlabel"), default="all")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--n-test-parameters", type=int, default=8)
    parser.add_argument("--n-labeled", type=int, default=24)
    parser.add_argument("--n-physics", type=int, default=102)
    parser.add_argument("--sweep-points", type=int, default=0,
                        help="evaluation grid size; 0 means a 1 Hz grid of the case band")
    parser.add_argument("--frequency-max-hz", type=float, default=None,
                        help="override the analysis band read from the case metadata")
    parser.add_argument("--reuse-reference", action="store_true",
                        help="load an existing test-reference archive on the same grid")
    parser.add_argument("--physics-weight", type=float, default=0.02)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()



def _bounds() -> ParameterBounds:
    return ParameterBounds(
        lower=np.asarray([0.85, 0.90, 0.80, 0.005], dtype=float),
        upper=np.asarray([1.15, 1.10, 1.20, 0.020], dtype=float),
    ).validated()


def _modal_config() -> ModalResonanceConfig:
    return ModalResonanceConfig(
        enabled=True,
        n_modes=8,
        relative_half_width=0.04,
        minimum_half_width_hz=0.75,
        transition_fraction=0.2,
        exclusion_threshold=0.5,
        exclude_labeled=True,
        exclude_physics=True,
        minimum_participation_ratio=1.0e-6,
    )


def _operator_config(args: argparse.Namespace, physics_weight: float) -> ParametricOperatorConfig:
    return ParametricOperatorConfig(
        representation="pod",
        epochs=int(args.epochs),
        learning_rate=2.0e-4,
        width=96,
        depth=3,
        fourier_modes=4,
        pod_modes=8,
        ensemble_size=int(args.ensemble_size),
        physics_weight=float(physics_weight),
        warmup_epochs=max(int(args.epochs) // 4, 1),
        log_every=max(int(args.epochs) // 20, 1),
        device=str(args.device),
        modal_resonance=_modal_config(),
    )


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - reference) / max(float(np.linalg.norm(reference)), 1.0e-30))


def _parabolic_peak(frequencies: np.ndarray, curve: np.ndarray) -> float:
    index = int(np.argmax(curve))
    if index <= 0 or index >= curve.size - 1:
        return float(frequencies[index])
    y = np.log10(np.maximum(curve[index - 1 : index + 2], 1.0e-30))
    denominator = y[0] - 2.0 * y[1] + y[2]
    if abs(denominator) < 1.0e-14:
        return float(frequencies[index])
    step = float(np.median(np.diff(frequencies)))
    offset = 0.5 * (y[0] - y[2]) / denominator
    return float(frequencies[index] + offset * step)


def _metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    frequencies: np.ndarray,
) -> dict[str, float]:
    epsilon = 1.0e-30
    squared_relative = (
        np.sum(np.abs(predicted - reference) ** 2, axis=-1)
        / (np.sum(np.abs(reference) ** 2, axis=-1) + epsilon)
    )
    curve_errors = [
        _relative_l2(predicted[index], reference[index]) for index in range(predicted.shape[0])
    ]
    ref_obs = np.abs(reference[..., OBSERVATION_INDEX]) * 1.0e3
    pred_obs = np.abs(predicted[..., OBSERVATION_INDEX]) * 1.0e3
    grid_errors = []
    interp_errors = []
    amplitude_errors = []
    for index in range(predicted.shape[0]):
        ref_grid = float(frequencies[np.argmax(ref_obs[index])])
        pred_grid = float(frequencies[np.argmax(pred_obs[index])])
        grid_errors.append(abs(pred_grid - ref_grid))
        ref_peak = _parabolic_peak(frequencies, ref_obs[index])
        pred_peak = _parabolic_peak(frequencies, pred_obs[index])
        interp_errors.append(abs(pred_peak - ref_peak))
        ref_amplitude = float(np.max(ref_obs[index]))
        pred_amplitude = float(np.max(pred_obs[index]))
        amplitude_errors.append(abs(pred_amplitude - ref_amplitude) / max(ref_amplitude, 1.0e-30))
    return {
        "full_domain_relative_mse": float(np.mean(squared_relative)),
        "curve_relative_l2_mean": float(np.mean(curve_errors)),
        "curve_relative_l2_max": float(np.max(curve_errors)),
        "peak_frequency_error_grid_max_hz": float(np.max(grid_errors)),
        "peak_frequency_error_interp_mean_hz": float(np.mean(interp_errors)),
        "peak_frequency_error_interp_max_hz": float(np.max(interp_errors)),
        "peak_amplitude_relative_error_mean": float(np.mean(amplitude_errors)),
        "peak_amplitude_relative_error_max": float(np.max(amplitude_errors)),
    }


def _shared_data(case: str, args: argparse.Namespace, n_train: int, seed: int,
                 reference_archive: Path | None = None):
    bounds = _bounds()
    mesh = MeshConfig()
    system = build_case_system(case, mesh)
    operators = prepare_constrained_operators(system)
    fmax = case_band(case, args.frequency_max_hz)
    sweep_points = int(args.sweep_points) if int(args.sweep_points) > 0 else int(round(fmax)) + 1
    print(f"[{case}] band 0-{fmax:.0f} Hz, {sweep_points} evaluation points", flush=True)
    family_direct = build_parametric_frequency_family(
        operators, bounds=bounds, modal_config=None, frequency_min_hz=0.0, frequency_max_hz=fmax
    )
    family_pole = build_parametric_frequency_family(
        operators, bounds=bounds, modal_config=_modal_config(), frequency_min_hz=0.0, frequency_max_hz=fmax
    )
    train_mu = bounds.sample_latin_hypercube(int(n_train), int(seed) + 101)
    test_mu = bounds.sample_latin_hypercube(int(args.n_test_parameters), 21 + 303)
    labeled = np.linspace(0.0, fmax, int(args.n_labeled))
    physics = np.unique(np.r_[labeled, np.linspace(0.0, fmax, int(args.n_physics))])
    sweep = np.linspace(0.0, fmax, sweep_points)
    reference = None
    if args.reuse_reference and reference_archive is not None and reference_archive.is_file():
        with np.load(reference_archive) as archive:
            cached = np.asarray(archive["frequencies_hz"], dtype=float)
            if cached.size == sweep.size and np.allclose(cached, sweep):
                reference = np.asarray(archive["test_reference_full"], dtype=complex)
                test_mu = np.asarray(archive["test_parameters"], dtype=float)
                print(f"[{case}] reused test reference {reference_archive.name} "
                      f"shape={reference.shape}", flush=True)
    if reference is None:
        print(f"[{case}] generating shared test reference ({test_mu.shape[0]} curves)", flush=True)
        reference = np.asarray(family_pole.solve_grid(sweep, test_mu), dtype=complex)
    return {
        "bounds": bounds,
        "operators": operators,
        "family_direct": family_direct,
        "family_pole": family_pole,
        "train_mu": train_mu,
        "test_mu": test_mu,
        "labeled": labeled,
        "physics": physics,
        "sweep": sweep,
        "reference": reference,
    }


def train_variants(
    case: str,
    out_dir: Path,
    args: argparse.Namespace,
    n_train: int,
    seed: int,
    shared,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_pole = build_parametric_training_dataset(
        shared["family_pole"], shared["train_mu"], shared["labeled"], shared["physics"],
        representation="pod", pod_modes=8,
    )
    frequency_min, frequency_range = _frequency_bounds(shared["labeled"], shared["physics"])
    rows = []
    for variant in VARIANTS:
        physics_weight = float(args.physics_weight if variant == "full" else 0.0)
        family = shared["family_pole"]
        dataset = dataset_pole
        config = _operator_config(args, physics_weight=physics_weight)
        print(
            f"[{case}] seed={seed} n_train={n_train} training {variant} "
            f"({args.epochs} epochs x {args.ensemble_size} member)",
            flush=True,
        )
        started = time.perf_counter()
        members = [
            _train_one_member(
                family, dataset, config, frequency_min, frequency_range,
                seed=int(seed) + 1009 * index,
            )
            for index in range(int(args.ensemble_size))
        ]
        result = ParametricOperatorResult(
            family=family,
            members=members,
            config=config,
            bounds=shared["bounds"],
            frequency_min=frequency_min,
            frequency_range=frequency_range,
            n_dofs=family.n_free,
            response_scaler=dataset.response_scaler,
            coefficient_scale=dataset.coefficient_scale,
            pod_basis=dataset.pod_basis,
            training_parameters=np.asarray(shared["train_mu"], dtype=float),
            labeled_frequencies_hz=shared["labeled"],
            physics_frequencies_hz=shared["physics"],
            high_fidelity_generation_s=dataset.high_fidelity_generation_s,
            high_fidelity_solve_count=dataset.high_fidelity_solve_count,
            ensemble_training_s=float(sum(member.training_s for member in members)),
        )
        checkpoint = out_dir / f"n{n_train}_s{seed}_{variant}.pt"
        save_parametric_operator(result, checkpoint)
        training_s = time.perf_counter() - started
        predicted = np.mean(predict_parametric_members(result, shared["sweep"], shared["test_mu"]), axis=0)
        metrics = _metrics(predicted, shared["reference"], shared["sweep"])
        row = {
            "case": case,
            "variant": variant,
            "seed": int(seed),
            "n_train": int(n_train),
            "epochs": int(args.epochs),
            "ensemble_size": int(args.ensemble_size),
            "training_s": round(float(training_s), 2),
            "best_epoch_per_member": [int(member.best_epoch) for member in members],
            "best_total_loss_per_member": [float(member.best_total_loss) for member in members],
            **metrics,
        }
        rows.append(row)
        print(
            f"[{case}] seed={seed} n_train={n_train} {variant}: MSE={metrics['full_domain_relative_mse']:.4e} "
            f"peak-freq={metrics['peak_frequency_error_interp_mean_hz']:.4f} Hz "
            f"peak-amp={100 * metrics['peak_amplitude_relative_error_mean']:.3f}% "
            f"[{training_s:.0f} s]",
            flush=True,
        )
    return {"rows": rows, "result_dir": out_dir}


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)

    # Multi-seed study: flat, 20 training realizations, seeds 21/121/221.
    if args.only in ("all", "multiseed"):
        shared_flat = _shared_data("flat", args, 20, 21,
                                   reference_archive=args.out / "flat_test_reference.npz")
        flat_archive = args.out / "flat_test_reference.npz"
        np.savez_compressed(
            flat_archive,
            frequencies_hz=shared_flat["sweep"],
            test_parameters=shared_flat["test_mu"],
            test_reference_full=shared_flat["reference"],
            observation_index=np.asarray(OBSERVATION_INDEX),
        )
        multi_rows = []
        shared_template = shared_flat
        for seed in (21, 121, 221):
            shared = dict(shared_template)
            shared["train_mu"] = shared_template["bounds"].sample_latin_hypercube(20, int(seed) + 101)
            summary = train_variants("flat", args.out / "multiseed", args, 20, seed, shared)
            multi_rows.extend(summary["rows"])
        with (args.out / "multiseed_summary.json").open("w", encoding="utf-8") as handle:
            json.dump({"case": "flat", "rows": multi_rows}, handle, indent=2, ensure_ascii=False)
        print("multi-seed study written", flush=True)

    # Low-label study: stiffened, 8/12/16 realizations, fixed seed.
    if args.only in ("all", "lowlabel"):
        low_rows = []
        shared_stiff = _shared_data("stiffened", args, 8, 21,
                                    reference_archive=args.out / "stiffened_test_reference.npz")
        stiff_archive = args.out / "stiffened_test_reference.npz"
        np.savez_compressed(
            stiff_archive,
            frequencies_hz=shared_stiff["sweep"],
            test_parameters=shared_stiff["test_mu"],
            test_reference_full=shared_stiff["reference"],
            observation_index=np.asarray(OBSERVATION_INDEX),
        )
        for n_train in (8, 12, 16):
            shared = dict(shared_stiff)
            shared["train_mu"] = shared_stiff["bounds"].sample_latin_hypercube(n_train, 21 + 101)
            summary = train_variants("stiffened", args.out / "lowlabel", args, n_train, 21, shared)
            low_rows.extend(summary["rows"])
        with (args.out / "lowlabel_summary.json").open("w", encoding="utf-8") as handle:
            json.dump({"case": "stiffened", "rows": low_rows}, handle, indent=2, ensure_ascii=False)
        print("low-label study written", flush=True)


if __name__ == "__main__":
    main()

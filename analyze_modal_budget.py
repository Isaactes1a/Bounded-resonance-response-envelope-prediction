# -*- coding: utf-8 -*-
"""How the learned dynamic remainder pays off as the modal budget grows.

Each configuration is analysed over its own frequency band. Band, excitation,
retained set and observation coordinate are properties of the model, so the
cases are compared at their own operating points instead of being forced into a
shared window.

For a retained set of ``k`` modes the response splits as

    u(f) = u_pole(f; k) + u_bg(f; k)

with ``u_pole`` the analytic rational branch of the retained eigenpairs and
``u_bg`` the dynamic remainder. The mode-acceleration static correction (MASC)
replaces ``u_bg`` by a frequency-independent term, so its error is exactly the
frequency dependence of ``u_bg``:

    e_static(k) = || u_bg - mean(u_bg) || / || u_bg ||

The exact response is a 40-mode modal sum, which reproduces the direct
constrained solve to about 4e-5 of the amplitude over the band.

Cases are rebuilt from their training arguments and the saved checkpoint, so a
run whose post-processing did not finish can still be analysed.

Outputs:
    results/modal_budget/modal_budget_metrics.csv
    results/modal_budget/modal_budget_summary.json
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from rbf_fd_full_order.core.modal_resonance import build_modal_resonance_interface  # noqa: E402
from rbf_fd_full_order.core.parametric_family import build_parametric_frequency_family  # noqa: E402
from rbf_fd_full_order.visualization.result_support import load_parametric_case  # noqa: E402

MODE_BUDGETS = (1, 2, 4, 6, 8)
TOTAL_MODES = 40
FREQUENCY_POINTS = 801
OBSERVATION_INDEX = 420

# label, model key, case directory, frequency_max, modal_modes, sweep_points, rebuild?
CONFIGS = (
    ("flat", "flat", "parametric_many_query_flat_5000_best\\flat", 200.0, 4, 401, False),
    ("stiffened", "stiffened", "parametric_many_query_stiffened_5000_best\\stiffened",
     200.0, 1, 401, False),
    ("stiffened_600", "stiffened", "parametric_many_query_stiffened_6mode_0_600\\stiffened",
     600.0, 6, 601, True),
)


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "mpq_driver", str(CODE_DIR / "run_parametric_many_query.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["mpq_driver"] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    return module


def build_family(driver, case: str, frequency_max: float, modal_modes: int,
                 sweep_points: int):
    argv = sys.argv
    sys.argv = ["x", "--case", case, "--frequency-min", "0",
                "--frequency-max", str(frequency_max),
                "--modal-modes", str(modal_modes),
                "--sweep-points", str(sweep_points), "--out", "results/_scratch"]
    try:
        args = driver.parse_args()
    finally:
        sys.argv = argv
    system = driver.build_case_system(case, driver._mesh(args))
    operators = driver.prepare_constrained_operators(system)
    family = build_parametric_frequency_family(
        operators, bounds=driver._bounds(args), modal_config=driver._modal_config(args),
        frequency_min_hz=0.0, frequency_max_hz=frequency_max)
    return system, operators, family


def modal_sum(frequencies, eigenvalues, modal_forces, phi, mu, loss_factor):
    omega_sq = (2.0 * np.pi * frequencies)[None, :] ** 2
    s_e, s_r, s_p, eta = mu
    denominator = (1.0 + 1j * eta) * s_e * eigenvalues[:, None] - s_r * omega_sq
    coefficients = s_p * modal_forces[:, None] / denominator
    return np.asarray(coefficients.T @ phi, dtype=complex)


def interior_dip(amplitude, frequencies, lo, hi):
    window = (frequencies >= lo) & (frequencies <= hi)
    curve = np.asarray(amplitude)[window]
    band = frequencies[window]
    if curve.size < 9:
        return False, float("nan")
    local = np.flatnonzero((curve[1:-1] < curve[:-2]) & (curve[1:-1] <= curve[2:])) + 1
    local = local[(local >= 4) & (local <= curve.size - 5)]
    if local.size == 0:
        return False, float("nan")
    idx = int(local[np.argmin(curve[local])])
    return True, float(band[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=CODE_DIR / "results")
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()

    out = args.results / "modal_budget"
    out.mkdir(parents=True, exist_ok=True)
    data_dir = (CODE_DIR.parent / "manuscript" / "current" / "submission_3"
                / "figure_scripts" / "data")
    parameters = np.load(data_dir / "flat_direct8_curves.npz",
                         allow_pickle=True)["test_parameters"]
    driver = load_driver()
    selected = set(args.cases) if args.cases else None

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"mode_budgets": list(MODE_BUDGETS),
                                 "curves": int(parameters.shape[0]),
                                 "observation_index": OBSERVATION_INDEX,
                                 "cases": {}}

    for tag, model_key, case_rel, fmax, modal_modes, sweep, from_ckpt in CONFIGS:
        if selected is not None and tag not in selected:
            continue
        case_dir = args.results / case_rel
        if not case_dir.exists():
            print(f"=== {tag}: {case_dir} not available, skipped ===")
            continue

        if from_ckpt:
            system, operators, family = build_family(driver, model_key,
                                                     fmax, modal_modes, sweep)
        else:
            _, _, operators, family, _, _ = load_parametric_case(str(case_dir), device="cpu")
        reference = family.modal.reference
        loss_factor = float(reference.loss_factor)

        frequencies = np.linspace(0.0, fmax, FREQUENCY_POINTS)
        wide = dataclasses.replace(reference.config, n_modes=TOTAL_MODES,
                                   frequency_min_hz=0.0,
                                   frequency_max_hz=max(6.0 * fmax, 4000.0))
        spectrum = build_modal_resonance_interface(operators, wide,
                                                   loss_factor=loss_factor)
        eigenvalues = np.asarray(spectrum.eigenvalues, dtype=float)
        modal_forces = np.asarray(spectrum.modal_forces, dtype=complex)
        mode_shapes = np.asarray(spectrum.mode_shapes, dtype=complex)
        natural = np.asarray(spectrum.natural_frequencies_hz, dtype=float)
        phi = mode_shapes[OBSERVATION_INDEX, :]
        participation = np.abs(phi * modal_forces)

        exact = np.array([modal_sum(frequencies, eigenvalues, modal_forces, phi, mu,
                                    loss_factor) for mu in parameters])

        print(f"=== {tag} : band 0-{fmax:.0f} Hz, {int((natural<=fmax).sum())} modes in band ===")
        print(f"  natural modes: {np.round(natural[:8], 1)}")
        case_summary = {"band_hz": [0.0, fmax],
                        "modes_in_band": int((natural <= fmax).sum()),
                        "natural_modes_hz": natural[:12].tolist(),
                        "budget": {}}
        for k in MODE_BUDGETS:
            static_error, dips, dip_freqs = [], [], []
            for index, mu in enumerate(parameters):
                pole = modal_sum(frequencies, eigenvalues[:k], modal_forces[:k],
                                 phi[:k], mu, loss_factor)
                background = exact[index] - pole
                centred = background - background.mean()
                static_error.append(float(np.linalg.norm(centred)
                                          / max(np.linalg.norm(background), 1.0e-30)))
                detected, freq = interior_dip(np.abs(exact[index]), frequencies,
                                              0.5 * fmax, fmax)
                dips.append(detected)
                if detected:
                    dip_freqs.append(freq)
            row = {
                "case": tag,
                "band_hi_hz": fmax,
                "retained_modes": k,
                "highest_retained_hz": float(natural[k - 1]),
                "retained_in_band": int((natural[:k] <= fmax).sum()),
                "masc_background_error_median": float(np.median(static_error)),
                "masc_background_error_max": float(np.max(static_error)),
                "curves_with_dip": int(sum(dips)),
                "dip_frequency_median_hz": float(np.median(dip_freqs)) if dip_freqs else float("nan"),
            }
            rows.append(row)
            case_summary["budget"][str(k)] = row
            print("    k=%d (top retained %7.1f Hz): MASC bg error median %6.2f%%  max %6.2f%%"
                  "   dips %d/%d"
                  % (k, row["highest_retained_hz"],
                     100 * row["masc_background_error_median"],
                     100 * row["masc_background_error_max"],
                     row["curves_with_dip"], parameters.shape[0]))
        summary["cases"][tag] = case_summary
        print()

    if rows:
        with (out / "modal_budget_metrics.csv").open("w", newline="",
                                                     encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (out / "modal_budget_summary.json").write_text(json.dumps(summary, indent=2),
                                                       encoding="utf-8")
        print("wrote modal_budget_metrics.csv and modal_budget_summary.json")


if __name__ == "__main__":
    main()

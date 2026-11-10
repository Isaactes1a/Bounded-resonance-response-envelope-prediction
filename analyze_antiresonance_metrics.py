# -*- coding: utf-8 -*-
"""Antiresonance-capability metrics for the equal-retained-mode comparison.

The equal-mode experiment (``run_equal_mode_dynamic_background_case.py``)
reports band-aggregated relative errors. Those aggregates dilute the one
feature that separates the two background models qualitatively: the interior
antiresonance of the transfer band. This script quantifies that feature at the
four retained modes used by both models.

For every withheld curve it reports

    dip depth ratio      min |w| over the band / shoulder level
    dip frequency        frequency of the interior amplitude minimum
    depth ratio gain     dip depth of the model / dip depth of the direct curve
    phase sweep          unwrapped phase rotation across the dip window

A model that returns no interior minimum has no resolvable antiresonance, and
its depth ratio is reported as NaN with ``dip_detected = False``; this is the
qualitative case, not merely a large error.

Reference data (all pre-existing, nothing is re-solved here):
    results/equal_mode_dynamic_background_quarter/test_direct_observation_180_n40.npz
    results/equal_mode_dynamic_background_quarter/independent_test_metrics.csv
    results/parametric_many_query_flat_5000_best/flat   (trained surrogate)

Outputs (written next to the experiment results):
    antiresonance_metrics.csv
    antiresonance_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from run_equal_mode_dynamic_background_case import (  # noqa: E402
    masc_observation,
    predict_parametric_observation_members,
    solve_static_nominal,
)
from rbf_fd_full_order.visualization.result_support import load_parametric_case  # noqa: E402

BAND_LO, BAND_HI = 130.0, 188.0     # transfer window of the reported statistics
DIP_LO, DIP_HI = 170.0, 188.0       # window in which the dip is localised
PHASE_LO, PHASE_HI = 175.0, 192.0   # window used for the phase-sweep metric


def interior_dip(amplitude: np.ndarray, frequencies: np.ndarray,
                 lo: float, hi: float) -> tuple[float, float]:
    """Depth ratio and frequency of the strongest interior amplitude minimum.

    Mirrors the detector used for candidate selection in the experiment script:
    an interior local minimum strictly inside the curve, scored against the
    larger of its two shoulders. Returns ``(nan, nan)`` when the curve has no
    interior minimum inside the window.
    """
    window = (frequencies >= lo) & (frequencies <= hi)
    curve = np.asarray(amplitude)[window]
    band = frequencies[window]
    if curve.size < 5:
        return float("nan"), float("nan")
    local = np.flatnonzero((curve[1:-1] < curve[:-2]) & (curve[1:-1] <= curve[2:])) + 1
    local = local[(local >= 2) & (local <= curve.size - 3)]
    if local.size == 0:
        return float("nan"), float("nan")
    idx = int(local[np.argmin(curve[local])])
    shoulder = max(float(np.max(curve[: idx + 1])), float(np.max(curve[idx:])), 1.0e-30)
    return float(curve[idx] / shoulder), float(band[idx])


def phase_sweep(response: np.ndarray, frequencies: np.ndarray,
                lo: float, hi: float) -> float:
    window = (frequencies >= lo) & (frequencies <= hi)
    phase = np.unwrap(np.angle(np.asarray(response)[window]))
    return float(phase[-1] - phase[0])


def dip_amplification(response: np.ndarray, reference: np.ndarray,
                      frequencies: np.ndarray, reference_frequency: float,
                      lo: float, hi: float, samples: int = 12) -> float:
    """How much larger the model response is at the direct antiresonance.

    The response of a model without a frequency-dependent background stays
    smooth through the dip, so it has no local minimum to compare. This measure
    instead evaluates every model at the frequency where the *direct* curve
    dips and normalises by the direct minimum over the same window, which is
    defined for all three models and states the failure as a number.

    ``samples`` interpolation points per grid interval are used so that the
    minimum is not locked to the $1$ Hz evaluation grid.
    """
    window = (frequencies >= lo) & (frequencies <= hi)
    fine = np.linspace(frequencies[window][0], frequencies[window][-1],
                       (int(np.count_nonzero(window)) - 1) * samples + 1)
    ref_fine = np.interp(fine, frequencies, np.abs(reference))
    ref_min = float(ref_fine.min())
    fine_freq = float(fine[int(np.argmin(ref_fine))])
    value = float(np.interp(reference_frequency, frequencies, np.abs(response)))
    return value / max(ref_min, 1.0e-30), fine_freq


def summarise(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=CODE_DIR / "results" / "equal_mode_dynamic_background_quarter",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=CODE_DIR / "results" / "parametric_many_query_flat_5000_best" / "flat",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cache_path = next(args.results.glob("test_direct_observation_*_n40.npz"))
    cache = np.load(cache_path, allow_pickle=True)
    frequencies = np.asarray(cache["frequencies_hz"], dtype=float)
    parameters = np.asarray(cache["parameters"], dtype=float)
    direct = np.asarray(cache["response"], dtype=complex)
    observation_index = int(cache["observation_index"])
    n_curves = parameters.shape[0]
    print(f"loaded {n_curves} withheld curves, observation index {observation_index}, "
          f"{frequencies.size} frequency points on {frequencies[0]:.0f}-{frequencies[-1]:.0f} Hz")

    _, _, operators, family, result, _ = load_parametric_case(args.case_dir, device=args.device)
    static_nominal = solve_static_nominal(operators)

    masc = masc_observation(family, frequencies, parameters, observation_index,
                            static_nominal[observation_index])
    surrogate = np.mean(
        predict_parametric_observation_members(result, frequencies, parameters, observation_index),
        axis=0,
    )
    print("masc and surrogate curves evaluated")

    rows: list[dict[str, object]] = []
    for index in range(n_curves):
        entry: dict[str, object] = {"curve": index}
        for name, offset in (("young_scale", 0), ("density_scale", 1),
                             ("pressure_scale", 2), ("loss_factor", 3)):
            entry[name] = float(parameters[index, offset])

        dips = {}
        for name, response in (("direct", direct[index]), ("masc", masc[index]),
                               ("surrogate", surrogate[index])):
            depth, freq = interior_dip(np.abs(response), frequencies, DIP_LO, DIP_HI)
            dips[name] = (depth, freq)
            entry[f"{name}_dip_depth_ratio"] = depth
            entry[f"{name}_dip_frequency_hz"] = freq
            entry[f"{name}_dip_detected"] = bool(np.isfinite(depth))
            entry[f"{name}_phase_sweep_rad"] = phase_sweep(response, frequencies,
                                                           PHASE_LO, PHASE_HI)

        ref_depth, ref_freq = dips["direct"]
        entry["direct_dip_reference_hz"] = ref_freq
        for name in ("masc", "surrogate"):
            depth, freq = dips[name]
            entry[f"{name}_depth_gain_vs_direct"] = (
                float(depth / ref_depth) if np.isfinite(depth) and np.isfinite(ref_depth) and ref_depth > 0 else float("nan")
            )
            entry[f"{name}_frequency_error_hz"] = (
                float(freq - ref_freq) if np.isfinite(freq) and np.isfinite(ref_freq) else float("nan")
            )

        # response at the direct antiresonance, for every model: defined even
        # when the model has no interior minimum of its own
        if np.isfinite(ref_freq):
            for name, response in (("direct", direct[index]), ("masc", masc[index]),
                                   ("surrogate", surrogate[index])):
                amp, fine_freq = dip_amplification(response, direct[index], frequencies,
                                                   ref_freq, DIP_LO, DIP_HI)
                entry[f"{name}_dip_amplification"] = amp
                entry["direct_fine_min_hz"] = fine_freq
                if name == "direct":
                    entry["direct_fine_min_hz"] = fine_freq
        else:
            for name in ("direct", "masc", "surrogate"):
                entry[f"{name}_dip_amplification"] = float("nan")
            entry["direct_fine_min_hz"] = float("nan")
        rows.append(entry)

    fieldnames = list(rows[0].keys())
    with (args.results / "antiresonance_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    detected = {
        name: int(sum(1 for row in rows if row[f"{name}_dip_detected"]))
        for name in ("direct", "masc", "surrogate")
    }
    summary = {
        "experiment": "antiresonance capability at an equal retained-mode count",
        "curves": n_curves,
        "band_hz": [BAND_LO, BAND_HI],
        "dip_window_hz": [DIP_LO, DIP_HI],
        "phase_window_hz": [PHASE_LO, PHASE_HI],
        "interior_dip_detected": detected,
        "depth_ratio_models": {
            name: summarise([row[f"{name}_dip_depth_ratio"] for row in rows])
            for name in ("direct", "masc", "surrogate")
        },
        "depth_gain_vs_direct": {
            name: summarise([row[f"{name}_depth_gain_vs_direct"] for row in rows])
            for name in ("masc", "surrogate")
        },
        "frequency_error_hz": {
            name: summarise([row[f"{name}_frequency_error_hz"] for row in rows])
            for name in ("masc", "surrogate")
        },
        "phase_sweep_rad": {
            name: summarise([row[f"{name}_phase_sweep_rad"] for row in rows])
            for name in ("direct", "masc", "surrogate")
        },
        "dip_amplification": {
            name: summarise([row[f"{name}_dip_amplification"] for row in rows])
            for name in ("direct", "masc", "surrogate")
        },
    }
    (args.results / "antiresonance_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    # curves of the three models over the withheld set, for the figure
    np.savez_compressed(
        args.results / "antiresonance_curves.npz",
        frequencies_hz=frequencies,
        parameters=parameters,
        direct=direct,
        masc=masc,
        surrogate=surrogate,
        observation_index=observation_index,
    )

    print()
    print(f"interior dip detected by: {detected} of {n_curves} curves")
    for name in ("direct", "masc", "surrogate"):
        s = summary["depth_ratio_models"][name]
        print(f"  depth ratio {name:<9} median {s.get('median', float('nan')):.3e} "
              f"min {s.get('min', float('nan')):.3e} max {s.get('max', float('nan')):.3e}")
    for name in ("masc", "surrogate"):
        g = summary["depth_gain_vs_direct"][name]
        p = summary["phase_sweep_rad"][name]
        print(f"  {name:<9} depth gain median {g.get('median', float('nan')):.1f}x   "
              f"phase sweep median {p.get('median', float('nan')):+.3f} rad")
    d = summary["phase_sweep_rad"]["direct"]
    print(f"  direct    phase sweep median {d.get('median', float('nan')):+.3f} rad")
    print()
    for name in ("direct", "masc", "surrogate"):
        a = summary["dip_amplification"][name]
        print(f"  response at the direct dip, {name:<9} median {a.get('median', float('nan')):.2f}x "
              f"min {a.get('min', float('nan')):.2f}x max {a.get('max', float('nan')):.2f}x")
    print()
    print("wrote antiresonance_metrics.csv, antiresonance_summary.json, antiresonance_curves.npz")


if __name__ == "__main__":
    main()

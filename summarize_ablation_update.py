# -*- coding: utf-8 -*-
"""Collect the 0--600 Hz stiffened values that replace Appendix C.4--C.6.

Reads the rebuilt ablation, the production/ablation/phase7 equilibrium-residual
diagnostics and the rebuilt low-label study, and prints the table rows and the
derived ratios that the manuscript quotes.  Nothing is written back to the
manuscript; the numbers are printed for verification and transcription.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("results")
ABLATION = RESULTS / "parametric_ablation_stiffened_0_600" / "stiffened" / "ablation_summary.json"
PRODUCTION_RESIDUAL = RESULTS / "equilibrium_residual_0_600" / "production_residual.json"
ABLATION_RESIDUAL = RESULTS / "equilibrium_residual_0_600" / "ablation_residual.json"
PHASE7_RESIDUAL = RESULTS / "equilibrium_residual_0_600" / "phase7_residual.json"
LOWLABEL = RESULTS / "phase7" / "lowlabel_summary.json"
VARIANTS = ("direct_data", "direct_physics", "pole_data", "full")


def _load(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sci(value: float, digits: int = 4) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def main() -> None:
    ablation = _load(ABLATION)
    if ablation:
        rows = {row["variant"]: row for row in ablation["rows"]}
        band = ablation.get("frequency_interval_hz")
        print(f"ablation band {band}  points {ablation.get('sweep_points')}")
        print("\nTable C.5 -- stiffened rows")
        for variant in VARIANTS:
            row = rows[variant]
            print(
                "  {:14s} & {} & {:.4f} & {:.4f} & {} \\\\".format(
                    variant.replace("_", " ").title(),
                    _sci(row["full_domain_relative_mse"]),
                    row["peak_frequency_error_interp_mean_hz"],
                    row["peak_frequency_error_interp_max_hz"],
                    _sci(row["peak_amplitude_relative_error_mean"], 3),
                )
            )
        print(
            "\n  orders of magnitude direct_data -> pole_data: {:.2f}  (direct {:.4f} -> pole {})".format(
                len(str(int(rows["direct_data"]["full_domain_relative_mse"])))
                - len(str(int(1.0 / max(rows["pole_data"]["full_domain_relative_mse"], 1e-300)))),
                rows["direct_data"]["full_domain_relative_mse"],
                _sci(rows["pole_data"]["full_domain_relative_mse"]),
            )
        )
        ratio = rows["direct_data"]["full_domain_relative_mse"] / rows["direct_physics"]["full_domain_relative_mse"]
        print(f"  equilibrium-term gain on pole-free variant: {ratio:.2f}x "
              f"({rows['direct_data']['full_domain_relative_mse']:.4f} -> "
              f"{rows['direct_physics']['full_domain_relative_mse']:.4f})")
        change = abs(rows["full"]["full_domain_relative_mse"]
                     - rows["pole_data"]["full_domain_relative_mse"]) / rows["pole_data"]["full_domain_relative_mse"]
        print(f"  full vs pole+data MSE change: {100 * change:.3f}%")
        peak_low = min(rows[v]["peak_frequency_error_interp_mean_hz"] for v in ("pole_data", "full"))
        peak_high = max(rows[v]["peak_frequency_error_interp_mean_hz"] for v in ("pole_data", "full"))
        amp_low = min(rows[v]["peak_amplitude_relative_error_mean"] for v in ("pole_data", "full"))
        amp_high = max(rows[v]["peak_amplitude_relative_error_mean"] for v in ("pole_data", "full"))
        print(f"  pole-assisted peak-freq err max {max(peak_low, peak_high):.2e} Hz, "
              f"peak-amp err {100 * amp_low:.3f}-{100 * amp_high:.3f}%")

    production = _load(PRODUCTION_RESIDUAL)
    if production:
        print("\nTable C.6 -- production rows")
        for row in production:
            print("  Production & {:11s} & {:.4f} & {:.4f} & {:.4f} \\\\".format(
                row["case"].title() if row["case"] != "flat" else "Unstiffened",
                row["relative_residual_mean"], row["relative_residual_p95"], row["relative_residual_max"]))
            print(f"      [frequencies {row['frequency_count']}, curves {row['curve_count']}]")

    ablation_residual = _load(ABLATION_RESIDUAL)
    if ablation_residual:
        print("\nTable C.6 -- ablation rows")
        for row in ablation_residual:
            if row["case"] != "stiffened":
                continue
            print("  {:14s} & Stiffened & {:.4f} & {:.3f} & {:.2f} \\\\".format(
                row["checkpoint"], row["relative_residual_mean"],
                row["relative_residual_p95"], row["relative_residual_max"]))
            print(f"      [frequencies {row['frequency_count']}]")

    phase7 = _load(PHASE7_RESIDUAL)
    if phase7:
        print("\nphase7 (low-label) residual comparison")
        by_checkpoint = {row["checkpoint"].split("/")[-1]: row for row in phase7 if row["case"] == "stiffened"}
        for stem, row in sorted(by_checkpoint.items()):
            print(f"  {stem:28s} mean {row['relative_residual_mean']:.6f}")

    lowlabel = _load(LOWLABEL)
    if lowlabel:
        print("\nlow-label study -- pole+data vs full")
        rows = lowlabel["rows"]
        by = {(row["n_train"], row["variant"]): row for row in rows}
        for n_train in sorted({row["n_train"] for row in rows}):
            pole = by[(n_train, "pole_data")]
            full = by[(n_train, "full")]
            change = abs(full["full_domain_relative_mse"] - pole["full_domain_relative_mse"]) / pole["full_domain_relative_mse"]
            print(f"  n={n_train:2d}  pole {pole['full_domain_relative_mse']:.4e}  "
                  f"full {full['full_domain_relative_mse']:.4e}  change {100 * change:.4f}%")


if __name__ == "__main__":
    main()

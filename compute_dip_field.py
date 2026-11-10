# -*- coding: utf-8 -*-
"""Transverse displacement fields at the nominal antiresonance of the benchmark.

The equal-retained-mode comparison separates the two background models on the
observation curve.  This script stores the same comparison on the complete
field, at the frequency of the direct antiresonance and at the nominal
parameter state, so the figure can show where the cancellation sits in space:

    direct    constrained RBF-FD reference
    static    four-mode mode-acceleration static correction (MASC)
    learned   four-mode pole branch + learned dynamic remainder

The reference field is read from the cached nominal direct sweep, the static
field is rebuilt from the static flexibility term, and the learned field is the
ensemble mean of the production surrogate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from run_equal_mode_dynamic_background_case import (  # noqa: E402
    masc_full,
    solve_static_nominal,
)
from rbf_fd_full_order import predict_parametric_members  # noqa: E402
from rbf_fd_full_order.visualization.result_support import load_parametric_case  # noqa: E402

CASE_DIR = Path("results") / "parametric_many_query_flat_5000_best" / "flat"
CACHE = Path("results") / "equal_mode_dynamic_background_quarter" / "nominal_direct_full.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=183.0)
    parser.add_argument("--out", type=Path, default=Path("results") / "dip_field")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    metadata, system, operators, family, result, observation_index = load_parametric_case(
        CASE_DIR, device=args.device)
    nominal = np.asarray(family.bounds.center, dtype=float)
    nodes = np.asarray(system.nodes, dtype=float)
    n_nodes = int(nodes.shape[0])
    grid = int(round(np.sqrt(n_nodes)))
    print(f"case={metadata.get('case')}  nodes={n_nodes}  grid={grid}x{grid}  "
          f"free dofs={operators.n_free}  retained modes={family.modal.n_modes}")

    with np.load(CACHE) as archive:
        frequencies = np.asarray(archive["frequencies_hz"], dtype=float)
        cached_parameters = np.asarray(archive["parameters"], dtype=float)
        direct_full = np.asarray(archive["response"], dtype=complex)
    if not np.allclose(cached_parameters, nominal):
        raise SystemExit("cached nominal state does not match the parameter centre")
    index = int(np.argmin(np.abs(frequencies - args.frequency)))
    frequency = float(frequencies[index])
    print(f"frequency {frequency:.3f} Hz (cached index {index})")

    static_nominal = solve_static_nominal(operators)
    query = np.asarray([frequency], dtype=float)
    static_field = np.asarray(masc_full(family, query, nominal, static_nominal), dtype=complex)[0]
    members = np.asarray(predict_parametric_members(result, query, nominal[None, :]), dtype=complex)
    learned_field = members.mean(axis=0)[0]
    direct_field = direct_full[index]


    def transverse(response: np.ndarray) -> np.ndarray:
        expanded = np.asarray(operators.expand(np.asarray(response, dtype=complex).reshape(1, -1)),
                              dtype=complex)[0]
        return expanded[:n_nodes].reshape(grid, grid)

    fields = {
        "direct": transverse(direct_field),
        "static": transverse(static_field),
        "learned": transverse(learned_field),
    }
    centre = (grid // 2, grid // 2)
    summary = {
        "case": metadata.get("case"),
        "frequency_hz": frequency,
        "retained_modes": int(family.modal.n_modes),
        "observation_index": int(observation_index),
        "nominal_parameters": nominal.tolist(),
        "centre_amplitude_mm": {key: float(np.abs(value[centre]) * 1.0e3)
                                for key, value in fields.items()},
        "field_max_mm": {key: float(np.abs(value).max() * 1.0e3)
                         for key, value in fields.items()},
    }
    print(json.dumps(summary["centre_amplitude_mm"], indent=2))
    print(json.dumps(summary["field_max_mm"], indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "nominal_dip_field.npz",
        frequency_hz=np.asarray(frequency),
        grid=np.asarray(grid),
        nodes=nodes,
        direct=np.abs(fields["direct"]) * 1.0e3,
        static=np.abs(fields["static"]) * 1.0e3,
        learned=np.abs(fields["learned"]) * 1.0e3,
        direct_complex=fields["direct"],
        static_complex=fields["static"],
        learned_complex=fields["learned"],
    )
    (args.out / "nominal_dip_field.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"written {args.out / 'nominal_dip_field.npz'}")


if __name__ == "__main__":
    main()

"""Generate the current 4x4 field figures and withheld frequency curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from rbf_fd_full_order import prepare_constrained_operators
from rbf_fd_full_order.examples.cases import MeshConfig, build_case_system
from rbf_fd_full_order.visualization import (
    COLOR_PROPOSED,
    COLOR_RBF_REFERENCE,
    add_consistent_colorbar,
    apply_journal_style,
    save_figure,
    style_axis,
)


CASES = ("flat", "annular", "stiffened")
TITLES = {
    "flat": "Flat plate",
    "annular": "Circular plate with central hole",
    "hollow": "Square plate with circular hole",
    "stiffened": "Stiffened plate",
}
FIELD_COLUMNS = (
    ("Data\nno correction", "full_data_only", "pod_data_only"),
    ("Data\n+ correction", "full_data_only_refined", "pod_data_only_refined"),
    ("Physics\nno correction", "full_physics", "pod_physics"),
    ("Physics\n+ correction", "full_physics_refined", "pod_physics_refined"),
)
FIELD_ROWS = ("FOM result", "POD result", "FOM error", "POD error")
FREQUENCY_METHODS = (
    "full_data_only",
    "full_data_only_refined",
    "full_physics",
    "full_physics_refined",
    "pod_data_only",
    "pod_data_only_refined",
    "pod_physics",
    "pod_physics_refined",
)
FREQUENCY_STYLES = {
    # Colour: FOM/POD; line: data/physics; marker: direct/corrected.
    "full_data_only": ("#C00000", "--", "o", "FOM-D"),
    "full_data_only_refined": ("#C00000", "--", "^", "FOM-D+R"),
    "full_physics": ("#C00000", "-", "o", "FOM-P"),
    "full_physics_refined": ("#C00000", "-", "^", "FOM-P+R"),
    "pod_data_only": ("#0070C0", "--", "o", "POD-D"),
    "pod_data_only_refined": ("#0070C0", "--", "^", "POD-D+R"),
    "pod_physics": ("#0070C0", "-", "o", "POD-P"),
    "pod_physics_refined": ("#0070C0", "-", "^", "POD-P+R"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results") / "unified_square_cases_modal_sweep_cuda",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "unified_square_cases_modal_sweep_figures",
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("results") / "unified_square_cases_modal_sweep_cuda",
        help="Directory containing the saved 0--200 Hz sweep outputs.",
    )
    return parser.parse_args()


def _load_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _selected_index(case_dir: Path) -> tuple[int, float, float]:
    frequencies = np.load(case_dir / "frequency.npy")
    rows = [
        row
        for row in _load_summary(case_dir / "validation_summary.csv")
        if row["method"] == "pod_physics_refined"
    ]
    selected = max(rows, key=lambda row: float(row["relative_field_error"]))
    frequency = float(selected["frequency_hz"])
    index = int(np.flatnonzero(np.isclose(frequencies, frequency))[0])
    return index, frequency, float(selected["relative_field_error"])


def _field_from_reduced(case: str, response: np.ndarray):
    system = build_case_system(case, MeshConfig())
    operators = prepare_constrained_operators(system)
    full = operators.expand(np.asarray(response, dtype=complex))
    n_nodes = system.nodes.shape[0]
    nodes = system.nodes
    field = np.column_stack((full[:n_nodes], full[n_nodes : 2 * n_nodes], full[2 * n_nodes : 3 * n_nodes]))
    return np.asarray(nodes, dtype=float), np.asarray(field, dtype=complex)


def _triangulation(case: str, nodes: np.ndarray) -> mtri.Triangulation:
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    if case in ("hollow", "annular"):
        triangles = nodes[tri.triangles]
        center = np.asarray([0.5, 0.5]) if case == "hollow" else np.zeros(2)
        radii = np.linalg.norm(np.mean(triangles, axis=1) - center, axis=1)
        tri.set_mask(radii < 0.125 * (1.0 + 1.0e-8))
    return tri


def _draw_geometry(axis: plt.Axes, case: str, nodes: np.ndarray) -> None:
    if case == "hollow":
        angle = np.linspace(0.0, 2.0 * np.pi, 361)
        axis.plot(0.5 + 0.125 * np.cos(angle), 0.5 + 0.125 * np.sin(angle), color="black", lw=0.8)
    if case == "annular":
        angle = np.linspace(0.0, 2.0 * np.pi, 361)
        axis.plot(0.125 * np.cos(angle), 0.125 * np.sin(angle), color="black", lw=0.8)
        axis.plot(0.5 * np.cos(angle), 0.5 * np.sin(angle), color="black", lw=0.8)
        return
    xmin, ymin = np.min(nodes, axis=0)
    xmax, ymax = np.max(nodes, axis=0)
    polygon = np.asarray([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]])
    axis.plot(polygon[:, 0], polygon[:, 1], color="black", lw=0.8)


def _plot_field_panel(
    axis: plt.Axes,
    tri: mtri.Triangulation,
    values: np.ndarray,
    *,
    case: str,
    nodes: np.ndarray,
    vmax: float,
) -> matplotlib.contour.QuadContourSet:
    contour = axis.tricontourf(
        tri,
        values,
        levels=np.linspace(0.0, vmax, 29),
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )
    axis.triplot(tri, color="black", lw=0.12, alpha=0.16)
    _draw_geometry(axis, case, nodes)
    axis.set_aspect("equal", adjustable="box")
    style_axis(axis)
    return contour


def make_field_figures(results_root: Path, output_dir: Path) -> list[dict[str, object]]:
    """Generate 4-by-4 all-method prediction/error figures for each case."""

    selected_rows: list[dict[str, object]] = []
    for case in CASES:
        case_dir = results_root / case
        index, frequency, pod_error = _selected_index(case_dir)
        reference = np.load(case_dir / "reference_response_real.npy")[index] + 1j * np.load(
            case_dir / "reference_response_imag.npy"
        )[index]
        nodes, reference_field = _field_from_reduced(case, reference)
        tri = _triangulation(case, nodes)
        reference_w = reference_field[:, 0]
        predicted_fields: dict[str, np.ndarray] = {}
        for _, fom_method, pod_method in FIELD_COLUMNS:
            for method in (fom_method, pod_method):
                response = np.load(case_dir / f"prediction_{method}_real.npy")[index] + 1j * np.load(
                    case_dir / f"prediction_{method}_imag.npy"
                )[index]
                _, field = _field_from_reduced(case, response)
                predicted_fields[method] = field[:, 0]

        fom_result = [np.abs(predicted_fields[fom_method]) * 1.0e3 for _, fom_method, _ in FIELD_COLUMNS]
        pod_result = [np.abs(predicted_fields[pod_method]) * 1.0e3 for _, _, pod_method in FIELD_COLUMNS]
        fom_error = [
            np.abs(predicted_fields[fom_method] - reference_w) * 1.0e3
            for _, fom_method, _ in FIELD_COLUMNS
        ]
        pod_error_fields = [
            np.abs(predicted_fields[pod_method] - reference_w) * 1.0e3
            for _, _, pod_method in FIELD_COLUMNS
        ]
        row_values = (fom_result, pod_result, fom_error, pod_error_fields)
        row_vmax = [max(float(np.max(values)) for values in row) for row in row_values]

        figure = plt.figure(figsize=(6.42, 6.95), constrained_layout=True)
        grid = figure.add_gridspec(4, 5, width_ratios=(1.0, 1.0, 1.0, 1.0, 0.075))
        axes: list[list[plt.Axes]] = [
            [figure.add_subplot(grid[row_index, column_index]) for column_index in range(4)]
            for row_index in range(4)
        ]
        colorbar_axes = [figure.add_subplot(grid[row_index, 4]) for row_index in range(4)]
        x_min, y_min = np.min(nodes, axis=0)
        x_max, y_max = np.max(nodes, axis=0)
        x_ticks = np.linspace(x_min, x_max, 3)
        y_ticks = np.linspace(y_min, y_max, 3)
        for row_index, (row_name, values, vmax) in enumerate(zip(FIELD_ROWS, row_values, row_vmax)):
            mappable = None
            for column_index, ((column_title, _, _), field_values) in enumerate(zip(FIELD_COLUMNS, values)):
                axis = axes[row_index][column_index]
                mappable = _plot_field_panel(
                    axis,
                    tri,
                    field_values,
                    case=case,
                    nodes=nodes,
                    vmax=max(vmax, 1.0e-30),
                )
                axis.set_xlim(x_min, x_max)
                axis.set_ylim(y_min, y_max)
                axis.set_xticks(x_ticks)
                axis.set_yticks(y_ticks)
                if row_index == 0:
                    axis.set_title(column_title)
                if row_index == len(FIELD_ROWS) - 1:
                    axis.set_xlabel("x (m)")
                else:
                    axis.set_xticklabels([])
                if column_index == 0:
                    axis.set_ylabel(f"{row_name}\n\ny (m)")
                else:
                    axis.set_yticklabels([])
            colorbar_label = "|w| (mm)" if row_index < 2 else "absolute error (mm)"
            add_consistent_colorbar(
                figure,
                mappable,
                None,
                colorbar_label,
                cax=colorbar_axes[row_index],
                endpoints_only=True,
            )
        selected_rows.append(
            {
                "case": case,
                "selected_withheld_frequency_hz": frequency,
                "pod_refined_relative_field_error": pod_error,
                "max_fom_result_mm": row_vmax[0],
                "max_pod_result_mm": row_vmax[1],
                "max_fom_error_mm": row_vmax[2],
                "max_pod_error_mm": row_vmax[3],
            }
        )
        save_figure(figure, output_dir / f"all_method_field_error_comparison_{case}.png", tight=False)
        plt.close(figure)
    return selected_rows


def make_frequency_figure(sweep_root: Path, output_dir: Path) -> None:
    """Plot all eight absolute-response scans in a three-row, one-column figure."""

    figure, axes = plt.subplots(3, 1, figsize=(6.42, 7.55))
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.13, top=0.96, hspace=0.34)
    legend_handles = None
    for row_index, (axis, case) in enumerate(zip(axes, CASES)):
        rows = _load_summary(sweep_root / case / "frequency_sweep_observation.csv")
        frequency = np.asarray([float(row["frequency_hz"]) for row in rows])
        reference = np.asarray([float(row["reference_abs_mm"]) for row in rows])
        reference_line = axis.plot(
            frequency,
            reference,
            color=COLOR_RBF_REFERENCE,
            linestyle="-",
            lw=1.45,
            label="RBF-FD reference",
            zorder=1,
        )[0]
        handles = []
        for method in FREQUENCY_METHODS:
            color, line_style, marker, label = FREQUENCY_STYLES[method]
            amplitude = np.asarray([float(row[f"{method}_abs_mm"]) for row in rows])
            line = axis.plot(
                frequency,
                amplitude,
                color=color,
                linestyle=line_style,
                marker=marker,
                markevery=10,
                markersize=4.2,
                markerfacecolor="white",
                markeredgewidth=0.9,
                lw=1.15,
                label=label,
            )[0]
            handles.append(line)
        if legend_handles is None:
            legend_handles = [reference_line, *handles]
        axis.set_title(TITLES[case])
        axis.set_xlim(0.0, 200.0)
        axis.set_xticks(np.arange(0.0, 201.0, 50.0))
        axis.set_xlabel("frequency (Hz)" if row_index == len(CASES) - 1 else "")
        axis.set_ylabel("|w| (mm)")
        axis.grid(True, alpha=0.28)
        style_axis(axis)
    figure.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.00),
    )
    save_figure(figure, output_dir / "withheld_frequency_response_comparison.png", tight=False)
    plt.close(figure)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    apply_journal_style()
    selected_rows = make_field_figures(args.results_root, args.out)
    make_frequency_figure(args.sweep_root, args.out)
    write_rows(args.out / "selected_withheld_field_metrics.csv", selected_rows)
    print(f"Wrote updated manuscript figures to {args.out}")


if __name__ == "__main__":
    main()

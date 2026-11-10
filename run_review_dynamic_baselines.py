"""Frozen, same-operator modal and dynamic-POD comparisons for the review.

This script never trains a network or changes an existing reference archive.
The new outputs are deliberately separate from the manuscript.  A rank-eight
real/imaginary POD trial space is reused without fitting it to test labels.
Its coefficients are either supplied by the frozen five-network mean, or by
linear least squares on the discrete equilibrium (raw and fixed W(0, mu_c)).
Thin QR of the operator images makes the latter an exact compressed least-
squares problem, rather than a normal-equation approximation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

# These limits also cover BLAS calls made before torch is imported.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as spla
import torch

from rbf_fd_full_order import ModalResonanceConfig, build_modal_resonance_interface
from rbf_fd_full_order import predict_parametric_observation_members
from rbf_fd_full_order.visualization.result_support import load_parametric_case


CODE = Path(__file__).resolve().parent
DEFAULT_OUT = CODE / "results/review_validation_20261103/baselines"
CASE_DIRS = {
    "flat": CODE / "results/parametric_many_query_flat_5000_best/flat",
    "stiffened": CODE / "results/parametric_many_query_stiffened_6mode_0_600/stiffened",
}
BUDGETS = {"flat": (4, 6, 8, 12, 16, 24), "stiffened": (6, 8, 12, 16, 24)}
EXPECTED_POLES = {"flat": 4, "stiffened": 6}
EIGSH_CALLS: list[dict] = []


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cases", nargs="+", choices=tuple(CASE_DIRS), default=tuple(CASE_DIRS))
    p.add_argument("--max-test-curves", type=int, default=40)
    p.add_argument("--max-field-states", type=int, default=8)
    p.add_argument("--primary-field-states", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--timing-runs", type=int, default=7)
    return p.parse_args()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(x) for x in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(clean_json(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fingerprint(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": h.hexdigest()}


@contextmanager
def deterministic_eigsh():
    original = spla.eigsh

    def fixed(A, *pos, **kw):
        if kw.get("v0") is None:
            v = np.random.default_rng(20261103).standard_normal(A.shape[0])
            kw["v0"] = v / np.linalg.norm(v)
        EIGSH_CALLS.append({"n": A.shape[0], "requested_k": kw.get("k", pos[0] if pos else 6),
                            "ncv": kw.get("ncv"), "tol": kw.get("tol", 0.0),
                            "maxiter": kw.get("maxiter"), "sigma": kw.get("sigma"),
                            "v0_rule": "default_rng(20261103).standard_normal(n), normalized"})
        return original(A, *pos, **kw)

    spla.eigsh = fixed
    try:
        yield
    finally:
        spla.eigsh = original


def stack(z):
    return np.concatenate((np.asarray(z).real, np.asarray(z).imag), axis=0)


def radius(mu, bounds):
    return np.linalg.norm((np.asarray(mu) - bounds.center) / ((bounds.upper - bounds.lower) / 2), axis=-1)


def near_cluster(a, b):
    return abs(a - b) <= max(0.02, 2.0e-5 * max(abs(a), abs(b)))


def complete_count(target, frequencies):
    k = int(target)
    while k < len(frequencies) and near_cluster(frequencies[k - 1], frequencies[k]):
        k += 1
    return k


def coeffs(eigenvalues, forces, frequencies, mu):
    se, sr, sp, eta = mu
    d = (1 + 1j * eta) * se * eigenvalues[None, :] - sr * (2 * np.pi * np.asarray(frequencies)[:, None]) ** 2
    return sp * forces[None, :] / d


class FrozenNN:
    name = "frozen_nn_mean_complex"

    def __init__(self, result):
        self.result = result
        self.poles = result.family.modal.reference
        self.n = result.n_dofs
        self.v = result.pod_basis.basis[:self.n] + 1j * result.pod_basis.basis[self.n:]
        self.mean = result.coefficient_scale * (result.pod_basis.mean[:self.n] + 1j * result.pod_basis.mean[self.n:])
        self.actual_count = self.poles.n_modes
        self.requested_count = self.actual_count

    def coordinates(self, f, mu):
        nf = self.result.normalize_frequency(np.asarray(f))
        nm = self.result.normalize_parameters(np.repeat(np.asarray(mu)[None, :], len(f), axis=0))
        raws = []
        for member in self.result.members:
            parameter = next(member.network.parameters())
            with torch.no_grad():
                raw = member.network(torch.as_tensor(nf, dtype=parameter.dtype, device=parameter.device),
                                     torch.as_tensor(nm, dtype=parameter.dtype, device=parameter.device))
            raws.append(raw.cpu().numpy())
        return np.mean(raws, axis=0) * self.result.pod_basis.alpha_scale[None, :] * self.result.coefficient_scale

    def predict(self, f, mu, indices=None):
        ix = np.arange(self.n) if indices is None else np.asarray(indices, dtype=int)
        c = coeffs(self.poles.eigenvalues, self.poles.modal_forces, f, mu)
        return self.mean[ix][None, :] + self.coordinates(f, mu) @ self.v[ix].T + c @ self.poles.mode_shapes[ix].T


class Modal:
    def __init__(self, spectrum, k, target, static, corrected, source="shared_wide"):
        self.actual_count, self.requested_count = k, target
        self.lam = spectrum.eigenvalues[:k]
        self.forces = spectrum.modal_forces[:k]
        self.phi = spectrum.mode_shapes[:, :k]
        self.static = static
        self.corrected = corrected
        self.spectrum_source = source
        self.name = f"{'masc' if corrected else 'ms'}_requested{target}_actual{k}"

    def predict(self, f, mu, indices=None):
        ix = np.arange(len(self.static)) if indices is None else np.asarray(indices, dtype=int)
        pred = coeffs(self.lam, self.forces, f, mu) @ self.phi[ix].T
        if self.corrected:
            se, _, sp, eta = mu
            residual = self.static[ix] - self.phi[ix] @ (self.forces / self.lam)
            pred += sp / ((1 + 1j * eta) * se) * residual[None, :]
        return pred


class LSPG:
    """Exact thin-QR compression of a weighted full residual trial problem."""

    def __init__(self, nn, family, weighted):
        self.family, self.nn = family, nn
        self.actual_count = self.requested_count = nn.actual_count
        self.name = "lspg_fixed_W0" if weighted else "lspg_raw"
        self.n, self.q, self.p = nn.n, nn.v.shape[1], nn.actual_count
        self.weight = family.row_scaling_paired(np.array([0.0]), family.bounds.center)[0] if weighted else np.ones(self.n)
        self.weight_block = np.r_[self.weight, self.weight]
        phi = nn.poles.mode_shapes
        # The real POD coordinates are kept real.  Pole coordinates need both
        # real and imaginary columns; the affine mean is already physical.
        pole_block = la.block_diag(phi, phi)
        u = np.column_stack((np.vstack((nn.v.real, nn.v.imag)), stack(nn.mean), pole_block))
        self.u = u
        self.cols = u.shape[1]
        K, M = family.nominal.K, family.nominal.M
        ku = np.vstack((K @ u[:self.n], K @ u[self.n:]))
        jku = np.vstack((-ku[self.n:], ku[:self.n]))
        mass = np.vstack((M @ u[:self.n], M @ u[self.n:]))
        image = np.column_stack((ku, jku, mass, u, stack(family.nominal.rhs)))
        image *= self.weight_block[:, None]
        started = time.perf_counter()
        orthogonal, self.r = la.qr(image, mode="economic", check_finite=False)
        self.preprocess_s = time.perf_counter() - started
        self.qr_relative_error = np.linalg.norm(orthogonal @ self.r - image) / np.linalg.norm(image)
        self.kdiag = np.asarray(K.diagonal()).real
        self.mdiag = np.asarray(M.diagonal()).real
        self.minimum_rank, self.maximum_condition, self.rank_failure_count = self.q, 0.0, 0
        self.rank_failure_examples = []

    def shift(self, f, mu):
        se, sr, _, eta = mu
        scale = max(float(np.max(np.abs((1 + 1j * eta) * se * self.kdiag - (2 * np.pi * f) ** 2 * sr * self.mdiag))), 1.0)
        return self.family.nominal.diagonal_regularization * scale

    def coordinates(self, f, mu):
        se, sr, sp, eta = mu
        c = self.cols
        pole = coeffs(self.nn.poles.eigenvalues, self.nn.poles.modal_forces, f, mu)
        answers = []
        for j, fj in enumerate(f):
            d = (se * self.r[:, :c] + eta * se * self.r[:, c:2*c]
                 - sr * (2 * np.pi * fj) ** 2 * self.r[:, 2*c:3*c]
                 + self.shift(float(fj), mu) * self.r[:, 3*c:4*c])
            known = np.r_[1.0, pole[j].real, pole[j].imag]
            b = sp * self.r[:, -1] - d[:, self.q:] @ known
            a = d[:, :self.q]
            scale = np.maximum(np.linalg.norm(a, axis=0), np.finfo(float).tiny)
            scaled, _, rank_, singular = la.lstsq(a / scale[None, :], b, cond=1e-12, lapack_driver="gelsd", check_finite=False)
            answers.append(scaled / scale)
            condition = float(singular[0] / singular[-1]) if singular[-1] else float("inf")
            self.minimum_rank = min(self.minimum_rank, int(rank_))
            self.maximum_condition = max(self.maximum_condition, condition)
            if rank_ < self.q:
                self.rank_failure_count += 1
                if len(self.rank_failure_examples) < 20:
                    self.rank_failure_examples.append({"f_hz": float(fj), "mu": np.asarray(mu).tolist(), "rank": int(rank_)})
        return np.asarray(answers)

    def predict(self, f, mu, indices=None):
        ix = np.arange(self.n) if indices is None else np.asarray(indices, dtype=int)
        pole = coeffs(self.nn.poles.eigenvalues, self.nn.poles.modal_forces, f, mu)
        return self.nn.mean[ix][None, :] + self.coordinates(f, mu) @ self.nn.v[ix].T + pole @ self.nn.poles.mode_shapes[ix].T

    def explicit_check(self, f, mu):
        dynamic = self.family._dynamic(float(f), mu)
        pole = self.nn.poles.mode_shapes @ coeffs(self.nn.poles.eigenvalues, self.nn.poles.modal_forces, [f], mu)[0]
        a = np.vstack(((dynamic @ self.nn.v).real, (dynamic @ self.nn.v).imag)) * self.weight_block[:, None]
        b = stack(self.family.rhs(mu) - dynamic @ (self.nn.mean + pole)) * self.weight_block
        scales = np.linalg.norm(a, axis=0)
        expected, _, rank_, _ = la.lstsq(a / scales, b, cond=1e-12, lapack_driver="gelsd")
        expected /= scales
        actual = self.coordinates([f], mu)[0]
        response_difference = np.linalg.norm(self.nn.v @ (actual - expected)) / max(np.linalg.norm(self.nn.v @ expected), 1e-30)
        residual_expected = np.linalg.norm(a @ expected - b)
        residual_actual = np.linalg.norm(a @ actual - b)
        return {"frequency_hz": f, "rank_explicit": int(rank_), "coefficient_relative_difference": float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1e-30)),
                "background_response_relative_difference": float(response_difference),
                "explicit_residual": float(residual_expected), "compressed_residual": float(residual_actual)}


def rel(pred, ref):
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(ref)) / max(np.linalg.norm(ref), 1e-30))


def curve_metrics(pred, ref, amplitude_only=False):
    ref_amp = ref if amplitude_only else np.abs(ref)
    data = {"amplitude_l2_percent": 100 * rel(np.abs(pred), ref_amp)}
    if not amplitude_only:
        data.update(complex_l2_percent=100 * rel(pred, ref),
                    wrapped_phase_rmse_rad=float(np.sqrt(np.mean(np.angle(pred * np.conj(ref)) ** 2))))
    return data


def dip_info(f, curve, case):
    # The detector is applied independently to every predicted curve.
    lo, hi = (170.0, 188.0) if case == "flat" else (350.0, 500.0)
    shoulder_lo = 130.0 if case == "flat" else lo
    f, y = np.asarray(f), np.abs(curve)
    interior = np.flatnonzero((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:])) + 1
    candidates = interior[(f[interior] >= lo) & (f[interior] <= hi)]
    window = (f >= shoulder_lo) & (f <= hi)
    phase = np.unwrap(np.angle(np.asarray(curve)[window]))
    result = {"detected": bool(candidates.size), "phase_sweep_rad": float(phase[-1]-phase[0]) if phase.size > 1 else float("nan")}
    if candidates.size:
        idx = candidates[np.argmin(y[candidates])]
        shoulder = max(float(y[window][0]), float(y[window][-1]), 1e-30)
        result.update(frequency_hz=float(f[idx]), amplitude_m=float(y[idx]), depth_ratio=float(y[idx]/shoulder))
    return result


def dip_comparison(f, pred, ref, case):
    p, r = dip_info(f, pred, case), dip_info(f, ref, case)
    result = {"reference_detected": r["detected"], "predicted_detected": p["detected"],
              "missed": r["detected"] and not p["detected"], "spurious": p["detected"] and not r["detected"],
              "reference_phase_sweep_rad": r["phase_sweep_rad"], "predicted_phase_sweep_rad": p["phase_sweep_rad"]}
    if r["detected"]:
        result.update(reference_dip_hz=r["frequency_hz"], reference_depth=r["depth_ratio"])
    if p["detected"]:
        result.update(predicted_dip_hz=p["frequency_hz"], predicted_depth=p["depth_ratio"])
    if r["detected"] and p["detected"]:
        result.update(dip_frequency_error_hz=abs(p["frequency_hz"]-r["frequency_hz"]), depth_gain=p["depth_ratio"]/r["depth_ratio"])
    return result


def timing(function, runs):
    function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter()-start)
    return {"mean_s": float(np.mean(samples)), "median_s": float(np.median(samples)), "std_s": float(np.std(samples)),
            "minimum_s": float(np.min(samples)), "maximum_s": float(np.max(samples)), "runs": runs, "samples_s": samples}


def observation_indices(system, operators):
    indices = []
    for xy in ((0.5, 0.5), (7/30, 7/30)):
        node = int(np.argmin(np.sum((system.nodes - np.asarray(xy)[None, :])**2, axis=1)))
        assert np.allclose(system.nodes[node], xy, atol=1e-12)
        indices.append(int(np.flatnonzero(operators.free_dofs == node)[0]))
    return indices


def direct_points(family, points, path):
    """New state-frequency pairs only; resume in this independent output tree."""
    mus = np.asarray([x[0] for x in points])
    freqs = np.asarray([x[1] for x in points])
    if path.exists():
        old = np.load(path)
        if np.array_equal(old["parameters"], mus) and np.array_equal(old["frequencies_hz"], freqs):
            return old["response"], {"reused_new_cache": True, **fingerprint(path)}
        raise RuntimeError(f"Existing new cache has different state/grid: {path}")
    values = []
    start = time.perf_counter()
    for i, (mu, f) in enumerate(points):
        values.append(family.solve(float(f), mu))
        print(f"direct point {i+1}/{len(points)} f={f:.6g} Hz", flush=True)
    response = np.asarray(values)
    np.savez_compressed(path, parameters=mus, frequencies_hz=freqs, response=response, elapsed_s=time.perf_counter()-start)
    return response, {"reused_new_cache": False, **fingerprint(path)}


def evaluate_case(case, args):
    out = args.out / case
    out.mkdir(parents=True, exist_ok=True)
    direct_dir = out / "direct_checks"
    direct_dir.mkdir(exist_ok=True)
    start = time.perf_counter()
    with deterministic_eigsh():
        metadata, system, operators, family, result, observation = load_parametric_case(CASE_DIRS[case], device="cpu")
    assert family.modal.n_modes == EXPECTED_POLES[case], (case, family.modal.n_modes)
    assert result.pod_basis.n_modes == 8
    assert len(result.members) == 5
    nn = FrozenNN(result)
    indices = observation_indices(system, operators)
    assert indices[0] == observation
    n = family.n_free
    displacement = np.flatnonzero(np.asarray(operators.free_dofs) < len(system.nodes))
    band = metadata["frequency_interval_hz"]
    assert band == [0.0, 200.0 if case == "flat" else 600.0]
    print(f"[{case}] production poles={family.modal.n_modes}, POD real rank=8, n_free={n}", flush=True)

    uq_path = CODE / f"results/ellipsoidal_uq/{case}/ellipsoidal_calibration_40x40.npz"
    uq = np.load(uq_path)
    f = np.asarray(uq["frequencies_hz"])
    mus = np.asarray(uq["test_parameters"])[:args.max_test_curves]
    amplitude = np.asarray(uq["test_hf_mm"])[:len(mus)] / 1e3
    assert f[0] == band[0] and f[-1] == band[1]
    assert amplitude.shape == (len(mus), len(f))
    assert np.max(radius(mus, family.bounds)) <= 1 + 1e-8
    # Calibration arrays are intentionally never read.

    full_path = CODE / ("results/phase7/flat_test_reference.npz" if case == "flat" else "results/phase7/stiffened_test_reference_600.npz")
    field_data = np.load(full_path)
    ff = np.asarray(field_data["frequencies_hz"])
    field_mu = np.asarray(field_data["test_parameters"])[:args.max_field_states]
    field_ref = np.asarray(field_data["test_reference_full"])[:len(field_mu)]
    assert field_ref.shape == (len(field_mu), len(ff), n)
    assert np.array_equal(ff, f)
    assert np.all(family.bounds.contains(field_mu))
    assert np.all(np.isfinite(field_ref))
    assert not np.any(np.all(np.isclose(mus[:, None, :], result.training_parameters[None, :, :], atol=1e-12, rtol=0), axis=-1))

    probe_ids = [0, int(np.argmax(np.abs(field_ref[0, :, observation]))), len(ff)-1]
    probes = [(field_mu[0], float(ff[j])) for j in probe_ids]
    probed, probe_manifest = direct_points(family, probes, direct_dir / "box_cache_identity.npz")
    probe_errors = [rel(probed[i], field_ref[0, j]) for i, j in enumerate(probe_ids)]
    write_json(out / "cache_identity_checks.json", {"frequencies_hz": [p[1] for p in probes], "relative_full_errors": probe_errors, "source": fingerprint(full_path), "new_probe_cache": probe_manifest})
    if max(probe_errors) > 1e-7:
        raise RuntimeError(f"{case}: old field cache does not match reconstructed direct model: {probe_errors}")

    eig_start = time.perf_counter()
    with deterministic_eigsh():
        wide = build_modal_resonance_interface(operators, ModalResonanceConfig(enabled=True, n_modes=40, frequency_min_hz=0.0, frequency_max_hz=None))
    eig_s = time.perf_counter()-eig_start
    np.savez_compressed(out / "reconstructed_eigenpairs.npz", production_eigenvalues=nn.poles.eigenvalues,
                        production_mode_shapes=nn.poles.mode_shapes, production_modal_forces=nn.poles.modal_forces,
                        wide_eigenvalues=wide.eigenvalues, wide_mode_shapes=wide.mode_shapes, wide_modal_forces=wide.modal_forces)
    static_start = time.perf_counter()
    static = spla.spsolve(operators.K.tocsc(), operators.rhs)
    static_s = time.perf_counter()-static_start
    methods = []
    for target in BUDGETS[case]:
        spectrum = nn.poles if target == EXPECTED_POLES[case] else wide
        k = complete_count(target, spectrum.natural_frequencies_hz)
        for corrected in (False, True):
            methods.append(Modal(spectrum, k, target, static, corrected,
                                 "shared_production" if spectrum is nn.poles else "shared_wide"))
    lspg_start = time.perf_counter()
    lspgs = [LSPG(nn, family, False), LSPG(nn, family, True)]
    lspg_s = time.perf_counter()-lspg_start
    methods.extend(lspgs + [nn])

    verify = []
    for ls in lspgs:
        for fj in (0.0, float(nn.poles.natural_frequencies_hz[0]), 0.9*band[1]):
            check = ls.explicit_check(fj, family.bounds.center)
            check["method"] = ls.name
            verify.append(check)
    # This tests the implementation against full rows, independently of caches.
    if max(x["background_response_relative_difference"] for x in verify) > 1e-4:
        write_json(out / "lspg_explicit_checks.json", verify)
        raise RuntimeError(f"{case}: compressed and explicit residual least squares disagree")
    library = np.mean(predict_parametric_observation_members(result, [0.0, 0.9*band[1]], family.bounds.center, observation), axis=0)[0]
    nn_difference = rel(nn.predict([0.0, 0.9*band[1]], family.bounds.center, [observation])[:, 0], library)
    assert nn_difference < 1e-12
    write_json(out / "lspg_explicit_checks.json", verify)

    rows, dips, field_rows, timing_rows = [], [], [], []
    observation_arrays = {"frequencies_hz": f, "parameters": mus, "reference_center_amplitude_m": amplitude}
    transfer = None
    sources = {"uq_amplitude": fingerprint(uq_path), "box_field": fingerprint(full_path)}
    if case == "flat":
        transfer_path = CODE / "results/equal_mode_dynamic_background_quarter/test_direct_observation_180_n40.npz"
        td = np.load(transfer_path)
        tm = np.asarray(td["parameters"])[:len(mus)]
        tf = np.asarray(td["frequencies_hz"])
        tr = np.asarray(td["response"])[:len(mus)]
        assert np.array_equal(tm, mus) and int(td["observation_index"]) == indices[1]
        assert np.all(td["completed"])
        transfer = (tf, tr)
        sources["transfer_complex"] = fingerprint(transfer_path)
        observation_arrays.update(transfer_frequencies_hz=tf, reference_transfer_complex_m=tr)

    for method in methods:
        print(f"[{case}] held-out curves: {method.name}", flush=True)
        preds = []
        for j, mu in enumerate(mus):
            pred = method.predict(f, mu, indices)
            preds.append(pred)
            rows.append({"case": case, "method": method.name, "protocol": "ellipsoid40_center_amplitude", "state": j,
                         "requested_modes": method.requested_count, "actual_modes": method.actual_count,
                         "ellipsoid_radius": radius(mu, family.bounds), **curve_metrics(pred[:, 0], amplitude[j], True)})
        observation_arrays[method.name] = np.asarray(preds)
        if transfer is not None:
            tf, tr = transfer
            tp = []
            for j, mu in enumerate(mus):
                pred = method.predict(tf, mu, [indices[1]])[:, 0]
                tp.append(pred)
                rows.append({"case": case, "method": method.name, "protocol": "ellipsoid40_transfer_complex_130_188", "state": j,
                             "requested_modes": method.requested_count, "actual_modes": method.actual_count, **curve_metrics(pred, tr[j])})
                dips.append({"case": case, "method": method.name, "protocol": "ellipsoid40_transfer_130_188", "state": j, **dip_comparison(tf, pred, tr[j], case)})
            observation_arrays[method.name+"__transfer"] = np.asarray(tp)
        print(f"[{case}] box interpolation full fields: {method.name}", flush=True)
        for j, mu in enumerate(field_mu):
            pred = method.predict(ff, mu)
            ref = field_ref[j]
            field_rows.append({"case": case, "method": method.name, "protocol": "box8_full_curve_field", "state": j,
                               "frequency_count": len(ff), "ellipsoid_radius": radius(mu, family.bounds), "inside_ellipsoid": bool(radius(mu, family.bounds) <= 1+1e-10),
                               "mixed_coordinate_complex_l2_percent": 100*rel(pred, ref),
                               "transverse_complex_l2_percent": 100*rel(pred[:, displacement], ref[:, displacement])})
            for cidx, label in enumerate(("center", "transfer")):
                rows.append({"case": case, "method": method.name, "protocol": "box8_"+label+"_complex", "state": j,
                             "ellipsoid_radius": radius(mu, family.bounds), **curve_metrics(pred[:, indices[cidx]], ref[:, indices[cidx]])})
            dips.append({"case": case, "method": method.name, "protocol": "box8_transfer", "state": j,
                         **dip_comparison(ff, pred[:, indices[1]], ref[:, indices[1]], case)})

    # Three predeclared ellipsoid states, four frequencies apiece.  These are
    # primary admissible field checks, distinct from the legacy box/LHS curves.
    primary_mu = mus[:args.primary_field_states]
    points, point_tags = [], []
    for j, mu in enumerate(primary_mu):
        shift = np.sqrt(mu[0]/mu[1])
        fs = (0.0, nn.poles.natural_frequencies_hz[0]*shift, (183.0 if case == "flat" else 432.0)*shift, band[1])
        for tag, fj in zip(("static", "scaled_first_eigenfrequency", "scaled_transfer_window", "band_upper"), fs):
            points.append((mu, float(fj)))
            point_tags.append((j, tag))
    primary_ref, primary_manifest = direct_points(family, points, direct_dir / "ellipsoid_primary_fields.npz")
    sources["primary_field_points"] = primary_manifest
    for method in methods:
        for j, ((mu, fj), (state, tag)) in enumerate(zip(points, point_tags)):
            pred = method.predict([fj], mu)[0]
            ref = primary_ref[j]
            field_rows.append({"case": case, "method": method.name, "protocol": "ellipsoid3_four_frequency_fields", "state": state,
                               "frequency_tag": tag, "frequency_hz": fj, "ellipsoid_radius": radius(mu, family.bounds), "inside_ellipsoid": True,
                               "mixed_coordinate_complex_l2_percent": 100*rel(pred, ref),
                               "transverse_complex_l2_percent": 100*rel(pred[displacement], ref[displacement])})

    for method in methods:
        for label, ix in (("one_center_observation_curve", [indices[0]]), ("complete_complex_field_curve", None)):
            measure = timing(lambda m=method, use=ix: m.predict(f, mus[0], use), args.timing_runs)
            timing_rows.append({"case": case, "method": method.name, "output": label, "device": "cpu", "threads": args.threads,
                                "frequency_count": len(f), "requested_modes": method.requested_count, "actual_modes": method.actual_count, **measure})
    np.savez_compressed(out / "heldout_observation_predictions.npz", **observation_arrays)
    write_csv(out / "curve_metrics.csv", rows)
    write_csv(out / "antiresonance_metrics.csv", dips)
    write_csv(out / "field_metrics.csv", field_rows)
    write_csv(out / "timings_cpu.csv", timing_rows)
    static_checks = []
    for method in methods:
        if isinstance(method, Modal) and method.corrected:
            z = method.predict([0.0], family.bounds.center)[0]
            eta = family.bounds.center[3]
            expected = static/(1+1j*eta)
            static_checks.append({"method": method.name, "relative_to_eq9_static": rel(z, expected),
                                  "relative_to_regularized_direct": rel(z, family.solve(0.0, family.bounds.center))})
    write_json(out / "static_limit_checks.json", static_checks)

    aggregates = []
    for protocol in sorted(set(x["protocol"] for x in rows)):
        for method in methods:
            selected = [x for x in rows if x["protocol"] == protocol and x["method"] == method.name]
            if not selected:
                continue
            a = {"case": case, "method": method.name, "protocol": protocol, "states": len(selected)}
            for metric in ("amplitude_l2_percent", "complex_l2_percent", "wrapped_phase_rmse_rad"):
                vals = [x[metric] for x in selected if metric in x]
                if vals:
                    a.update({metric+"_median": float(np.median(vals)), metric+"_p95": float(np.percentile(vals, 95)), metric+"_max": float(np.max(vals))})
            aggregates.append(a)
    write_csv(out / "curve_summary.csv", aggregates)
    dip_summary = []
    for protocol in sorted(set(x["protocol"] for x in dips)):
        for method in methods:
            ss = [x for x in dips if x["protocol"] == protocol and x["method"] == method.name]
            if ss:
                dip_summary.append({"method": method.name, "protocol": protocol, "states": len(ss),
                                    "reference_detected": sum(x["reference_detected"] for x in ss), "predicted_detected": sum(x["predicted_detected"] for x in ss),
                                    "missed": sum(x["missed"] for x in ss), "spurious": sum(x["spurious"] for x in ss)})
    write_csv(out / "antiresonance_summary.csv", dip_summary)
    manifest = {"case": case, "checkpoint": fingerprint(CASE_DIRS[case]/"parametric_operator.pt"), "metadata": fingerprint(CASE_DIRS[case]/"metadata.json"),
                "source_files": sources, "production_poles": family.modal.n_modes, "production_frequencies_hz": nn.poles.natural_frequencies_hz,
                "pole_provenance": "reconstructed from checkpoint configuration; original training eigenpairs were not saved, so bitwise recovery is not claimed",
                "pod_real_rank": nn.v.shape[1], "pod_basis_not_refitted": True, "additional_labels_for_baseline_construction": 0,
                "mean_complex_ensemble_not_mean_amplitude": True, "network_count": len(result.members), "network_decode": "ensemble mean coefficients decoded once; equivalent to mean complex fields",
                "neural_equivalence_relative_error": nn_difference, "observation_indices": indices, "observation_coordinates": [[0.5, 0.5], [7/30, 7/30]],
                "full_curve_box_state_parameters": field_mu, "full_curve_box_state_ellipsoid_radii": radius(field_mu, family.bounds),
                "primary_field_state_parameters": primary_mu, "primary_field_points": [{"state": s, "tag": tag, "f_hz": p[1], "mu": p[0]} for p, (s, tag) in zip(points, point_tags)],
                "cluster_completion": [{"requested": k, "actual": complete_count(k, wide.natural_frequencies_hz)} for k in BUDGETS[case]],
                "cluster_tolerance": "gap <= max(0.02 Hz, 2e-5 * larger frequency)", "wide_frequencies_hz": wide.natural_frequencies_hz,
                "modal_method_spectrum": [{"method": m.name, "spectrum_source": m.spectrum_source} for m in methods if isinstance(m, Modal)],
                "production_minus_wide_frequency_hz": nn.poles.natural_frequencies_hz - wide.natural_frequencies_hz[:nn.actual_count],
                "diagonal_regularization": operators.diagonal_regularization, "least_squares": {"driver": "scipy.linalg.lstsq gelsd", "relative_singular_cutoff": 1e-12, "column_scaling": "Euclidean norm at each query", "row_scaling": "raw and fixed W(0, mu_center), neither selected using test error"},
                "lspg": [{"method": ls.name, "qr_relative_error": ls.qr_relative_error, "minimum_rank": ls.minimum_rank, "maximum_column_scaled_condition": ls.maximum_condition,
                          "rank_failure_count": ls.rank_failure_count, "rank_failure_examples": ls.rank_failure_examples} for ls in lspgs],
                "preprocessing_s": {"shared_wide_40mode_eigensolve": eig_s, "shared_static_Ksolve": static_s, "both_lspg_operator_images_and_qr": lspg_s},
                "cost_limit": "frozen POD and neural training generation costs are inherited, not zero independent-deployment costs; timings are marginal CPU matched-output queries",
                "cache_identity_relative_errors": probe_errors, "elapsed_s": time.perf_counter()-start,
                "metric_definitions": {"l2": "unweighted Euclidean over stated frequency samples/DOFs, percent; complex and amplitude separate", "field": "original w/theta constrained coordinates; transverse-only also reported", "phase": "RMSE of wrapped angle(pred*conj(ref)); all evaluated frequencies", "dip": "strict/non-strict adjacent-point interior minimum; flat search 170-188, shoulders 130-188 Hz; stiff search/shoulders 350-500 Hz; deepest candidate; depth divided by larger endpoint amplitude"}}
    write_json(out / "manifest.json", manifest)
    write_json(out / "summary.json", {"curve_summary": aggregates, "antiresonance": dip_summary, "timings": timing_rows, "manifest": manifest})
    print(f"[{case}] finished in {manifest['elapsed_s']:.1f} s: {out}", flush=True)
    return {"case": case, "curve_summary": aggregates, "antiresonance": dip_summary, "elapsed_s": manifest["elapsed_s"]}


def main():
    args = args_parser()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    environment = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
                   "platform": platform.platform(), "threads": args.threads, "script": fingerprint(__file__), "arguments": vars(args),
                   "blas_thread_environment": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                   "cuda_available": torch.cuda.is_available(), "timing_device": "cpu", "no_retraining": True}
    write_json(args.out / "environment.json", environment)
    summaries = []
    for case in args.cases:
        summaries.append(evaluate_case(case, args))
    write_json(args.out / "eigsh_calls.json", EIGSH_CALLS)
    write_json(args.out / "summary_all.json", summaries)


if __name__ == "__main__":
    main()

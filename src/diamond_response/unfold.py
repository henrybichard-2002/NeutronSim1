"""Unfold a pulse-height spectrum into an incident neutron fluence spectrum and compare with the truth.

Model: counts per second ``y = M @ phi`` with ``M[dep, inc]`` the response matrix (cm^2) and ``phi`` the fluence
rate per incident-energy bin (n/cm^2/s). ``phi`` is recovered with MLEM (Poisson maximum likelihood, non-negative),
stopped early at the discrepancy ``chi2/ndf <= stop`` which acts as the regularisation. Uncertainties come from
repeating the unfolding on Poisson-resampled spectra with the response matrix perturbed by its own Monte-Carlo error.
They do not include the bias of the early stopping or any mismatch between the matrix and the real detector/field.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from .config import SimulationConfig, UnfoldConfig
from .response import ResponseMatrix, resolution_kernel

BANDS_MEV = ((1.0, 5.0), (5.0, 10.0), (10.0, 13.0), (13.0, 14.4), (14.4, 16.0))


@dataclasses.dataclass
class UnfoldResult:
    phi: np.ndarray            # fluence rate per incident bin, n/cm^2/s
    phi_err: np.ndarray        # standard deviation over the replicas
    iterations: int
    chi2_ndf: float
    refolded: np.ndarray       # M @ phi, counts/s


def mlem(matrix: np.ndarray, counts: np.ndarray, live_time_s: float, max_iterations: int, stop_chi2_ndf: float,
         start: np.ndarray | None = None):
    """MLEM for ``counts ~ Poisson(live_time * matrix @ phi)``. Returns (phi, iterations, chi2/ndf)."""
    a = matrix * live_time_s
    sens = a.sum(axis=0)
    ok = sens > 0
    phi = np.where(ok, 1.0, 0.0) if start is None else np.where(ok, start, 0.0)
    scale = counts.sum() / max((a @ phi).sum(), 1e-300)
    phi = phi * scale
    ndf = max(int((counts > 0).sum()) - int(ok.sum()), 1)
    chi2 = np.inf
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        model = a @ phi
        ratio = np.divide(counts, model, out=np.zeros_like(model), where=model > 0)
        phi = phi * np.divide(a.T @ ratio, sens, out=np.zeros_like(phi), where=ok)
        if iteration % 5 == 0 or iteration == max_iterations:
            model = a @ phi
            chi2 = float(((counts - model) ** 2 / np.maximum(model, 1.0)).sum()) / ndf
            if chi2 <= stop_chi2_ndf:
                break
    return phi, iteration, chi2


def unfold(matrix: ResponseMatrix, rate: np.ndarray, cfg: UnfoldConfig, rng: np.random.Generator | None = None,
           rate_err: np.ndarray | None = None, counts: np.ndarray | None = None,
           max_pulse_mev: float = np.inf) -> UnfoldResult:
    """Unfold a pulse-height spectrum ``rate`` (counts/s per deposited bin of ``matrix``).

    If ``counts`` (measured counts over ``cfg.live_time_s``) is given it is used directly and ``rate`` is ignored.
    Pulses outside [``cfg.min_pulse_mev``, ``max_pulse_mev``] are excluded from the fit.
    """
    rng = rng or np.random.default_rng(cfg.seed)
    use = (matrix.dep_centers >= cfg.min_pulse_mev) & (matrix.dep_centers <= max_pulse_mev)
    m = np.where(use[:, None], matrix.matrix, 0.0)
    if counts is not None:
        rate = counts / cfg.live_time_s
        counts = np.where(use, counts, 0.0)
    else:
        counts = rng.poisson(np.where(use, rate, 0.0) * cfg.live_time_s).astype(float)
    phi, iterations, chi2 = mlem(m, counts, cfg.live_time_s, cfg.max_iterations, cfg.stop_chi2_ndf)
    replicas = []
    for _ in range(max(cfg.replicas, 0)):
        m_r = np.clip(m + rng.normal(size=m.shape) * np.where(use[:, None], matrix.error, 0.0), 0.0, None)
        expected = np.clip(np.where(use, rate, 0.0), 0.0, None)
        c_r = rng.poisson(expected * cfg.live_time_s).astype(float)
        replicas.append(mlem(m_r, c_r, cfg.live_time_s, cfg.max_iterations, cfg.stop_chi2_ndf)[0])
    err = np.std(replicas, axis=0, ddof=1) if len(replicas) > 1 else np.zeros_like(phi)
    return UnfoldResult(phi, err, iterations, chi2, m @ phi)


def band_table(edges: np.ndarray, truth: np.ndarray, result: UnfoldResult, bands=BANDS_MEV) -> list[dict]:
    centers = 0.5 * (edges[1:] + edges[:-1])
    rows = []
    for lo, hi in bands:
        sel = (centers > lo) & (centers < hi)
        if not sel.any():
            continue
        t = float(truth[sel].sum())
        u = float(result.phi[sel].sum())
        e = float(np.sqrt((result.phi_err[sel] ** 2).sum()))
        rows.append({"band_MeV": [lo, hi], "true_n_cm2_s": t, "unfolded_n_cm2_s": u, "unfolded_error_n_cm2_s": e,
                     "ratio_unfolded_to_true": u / t if t > 0 else None})
    return rows


def run_unfolding(config: SimulationConfig, field, matrices: dict, outdir: Path, measured: np.ndarray | None = None,
                  measured_err: np.ndarray | None = None, log=print) -> list[Path]:
    """Unfold the detector's pulse-height spectrum and compare it with the actual incident neutron spectrum.

    ``measured`` (counts/s per deposited bin) defaults to the simulated detector response in the reactor field.
    Two response matrices are tried: the angle-averaged one (incidence distribution known) and the one of the
    class closest to normal incidence (what one would assume without knowing the angles).
    """
    from .reactor import effective_matrix, fold_field

    cfg = config.unfold
    res = config.resolution
    effective = effective_matrix(field, matrices)
    normal = matrices[min(matrices)]
    if cfg.with_resolution:
        effective = effective.with_resolution(res.noise_kev, res.relative)
        normal = normal.with_resolution(res.noise_kev, res.relative)
    if measured is None:
        _, measured, measured_err = fold_field(field, matrices)
        if cfg.with_resolution:
            g = resolution_kernel(effective.dep_edges, res.noise_kev, res.relative)
            measured = g @ measured
    edges = effective.inc_edges
    centers = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)
    truth = field.fluence_rate
    variants = {"effective (angle-averaged matrix)": effective, "normal-incidence matrix": normal}

    results = {name: unfold(m, measured, cfg, np.random.default_rng(cfg.seed)) for name, m in variants.items()}
    for name, r in results.items():
        log(f"Unfolding with {name}: {r.iterations} iterations, chi2/ndf = {r.chi2_ndf:.2f}")

    outdir.mkdir(parents=True, exist_ok=True)
    columns = [edges[:-1], edges[1:], centers, truth, field.fluence_rate_err]
    header = ["energy_low_MeV", "energy_high_MeV", "energy_center_MeV", "true_fluence_rate_n_cm2_s",
              "true_uncertainty_n_cm2_s"]
    for tag, (name, r) in zip(("effective", "normal"), results.items()):
        columns += [r.phi, r.phi_err, np.divide(r.phi, truth, out=np.zeros_like(truth), where=truth > 0)]
        header += [f"unfolded_{tag}_n_cm2_s", f"unfolded_{tag}_uncertainty_n_cm2_s", f"unfolded_{tag}_over_true"]
    csv_path = outdir / "unfolded_spectrum.csv"
    np.savetxt(csv_path, np.column_stack(columns), delimiter=",", header=",".join(header), comments="", fmt="%.8g")
    summary = {
        "live_time_s": cfg.live_time_s, "min_pulse_MeV": cfg.min_pulse_mev, "with_resolution": cfg.with_resolution,
        "stopping": f"first iteration (checked every 5) with chi2/ndf <= {cfg.stop_chi2_ndf}",
        "note": "Uncertainty = Poisson + response-matrix MC error over replicas; excludes unfolding bias.",
        "methods": {name: {"iterations": r.iterations, "chi2_per_ndf": r.chi2_ndf,
                           "total_unfolded_n_cm2_s": float(r.phi.sum()), "total_true_n_cm2_s": float(truth.sum()),
                           "bands": band_table(edges, truth, r)} for name, r in results.items()},
    }
    json_path = outdir / "unfolding_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    paths = [csv_path, json_path]
    try:
        paths += plot_unfolding(edges, width, truth, field.fluence_rate_err, results, effective.dep_edges, measured,
                                outdir / "plots")
    except ImportError:
        pass
    return paths


def plot_unfolding(edges, width, truth, truth_err, results, dep_edges, measured, plotdir: Path) -> list[Path]:
    from .plot import _plt, _save, grid

    plt = _plt()
    plotdir.mkdir(parents=True, exist_ok=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    paths = []

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    top.step(centers, truth / width, where="mid", color="k", lw=1.6, label="actual incident spectrum (reactor field)")
    top.fill_between(centers, (truth - truth_err) / width, (truth + truth_err) / width, step="mid", color="k",
                     alpha=0.2)
    for k, (name, r) in enumerate(results.items()):
        color = f"C{k}"
        top.step(centers, r.phi / width, where="mid", color=color, lw=1.1, label=f"unfolded, {name}")
        top.fill_between(centers, (r.phi - r.phi_err) / width, (r.phi + r.phi_err) / width, step="mid", color=color,
                         alpha=0.3)
        ratio = np.divide(r.phi, truth, out=np.full_like(truth, np.nan), where=truth > 0)
        rerr = np.divide(r.phi_err, truth, out=np.zeros_like(truth), where=truth > 0)
        bottom.errorbar(centers, ratio, yerr=rerr, fmt=".", ms=3, lw=0.6, color=color)
    top.set_ylabel("fluence rate per MeV [n/cm$^2$/s/MeV]  (bands: 1 sigma)")
    top.set_title("Unfolded vs actual incident neutron spectrum")
    top.set_ylim(bottom=0)
    grid(top)
    top.legend(fontsize=8)
    bottom.axhline(1.0, color="k", lw=0.8)
    bottom.set_ylim(0, 2)
    bottom.set_xlabel("incident neutron energy [MeV]")
    bottom.set_ylabel("unfolded / actual")
    grid(bottom)
    paths.append(_save(fig, plotdir / "unfolded_vs_true.png"))

    fig, ax = plt.subplots(figsize=(10, 5))
    dep_centers = 0.5 * (dep_edges[1:] + dep_edges[:-1])
    dw = np.diff(dep_edges)
    ax.step(dep_centers, measured / dw, where="mid", color="k", lw=1.4, label="detector spectrum (input)")
    for k, (name, r) in enumerate(results.items()):
        ax.step(dep_centers, r.refolded / dw, where="mid", color=f"C{k}", lw=1, label=f"refolded unfolded, {name}")
    ax.set_yscale("log")
    top_val = float(np.max(measured / dw)) if np.any(measured > 0) else 1.0
    ax.set_ylim(top_val * 1e-6, top_val * 3)
    ax.set_xlabel("pulse height [MeV]")
    ax.set_ylabel("counts per second per MeV")
    ax.set_title("Check: unfolded spectrum folded back through the response")
    grid(ax)
    ax.legend(fontsize=8)
    paths.append(_save(fig, plotdir / "unfold_refold_check.png"))
    return paths

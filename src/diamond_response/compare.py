"""Compare measured pulse-height spectra with the simulation, and unfold both.

Inputs: a saved reactor field at the measurement position (true incident neutron spectrum), a response matrix
(``response_matrix.npz``) and one or more measured spectra (``all_datasets.npz`` + ``metadata.json`` folders).
Everything is plotted as *relative flux*: each curve is normalised to unit area over its comparison range.

Simulated deposited spectrum = field folded through the (resolution-smeared) response matrix. The matrix is for normal
incidence, so the angular distribution of the field is not accounted for. Both the measured and the simulated spectra
are unfolded with the same matrix, MLEM settings and statistics (the simulated one is scaled to the measured counts).
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import numpy as np

from .config import SimulationConfig
from .response import ResponseMatrix, resolution_kernel
from .unfold import unfold


def resolve_path(path: str | Path) -> Path:
    """Accept Windows paths (``C:\\...``) when running under WSL."""
    text = str(path)
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if m:
        return Path("/mnt") / m.group(1).lower() / m.group(2).replace("\\", "/")
    return Path(text)


@dataclasses.dataclass
class Measured:
    label: str
    counts: np.ndarray      # per deposited bin of the matrix
    live_time_s: float
    max_mev: float          # highest pulse height covered by the data
    notes: str


DEFAULT_MV_PER_MEV = 100.0


def load_measured(path: str | Path, dep_edges: np.ndarray, mv_per_mev: float = DEFAULT_MV_PER_MEV,
                  assumed_mv_per_mev: float = DEFAULT_MV_PER_MEV) -> Measured:
    """Rebin a measured spectrum; the stored energy axis was made with ``assumed_mv_per_mev`` and is
    rescaled to the true ``mv_per_mev`` (energy = stored energy * assumed / true)."""
    folder = resolve_path(path)
    data_file = folder / "all_datasets.npz" if folder.is_dir() else folder
    if not data_file.exists():
        raise FileNotFoundError(f"measured spectrum not found: {data_file}")
    meta_file = data_file.parent / "metadata.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    with np.load(data_file) as z:
        energy = np.asarray(z["channel a_spectrum_energy [MeV]"], float)
        counts = np.asarray(z["channel a_spectrum_counts"], float)
    energy = energy * (assumed_mv_per_mev / mv_per_mev)
    # the last ADC bin collects overflow
    energy, counts = energy[:-1], counts[:-1]
    index = np.searchsorted(dep_edges, energy, side="right") - 1
    keep = (index >= 0) & (index < len(dep_edges) - 1)
    binned = np.bincount(index[keep], weights=counts[keep], minlength=len(dep_edges) - 1)
    duration = float(meta.get("duration [s]", 0.0))
    if duration <= 0:
        raise ValueError(f"{data_file.parent}: metadata.json has no duration")
    return Measured(data_file.parent.name, binned, duration, float(energy[-1]), str(meta.get("rosy notes", "")))


def _normalise(values: np.ndarray, errors: np.ndarray, widths: np.ndarray, selected: np.ndarray):
    """Scale to unit area over ``selected``; returns (density, density error) per MeV."""
    area = float((values[selected]).sum())
    if area <= 0:
        return np.zeros_like(values), np.zeros_like(values)
    return values / area / widths, errors / area / widths


def run_comparison(config: SimulationConfig, field, matrix: ResponseMatrix, measured: list[Measured], outdir: Path,
                   noise_kev: float | None = None, relative: float | None = None, log=print) -> list[Path]:
    cfg = config.unfold
    noise = config.resolution.noise_kev if noise_kev is None else noise_kev
    rel = config.resolution.relative if relative is None else relative
    if matrix.inc_edges.shape != field.inc_edges.shape or not np.allclose(matrix.inc_edges, field.inc_edges):
        raise ValueError("response-matrix incident grid differs from the reactor field grid; "
                         "use a field and matrix with the same incident binning")
    smeared = matrix.with_resolution(noise, rel)
    dep_edges = smeared.dep_edges
    dep_centers = smeared.dep_centers
    dep_width = np.diff(dep_edges)
    inc_edges = smeared.inc_edges
    inc_width = np.diff(inc_edges)
    inc_centers = smeared.inc_centers
    truth = field.fluence_rate

    sim_rate = smeared.matrix @ truth
    sim_err = np.sqrt((smeared.matrix**2) @ field.fluence_rate_err**2 + (smeared.error**2) @ truth**2)

    outdir.mkdir(parents=True, exist_ok=True)
    plotdir = outdir / "plots"
    summary = {"noise_kev": noise, "relative_resolution": rel, "min_pulse_MeV": cfg.min_pulse_mev, "datasets": {}}
    true_n, true_ne = _normalise(truth, field.fluence_rate_err, inc_width, np.ones_like(truth, bool))

    dep_cols = [dep_edges[:-1], dep_edges[1:], dep_centers]
    dep_head = ["deposited_low_MeV", "deposited_high_MeV", "deposited_center_MeV"]
    inc_cols = [inc_edges[:-1], inc_edges[1:], inc_centers, true_n, true_ne]
    inc_head = ["incident_low_MeV", "incident_high_MeV", "incident_center_MeV", "true_sim_relative_flux_per_MeV",
                "true_sim_uncertainty"]
    curves = []
    sim_cols_added = False
    for k, meas in enumerate(measured):
        sel = (dep_centers >= cfg.min_pulse_mev) & (dep_centers <= meas.max_mev)
        m_n, m_ne = _normalise(meas.counts, np.sqrt(meas.counts), dep_width, sel)
        scale = meas.counts[sel].sum() / max(sim_rate[sel].sum() * meas.live_time_s, 1e-300)
        s_n, s_ne = _normalise(sim_rate, sim_err, dep_width, sel)
        label = meas.label
        dep_cols += [m_n, m_ne] + ([s_n, s_ne] if not sim_cols_added else [])
        dep_head += [f"measured_{label}_relative_per_MeV", f"measured_{label}_uncertainty"]
        if not sim_cols_added:
            dep_head += ["simulated_relative_per_MeV", "simulated_uncertainty"]
            sim_cols_added = True

        ucfg = dataclasses.replace(cfg, live_time_s=meas.live_time_s)
        r_meas = unfold(smeared, None, ucfg, np.random.default_rng(cfg.seed), counts=meas.counts,
                        max_pulse_mev=meas.max_mev)
        r_sim = unfold(smeared, sim_rate * scale, ucfg, np.random.default_rng(cfg.seed + 1), max_pulse_mev=meas.max_mev)
        log(f"{label}: unfolded measured ({r_meas.iterations} it, chi2/ndf {r_meas.chi2_ndf:.2f}), "
            f"simulated ({r_sim.iterations} it, chi2/ndf {r_sim.chi2_ndf:.2f})")
        um, ume = _normalise(r_meas.phi, r_meas.phi_err, inc_width, np.ones_like(truth, bool))
        us, use_ = _normalise(r_sim.phi, r_sim.phi_err, inc_width, np.ones_like(truth, bool))
        inc_cols += [um, ume, us, use_]
        inc_head += [f"unfolded_measured_{label}_relative_per_MeV", f"unfolded_measured_{label}_uncertainty",
                     f"unfolded_simulated_for_{label}_relative_per_MeV", f"unfolded_simulated_for_{label}_uncertainty"]
        in_range = sel.copy()
        summary["datasets"][label] = {
            "notes": meas.notes, "live_time_s": meas.live_time_s,
            "measured_counts_per_s_in_range": float(meas.counts[in_range].sum() / meas.live_time_s),
            "simulated_counts_per_s_in_range": float(sim_rate[in_range].sum()),
            "simulated_over_measured_rate": float(1.0 / scale),
            "range_MeV": [cfg.min_pulse_mev, meas.max_mev],
            "iterations": {"measured": r_meas.iterations, "simulated": r_sim.iterations},
            "chi2_per_ndf": {"measured": r_meas.chi2_ndf, "simulated": r_sim.chi2_ndf},
        }
        curves.append((label, (m_n, m_ne), (s_n, s_ne), (um, ume), (us, use_), sel))

    paths = []
    for name, cols, head in (("compare_deposited_spectra.csv", dep_cols, dep_head),
                             ("compare_unfolded_spectra.csv", inc_cols, inc_head)):
        path = outdir / name
        np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(head), comments="", fmt="%.8g")
        paths.append(path)
    summary_path = outdir / "compare_summary.json"
    summary["note"] = ("Relative flux = unit area over the comparison range. Simulated spectrum uses the normal-incidence "
                       "matrix; the unfolded uncertainty excludes unfolding bias.")
    summary_path.write_text(json.dumps(summary, indent=2))
    paths.append(summary_path)
    paths += plot_comparison(curves, dep_edges, inc_edges, (true_n, true_ne), plotdir)
    return paths


def _inset(ax, series):
    inset = ax.inset_axes([0.58, 0.35, 0.39, 0.55])
    top = 0.0
    for x, y, kw in series:
        inset.step(x, np.where(y > 0, y, np.nan), where="mid", **kw)
        if np.any(y > 0):
            top = max(top, float(np.nanmax(y)))
    inset.set_yscale("log")
    if top > 0:
        inset.set_ylim(top * 1e-4, top * 3)
    inset.set_title("log scale", fontsize=8)
    inset.tick_params(labelsize=7)
    from .plot import grid

    grid(inset)


def plot_comparison(curves, dep_edges, inc_edges, truth, plotdir: Path) -> list[Path]:
    from .plot import _plt, _save, grid

    plt = _plt()
    plotdir.mkdir(parents=True, exist_ok=True)
    dc = 0.5 * (dep_edges[1:] + dep_edges[:-1])
    ic = 0.5 * (inc_edges[1:] + inc_edges[:-1])
    paths = []

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    series = []
    for k, (label, (m, me), (s, se), _, _, sel) in enumerate(curves):
        top.step(dc[sel], m[sel], where="mid", color=f"C{k + 1}", lw=1.1, label=f"measured {label}")
        top.fill_between(dc[sel], (m - me)[sel], (m + me)[sel], step="mid", color=f"C{k + 1}", alpha=0.3)
        series.append((dc[sel], m[sel], {"color": f"C{k + 1}", "lw": 0.8}))
        ratio = np.divide(s, m, out=np.full_like(s, np.nan), where=m > 0)
        bottom.plot(dc[sel], ratio[sel], ".", ms=2, color=f"C{k + 1}")
    first_sel = curves[0][5]
    s, se = curves[0][2]
    top.step(dc[first_sel], s[first_sel], where="mid", color="k", lw=1.3, label="simulated (field x response)")
    top.fill_between(dc[first_sel], (s - se)[first_sel], (s + se)[first_sel], step="mid", color="k", alpha=0.25)
    series.append((dc[first_sel], s[first_sel], {"color": "k", "lw": 0.8}))
    _inset(top, series)
    top.set_ylim(bottom=0)
    top.set_ylabel("relative flux [1/MeV] (unit area)")
    top.set_title("Deposited-energy spectrum: measured vs simulated")
    grid(top)
    top.legend(fontsize=8, loc="upper left")
    bottom.axhline(1.0, color="k", lw=0.8)
    bottom.set_ylim(0, 3)
    bottom.set_xlabel("deposited energy [MeV]")
    bottom.set_ylabel("simulated / measured")
    grid(bottom)
    paths.append(_save(fig, plotdir / "compare_deposited.png"))

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    t, te = truth
    top.step(ic, t, where="mid", color="k", lw=1.6, label="actual simulated incident spectrum")
    top.fill_between(ic, t - te, t + te, step="mid", color="k", alpha=0.2)
    series = [(ic, t, {"color": "k", "lw": 0.8})]
    for k, (label, _, _, (um, ume), (us, use_), _) in enumerate(curves):
        top.step(ic, um, where="mid", color=f"C{k + 1}", lw=1.1, label=f"unfolded measured {label}")
        top.fill_between(ic, um - ume, um + ume, step="mid", color=f"C{k + 1}", alpha=0.3)
        top.step(ic, us, where="mid", color=f"C{k + 4}", lw=1.1, ls="--", label=f"unfolded simulated (same statistics as {label})")
        top.fill_between(ic, us - use_, us + use_, step="mid", color=f"C{k + 4}", alpha=0.25)
        series += [(ic, um, {"color": f"C{k + 1}", "lw": 0.8}), (ic, us, {"color": f"C{k + 4}", "lw": 0.8})]
        for y, ye, c in ((um, ume, f"C{k + 1}"), (us, use_, f"C{k + 4}")):
            ratio = np.divide(y, t, out=np.full_like(t, np.nan), where=t > 0)
            bottom.errorbar(ic, ratio, yerr=np.divide(ye, t, out=np.zeros_like(t), where=t > 0), fmt=".", ms=2, lw=0.5,
                            color=c)
    _inset(top, series)
    top.set_ylim(bottom=0)
    top.set_ylabel("relative flux [1/MeV] (unit area, 1-16 MeV)")
    top.set_title("Unfolded spectra vs actual simulated incident spectrum")
    grid(top)
    top.legend(fontsize=8, loc="upper left")
    bottom.axhline(1.0, color="k", lw=0.8)
    bottom.set_ylim(0, 3)
    bottom.set_xlabel("incident neutron energy [MeV]")
    bottom.set_ylabel("unfolded / actual")
    grid(bottom)
    paths.append(_save(fig, plotdir / "compare_unfolded.png"))
    return paths

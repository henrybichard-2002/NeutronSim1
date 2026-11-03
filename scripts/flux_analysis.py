#!/usr/bin/env python3
"""Neutron flux from a measured diamond pulse-height run (freestanding; needs numpy, scipy, matplotlib).

Inputs
  --data         folder with all_datasets.npz + metadata.json (the acquisition output)
  --matrix       response_matrix.npz (pulses per unit fluence, cm^2; deposited x incident)
  --field        reactor_field.npz of the measurement position (gives detector -> reactor-surface scaling)
  --efficiency   optional efficiency table (CSV: first column incident energy [MeV], header = PH thresholds [MeV],
                 values = efficiency [cm^2]); by default it is computed from the (resolution-smeared) matrix.
                 The table actually used is always written to efficiency_matrix.csv.

Method
  1. The pulse-height spectrum is unfolded (MLEM) with the smeared response matrix, which gives the spectral shape
     of the incident neutrons (absolute, n/cm^2/s at the detector) and, by replicas that vary the counts (Poisson)
     and the matrix (its statistical error), its variance.
  2. Pulse-height window efficiency for that spectrum: eff = sum_j w_j * eps_j(window), w_j = unfolded shape.
  3. Flux at the detector = pulse rate in the window / eff.  The time series uses the acquisition's per-second
     counters (``total`` = above the trigger, ``roi`` = hardware ROI window).
  4. Flux at the reactor surface = detector flux * surface_flux / detector_flux of the simulated field
     (one energy-independent factor, k).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_MV_PER_MEV = 100.0


def resolve_path(text: str) -> Path:
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if m and not Path(text).exists():
        return Path("/mnt") / m.group(1).lower() / m.group(2).replace("\\", "/")
    return Path(text)


def smearing_kernel(edges: np.ndarray, noise_kev: float, relative: float) -> np.ndarray:
    from scipy.special import ndtr

    centers = 0.5 * (edges[1:] + edges[:-1])
    fwhm = np.sqrt((noise_kev * 1e-3) ** 2 + (relative * centers) ** 2)
    sigma = np.maximum(fwhm / 2.354820045, 1e-9)
    return (ndtr((edges[1:, None] - centers[None, :]) / sigma[None, :])
            - ndtr((edges[:-1, None] - centers[None, :]) / sigma[None, :]))


def mlem(matrix: np.ndarray, counts: np.ndarray, live_time_s: float, iterations: int) -> np.ndarray:
    a = matrix * live_time_s
    sens = a.sum(axis=0)
    ok = sens > 0
    phi = np.where(ok, 1.0, 0.0)
    phi *= counts.sum() / max((a @ phi).sum(), 1e-300)
    for _ in range(iterations):
        model = a @ phi
        ratio = np.divide(counts, model, out=np.zeros_like(model), where=model > 0)
        phi = phi * np.divide(a.T @ ratio, sens, out=np.zeros_like(phi), where=ok)
    return phi


class Efficiency:
    """Efficiency (cm^2) per incident bin for a pulse-height window [lo, hi)."""

    def __init__(self, matrix: np.ndarray, dep_centers: np.ndarray, table: tuple | None = None):
        self.matrix, self.dep, self.table = matrix, dep_centers, table

    def window(self, lo: float, hi: float = np.inf, matrix: np.ndarray | None = None) -> np.ndarray:
        if self.table is not None:
            thresholds, values = self.table
            def above(t):
                if not np.isfinite(t):
                    return np.zeros(values.shape[1])
                return np.array([np.interp(t, thresholds, values[:, j]) for j in range(values.shape[1])])
            return above(lo) - above(hi)
        m = self.matrix if matrix is None else matrix
        return m[(self.dep >= lo) & (self.dep < hi)].sum(axis=0)


def load_efficiency_table(path: Path, inc_centers: np.ndarray):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    thresholds = np.array([float(x) for x in rows[0][1:]])
    body = np.array([[float(x) for x in r] for r in rows[1:]])
    if body.shape[0] != len(inc_centers) or not np.allclose(body[:, 0], inc_centers, atol=1e-6):
        raise ValueError("efficiency table incident energies differ from the response matrix grid")
    return thresholds, body[:, 1:].T          # [threshold, incident]


def write_efficiency_csv(path: Path, inc_centers, thresholds, eff) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["incident_MeV"] + [f"{t:g}" for t in thresholds])
        for j, e in enumerate(inc_centers):
            w.writerow([f"{e:.6f}"] + [f"{eff[k][j]:.6e}" for k in range(len(thresholds))])


def load_fields(args, inc_edges: np.ndarray):
    """[(distance_cm, surface_flux, record)] for every saved field on the matrix incident grid."""
    paths = [resolve_path(args.field)] if args.field else sorted(Path(args.field_dir).glob("*/npz/reactor_field.npz"))
    found = []
    for path in paths:
        z = np.load(path)
        if z["inc_edges"].shape != inc_edges.shape or not np.allclose(z["inc_edges"], inc_edges):
            continue
        surface = float(z["surface_flux_n_cm2_s"]) if "surface_flux_n_cm2_s" in z.files else args.surface_flux
        rec = {"path": str(path), "rate": np.asarray(z["fluence_rate"], float), "err": np.asarray(z["fluence_rate_err"], float),
               "extra": float(z["fluence_below_grid"]) + float(z["fluence_above_grid"]), "surface": surface}
        found.append((float(z["detector_x_cm"]) - args.surface_radius_cm, rec))
    return sorted(found, key=lambda item: item[0])


def field_at(args, distance, inc_edges):
    found = load_fields(args, inc_edges)
    if not found:
        raise SystemExit("no reactor_field.npz on the response-matrix incident grid found; use --field/--field-dir "
                         "or --surface-scale")
    if distance is None:
        if len(found) > 1 and not args.field:
            raise SystemExit("several fields found; give --distance-cm")
        distance = found[0][0]
    dists = np.array([d for d, _ in found])
    if len(found) == 1 or np.min(np.abs(dists - distance)) < 5e-3:
        d, rec = found[int(np.argmin(np.abs(dists - distance)))]
        if abs(d - distance) > 5e-3:
            raise SystemExit(f"only a field at {d:.2f} cm exists; cannot use it for {distance:.2f} cm "
                             "(run 'diamond-response compare' for that distance)")
        mix = [(1.0, rec)]
        description = f"{rec['path']} (simulated at {d:.2f} cm)"
    else:
        if not dists.min() <= distance <= dists.max():
            raise SystemExit(f"{distance:.2f} cm is outside the simulated range "
                             f"{dists.min():.2f}-{dists.max():.2f} cm; simulate that distance first")
        hi = int(np.searchsorted(dists, distance))
        (d0, r0), (d1, r1) = found[hi - 1], found[hi]
        w = (distance - d0) / (d1 - d0)
        mix = [(1 - w, r0), (w, r1)]
        description = (f"interpolated linearly between {d0:.2f} cm ({r0['path']}) and {d1:.2f} cm ({r1['path']}); "
                       "exact only if you simulate this distance")
        print("WARNING: no field simulated at this distance - interpolating")
    rate = sum(w * r["rate"] for w, r in mix)
    err = np.sqrt(sum((w * r["err"]) ** 2 for w, r in mix))
    total = rate.sum() + sum(w * r["extra"] for w, r in mix)
    surface = sum(w * r["surface"] for w, r in mix)
    return {"rate": rate, "err": err, "k": surface / total, "distance": distance, "description": description}


def grid(ax) -> None:
    ax.minorticks_on()
    ax.grid(which="major", alpha=0.6, linewidth=0.8)
    ax.grid(which="minor", alpha=0.25, linewidth=0.4)


def diagnostics(v: dict) -> None:
    """Plots that check each step of the chain; ``v`` is the namespace of main()."""
    import matplotlib.pyplot as plt

    out = Path(v["out"]) / "plots" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    args, mv, k, field = v["args"], v["mv"], v["k"], v["field"]
    dep_c, dep_edges, inc_c, inc_w = v["dep_c"], v["dep_edges"], v["inc_c"], v["inc_w"]
    counts, fit, duration, phi0, matrix = v["counts"], v["fit"], v["duration"], v["phi0"], v["matrix"]
    eff, e_max, win_series, win_thr = v["eff"], v["e_max"], v["win_series"], v["win_thr"]
    reachable, flux_t, flux_t_err, t_mid = v["reachable"], v["flux_t"], v["flux_t_err"], v["t_mid"]
    dep_w = np.diff(dep_edges)
    hi_s = win_series[1] if np.isfinite(win_series[1]) else e_max
    label = f"{win_series[0]:.2f}-{hi_s:.2f} MeV"
    saved = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=140)
        plt.close(fig)
        saved.append(name)

    # 1. raw ADC spectrum with trigger / ROI / fit limits
    amp, c_adc = v["amp_mv"][:-1], v["counts_adc"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(amp, c_adc, where="mid", lw=0.7)
    ax.set_yscale("log")
    for x, text, col in ((v["meta"]["channel a_trigger [mV]"], "trigger", "r"),
                         (v["meta"]["channel a_roi [mV]"][0], "ROI low", "g"),
                         (v["meta"]["channel a_roi [mV]"][1], "ROI high", "g"),
                         (args.min_pulse_mev * mv, "fit lower limit", "m")):
        ax.axvline(x, color=col, ls="--", lw=1, label=f"{text} ({x:.0f} mV = {x / mv:.2f} MeV)")
    ax.set(xlabel="pulse amplitude [mV]", ylabel="counts per ADC bin",
           title=f"Raw acquisition spectrum ({int(c_adc.sum()):,} counts; last/overflow bin removed)")
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / mv, lambda x: x * mv))
    sec.set_xlabel(f"pulse height [MeV] at {mv:g} mV/MeV")
    ax.legend(); grid(ax)
    save(fig, "d1_raw_adc_spectrum.png")

    # 2. refold check: does the unfolded spectrum reproduce the measurement?
    model = matrix @ phi0 * duration
    mask = fit & (model > 0)
    pull = np.where(mask, (counts - model) / np.sqrt(np.maximum(model, 1.0)), np.nan)
    chi2 = float(np.nansum(pull ** 2) / max(mask.sum(), 1))
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a.step(dep_c[fit], counts[fit] / duration / dep_w[fit], where="mid", label="measured", color="C0")
    a.step(dep_c[fit], model[fit] / duration / dep_w[fit], where="mid", label="refolded unfolded spectrum", color="C3")
    a.set(ylabel="counts/s/MeV", yscale="log", title=f"Refold check: chi2/ndf = {chi2:.1f} (bins above {args.min_pulse_mev:g} MeV)")
    a.legend(); grid(a)
    b.plot(dep_c[fit], pull[fit], ".", ms=3)
    b.axhline(0, color="k", lw=1)
    b.set(xlabel="deposited energy [MeV]", ylabel="(data - model)/sigma", ylim=(-10, 10)); grid(b)
    save(fig, "d2_refold_check.png")

    # 3. efficiency of the windows and which incident energies dominate the rate
    shapes = {"unfolded spectrum": phi0 / phi0.sum()}
    if field is not None:
        sim = np.where(reachable, field["rate"], 0.0)
        shapes["simulated field"] = sim / sim.sum()
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for lo, hi, text, col in ((win_series[0], win_series[1], f"series window {label}", "C0"),
                              (win_thr[0], win_thr[1], f"threshold window > {win_thr[0]:.2f} MeV", "C1"),
                              (args.min_pulse_mev, np.inf, f"all > {args.min_pulse_mev:g} MeV", "C2")):
        a.plot(inc_c, eff.window(lo, hi), color=col, label=text)
    a.set(ylabel="efficiency [cm$^2$]", title="Window efficiency vs incident energy"); a.legend(); grid(a)
    e_ser = eff.window(*win_series)
    for (name, w), col in zip(shapes.items(), ("C3", "k")):
        contrib = np.cumsum(w * e_ser) / (w @ e_ser)
        b.plot(inc_c, contrib, color=col, label=f"{name} (effective efficiency {w @ e_ser:.3g} cm$^2$)")
    b.set(xlabel="incident neutron energy [MeV]", ylabel="cumulative fraction of window pulses",
          title="Which incident energies produce the pulses in the series window"); b.legend(); grid(b)
    save(fig, "d3_efficiency_and_contribution.png")

    # 4. flux vs PH threshold (flat if the spectrum shape and efficiency are right)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ths = np.arange(args.min_pulse_mev, min(e_max - 0.5, 9.5) + 1e-9, 0.25)
    for (name, w), col in zip(shapes.items(), ("C3", "k")):
        f, fe = [], []
        for t in ths:
            n = counts[(dep_c >= t) & (dep_c < e_max)].sum()
            e = float(w @ eff.window(t, e_max))
            f.append(n / duration / e * k if e > 0 else np.nan)
            fe.append(np.sqrt(max(n, 1.0)) / duration / e * k if e > 0 else np.nan)
        ax.errorbar(ths, f, yerr=fe, fmt=".-", lw=0.8, ms=4, color=col, label=f"efficiency weighted with {name}")
    ax.axvline(win_thr[0], color="C1", ls="--", label=f"chosen threshold {win_thr[0]:.2f} MeV")
    ax.axvline(win_series[0], color="C0", ls=":", label=f"series window start {win_series[0]:.2f} MeV")
    ax.set(xlabel="pulse-height threshold [MeV]", ylabel="flux at reactor surface [n/cm$^2$/s]",
           title="Consistency: flux from pulses above each threshold (should be flat)")
    ymax = max((np.nanmedian(ln.get_ydata()) for ln in ax.get_lines() if len(ln.get_xdata()) > 2), default=1.0)
    ax.set_ylim(0, 2.5 * ymax)
    ax.legend(); grid(ax)
    save(fig, "d4_flux_vs_threshold.png")

    # 5. sensitivity of the series flux to the analysis choices
    raw_energy = v["raw_energy"]
    z_matrix, z_error = v["z_matrix"], v["z_error"]
    roi_mv = v["meta"]["channel a_roi [mV]"]
    trig_mv = v["meta"]["channel a_trigger [mV]"]

    def estimate(mv_, noise_, iters_):
        e = raw_energy[:-1] * args.assumed_mv_per_mev / mv_
        idx = np.searchsorted(dep_edges, e, side="right") - 1
        ok = (idx >= 0) & (idx < len(dep_edges) - 1)
        c = np.bincount(idx[ok], weights=v["counts_adc"][ok], minlength=len(dep_edges) - 1)
        g = smearing_kernel(dep_edges, noise_, args.relative_resolution)
        m = g @ z_matrix
        f_ = (dep_c >= args.min_pulse_mev) & (dep_c <= e[-1])
        mm = np.where(f_[:, None] & reachable[None, :], m, 0.0)
        phi = mlem(mm, np.where(f_, c, 0.0), duration, iters_)
        ef = Efficiency(m, dep_c, v["eff"].table)
        lo, hi = ((roi_mv[0] / mv_, roi_mv[1] / mv_) if args.series == "roi" else (trig_mv / mv_, np.inf))
        n = v["series_total"]
        return n / len(v["t_s"]) / float((phi / phi.sum()) @ ef.window(lo, hi))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    base = estimate(mv, args.noise_kev, args.iterations) * k
    for ax, values, fn, xl in (
            (axes[0], np.unique(np.round(np.r_[mv * np.array([0.9, 0.93, 0.96, 0.98, 1.0, 1.02, 1.04, 1.07, 1.1])], 3)),
             lambda x: estimate(x, args.noise_kev, args.iterations), "detector scaling [mV/MeV]"),
            (axes[1], np.array([0.0, 25, 50, 75, 100, 150, 200, 300]),
             lambda x: estimate(mv, x, args.iterations), "resolution noise [keV]"),
            (axes[2], np.array([10, 25, 50, 100, 150, 300, 600]),
             lambda x: estimate(mv, args.noise_kev, int(x)), "MLEM iterations")):
        ax.plot(values, [fn(x) * k for x in values], "o-")
        ax.axhline(base, color="k", lw=0.8, ls="--")
        ax.set(xlabel=xl, ylabel="surface flux [n/cm$^2$/s]"); grid(ax)
    axes[0].set_title(f"Series-window flux vs analysis choices ({args.series} counter)")
    save(fig, "d5_sensitivity.png")

    # 6. time-series diagnostics
    t_s, series = v["t_s"], v["series"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    a = axes[0, 0]
    kern = np.ones(60) / 60
    for name, col in (("total", "C0"), ("roi", "C1")):
        a.plot(t_s[29:-30] / 60, np.convolve(series[name], kern, "valid"), color=col, label=f"{name} counts/s (60 s mean)")
    a.set(xlabel="runtime [min]", ylabel="counts/s", title="Raw count rates", yscale="log"); a.legend(); grid(a)
    b = axes[0, 1]
    mean_f = np.average(flux_t, weights=1 / flux_t_err ** 2)
    zs = (flux_t - mean_f) / flux_t_err
    b.hist(zs, bins=30, density=True, alpha=0.6, label=f"pulls (std {zs.std(ddof=1):.2f})")
    xs = np.linspace(-4, 4, 200)
    b.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "k", label="N(0,1)")
    b.set(xlabel="(flux - mean)/stat. error", ylabel="density", title="Are fluctuations purely statistical?"); b.legend(); grid(b)
    c = axes[1, 0]
    cum = np.cumsum(series[args.series]) / np.arange(1, len(t_s) + 1) / v["eff_ser0"] * k
    sig = np.sqrt(np.cumsum(series[args.series])) / np.arange(1, len(t_s) + 1) / v["eff_ser0"] * k
    c.plot(t_s / 60, cum, color="C0"); c.fill_between(t_s / 60, cum - sig, cum + sig, alpha=0.3)
    c.set(xlabel="runtime [min]", ylabel="cumulative mean surface flux [n/cm$^2$/s]", title="Convergence of the run mean")
    grid(c)
    d = axes[1, 1]
    rate = series[args.series]
    sizes = np.unique(np.round(np.geomspace(1, len(rate) // 8, 14)).astype(int))
    rel = []
    for n_ in sizes:
        m_ = len(rate) // n_
        bins = rate[: m_ * n_].reshape(m_, n_).sum(axis=1)
        rel.append(bins.std(ddof=1) / bins.mean())
    d.loglog(sizes, rel, "o-", label="measured")
    d.loglog(sizes, 1 / np.sqrt(rate.mean() * sizes), "k--", label="Poisson expectation")
    d.set(xlabel="bin length [s]", ylabel="relative std of binned rate", title="Excess variance vs bin length (drift appears as a plateau)")
    d.legend(); grid(d)
    save(fig, "d6_time_series_diagnostics.png")

    # 7. detector flux and k vs distance for the available simulated fields
    found = load_fields(args, v["inc_edges"]) if not args.surface_scale else []
    if found:
        dists = [d_ for d_, _ in found]
        tot = [r["rate"].sum() + r["extra"] for _, r in found]
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
        a.plot(dists, tot, "o-"); a.set(xlabel="distance from reactor surface [cm]", ylabel="detector flux [n/cm$^2$/s]",
                                       title="Simulated detector flux vs distance")
        b.plot(dists, [r["surface"] / t for (_, r), t in zip(found, tot)], "o-")
        b.set(xlabel="distance from reactor surface [cm]", ylabel="k = surface / detector flux", title="Surface scaling factor")
        for ax in (a, b):
            if field is not None and "distance" in field and field["distance"] is not None:
                ax.axvline(field["distance"], color="r", ls="--", label=f"used: {field['distance']:.2f} cm")
            grid(ax); ax.legend() if ax.get_legend_handles_labels()[0] else None
        save(fig, "d7_distance_scaling.png")
    print("diagnostic plots:", ", ".join(saved), "->", out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--matrix", required=True)
    p.add_argument("--field", default=None, help="a single reactor_field.npz (distance taken from --distance-cm)")
    p.add_argument("--field-dir", default="runs",
                   help="directory searched for */npz/reactor_field.npz; the field at --distance-cm is picked or "
                        "interpolated between the two bracketing simulated distances")
    p.add_argument("--distance-cm", type=float, default=None,
                   help="detector distance from the reactor surface [cm]; prompted if omitted")
    p.add_argument("--surface-radius-cm", type=float, default=7.0,
                   help="radius of the normalisation surface (outer cowling); distance = detector_x - radius")
    p.add_argument("--surface-flux", type=float, default=2.5e6,
                   help="surface flux [n/cm2/s] for saved fields that do not store it")
    p.add_argument("--surface-scale", type=float, default=None,
                   help="k = surface flux / detector flux (overrides --field)")
    p.add_argument("--efficiency", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--mv-per-mev", type=float, default=None, help="detector scaling; prompted if omitted")
    p.add_argument("--assumed-mv-per-mev", type=float, default=DEFAULT_MV_PER_MEV,
                   help="scaling used for the stored energy axis")
    p.add_argument("--series", choices=["roi", "total"], default="roi", help="per-second counter for flux vs time")
    p.add_argument("--threshold-mev", type=float, default=None,
                   help="PH threshold for the spectrum-based flux (default: lower ROI edge)")
    p.add_argument("--bin-s", type=float, default=10.0, help="time bin for flux vs runtime and the histogram")
    p.add_argument("--noise-kev", type=float, default=100.0)
    p.add_argument("--relative-resolution", type=float, default=0.008)
    p.add_argument("--min-pulse-mev", type=float, default=0.5, help="lowest pulse height used in the unfolding")
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--replicas", type=int, default=30)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-diagnostics", action="store_true", help="skip the plots in <out>/plots/diagnostics")
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mv = args.mv_per_mev
    if mv is None:
        answer = input(f"Detector scaling [mV per MeV] (stored axis assumes {args.assumed_mv_per_mev:g}; "
                       "Enter keeps it): ").strip() if sys.stdin.isatty() else ""
        mv = float(answer) if answer else args.assumed_mv_per_mev
    if mv <= 0:
        raise SystemExit("mV per MeV must be positive")
    to_mev = args.assumed_mv_per_mev / mv
    print(f"using {mv:g} mV/MeV")

    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- measured data
    folder = resolve_path(args.data)
    data_file = folder / "all_datasets.npz" if folder.is_dir() else folder
    meta = json.loads((data_file.parent / "metadata.json").read_text())
    with np.load(data_file) as z:
        energy = np.asarray(z["channel a_spectrum_energy [MeV]"], float) * to_mev
        counts_adc = np.asarray(z["channel a_spectrum_counts"], float)
        amp_mv = np.asarray(z["channel a_spectrum_amplitude [mV]"], float)
        t_s = np.asarray(z["channel a_rate_time [s]"], float)
        series = {"total": np.asarray(z["channel a_rate_total counts"], float),
                  "roi": np.asarray(z["channel a_rate_roi counts"], float)}
    duration = float(meta["duration [s]"])
    raw_energy = energy / to_mev
    series_total = float(series[args.series].sum())
    trigger_mev = float(meta["channel a_trigger [mV]"]) / mv
    roi_lo, roi_hi = (float(x) / mv for x in meta["channel a_roi [mV]"])
    e_max = float(energy[-1])                      # last ADC bin is overflow and dropped from the spectrum
    energy, counts_adc = energy[:-1], counts_adc[:-1]

    # ---- response matrix and efficiency
    z = np.load(resolve_path(args.matrix), allow_pickle=False)
    inc_edges, dep_edges = z["inc_edges"], z["dep_edges"]
    z_matrix, z_error = np.asarray(z["matrix"]), np.asarray(z["error"])
    g = smearing_kernel(dep_edges, args.noise_kev, args.relative_resolution)
    matrix = g @ z["matrix"]
    matrix_err = np.sqrt((g ** 2) @ z["error"] ** 2)
    inc_c, dep_c = 0.5 * (inc_edges[1:] + inc_edges[:-1]), 0.5 * (dep_edges[1:] + dep_edges[:-1])
    inc_w = np.diff(inc_edges)
    table = load_efficiency_table(resolve_path(args.efficiency), inc_c) if args.efficiency else None
    eff = Efficiency(matrix, dep_c, table)

    thresholds = np.arange(0.0, 15.01, 0.5)
    write_efficiency_csv(out / "efficiency_matrix.csv", inc_c, thresholds, [eff.window(t) for t in thresholds])

    # ---- measured spectrum on the matrix grid
    index = np.searchsorted(dep_edges, energy, side="right") - 1
    keep = (index >= 0) & (index < len(dep_edges) - 1)
    counts = np.bincount(index[keep], weights=counts_adc[keep], minlength=len(dep_edges) - 1)
    fit = (dep_c >= args.min_pulse_mev) & (dep_c <= e_max)

    # ---- windows
    thr = roi_lo if args.threshold_mev is None else args.threshold_mev
    win_thr = (thr, e_max)
    win_series = (roi_lo, roi_hi) if args.series == "roi" else (trigger_mev, np.inf)
    in_thr = (dep_c >= win_thr[0]) & (dep_c < win_thr[1])

    k = args.surface_scale
    field = None
    distance = args.distance_cm
    if distance is None and k is None:
        answer = input("Detector distance from the reactor surface [cm] (Enter = use --field as is): ").strip() \
            if sys.stdin.isatty() else ""
        distance = float(answer) if answer else None
    if k is None:
        field = field_at(args, distance, inc_edges)
        k = field["k"]
        print(f"field: {field['description']}")
    print(f"surface/detector flux factor k = {k:.4f}")

    # ---- unfolding and replicas
    # a carbon recoil takes at most 4*12/13^2 = 28.4 % of the neutron energy; lower incident energies cannot
    # produce fitted pulses and would otherwise be driven by the smearing tails alone
    reachable = inc_c >= args.min_pulse_mev / 0.284

    def solve(m, c):
        mm = np.where(fit[:, None] & reachable[None, :], m, 0.0)
        return mlem(mm, np.where(fit, c, 0.0), duration, args.iterations)

    phi0 = solve(matrix, counts)
    shape0 = phi0 / phi0.sum()
    eff_thr0 = float(shape0 @ eff.window(*win_thr))
    eff_ser0 = float(shape0 @ eff.window(*win_series))
    reps = {"phi": [], "eff_thr": [], "eff_ser": [], "flux_thr": []}
    for r in range(args.replicas):
        m_r = np.clip(matrix + rng.normal(0.0, 1.0, matrix.shape) * matrix_err, 0.0, None)
        c_r = rng.poisson(counts).astype(float)
        phi_r = solve(m_r, c_r)
        shape_r = phi_r / phi_r.sum()
        e_thr = float(shape_r @ eff.window(*win_thr, matrix=m_r))
        reps["phi"].append(phi_r)
        reps["eff_thr"].append(e_thr)
        reps["eff_ser"].append(float(shape_r @ eff.window(*win_series, matrix=m_r)))
        reps["flux_thr"].append(c_r[in_thr].sum() / duration / e_thr)
        print(f"  replica {r + 1}/{args.replicas}", end="\r")
    print()
    phi_std = np.std(reps["phi"], axis=0, ddof=1) if args.replicas > 1 else np.zeros_like(phi0)
    eff_ser_rel = float(np.std(reps["eff_ser"], ddof=1) / np.mean(reps["eff_ser"])) if args.replicas > 1 else 0.0

    # ---- fluxes (detector, then surface = k * detector)
    rate_thr = counts[in_thr].sum() / duration
    flux_thr = rate_thr / eff_thr0
    flux_thr_err = float(np.std(reps["flux_thr"], ddof=1)) if args.replicas > 1 else 0.0
    n_bin = max(int(round(args.bin_s)), 1)
    nbins = len(t_s) // n_bin
    per_bin = series[args.series][: nbins * n_bin].reshape(nbins, n_bin).sum(axis=1)
    t_mid = t_s[: nbins * n_bin].reshape(nbins, n_bin).mean(axis=1)
    flux_t = per_bin / n_bin / eff_ser0
    flux_t_err = np.sqrt(np.maximum(per_bin, 1.0)) / n_bin / eff_ser0
    mean_series = series[args.series].sum() / len(t_s) / eff_ser0
    summary = {
        "mv_per_mev": mv, "surface_over_detector_factor_k": k, "duration_s": duration,
        "series_counter": args.series, "series_window_MeV": [win_series[0], None if not np.isfinite(win_series[1]) else win_series[1]],
        "spectrum_threshold_MeV": [win_thr[0], win_thr[1]],
        "effective_efficiency_series_cm2": eff_ser0, "effective_efficiency_series_rel_uncertainty": eff_ser_rel,
        "effective_efficiency_threshold_cm2": eff_thr0,
        "detector_flux_series_mean_n_cm2_s": mean_series,
        "surface_flux_series_mean_n_cm2_s": mean_series * k,
        "detector_flux_threshold_n_cm2_s": flux_thr, "detector_flux_threshold_uncertainty": flux_thr_err,
        "surface_flux_threshold_n_cm2_s": flux_thr * k, "surface_flux_threshold_uncertainty": flux_thr_err * k,
        "detector_flux_unfolded_total_1_16MeV_n_cm2_s": float(phi0.sum()),
        "surface_flux_unfolded_total_1_16MeV_n_cm2_s": float(phi0.sum() * k),
        "surface_flux_unfolded_total_uncertainty": float(np.std([p.sum() for p in reps["phi"]], ddof=1) * k)
        if args.replicas > 1 else 0.0,
        "series_flux_binned_std_over_poisson_expectation": float(np.std(flux_t, ddof=1) / np.sqrt(np.mean(flux_t_err ** 2)))
        if nbins > 2 else None,
        "notes": "flux in the series/threshold estimates assumes the unfolded spectral shape over 1-16 MeV; "
                 "uncertainties are statistical (counts + matrix), not unfolding bias.",
    }
    if win_series[0] < args.min_pulse_mev:
        summary["warning"] = (f"series window starts at {win_series[0]:.2f} MeV, below the unfolding limit "
                              f"({args.min_pulse_mev:g} MeV): pulses there (gammas, noise, sub-MeV neutrons) are not "
                              "described by the response matrix, so this flux is biased; compare with the ROI series")
        print("WARNING:", summary["warning"])
    (out / "flux_summary.json").write_text(json.dumps(summary, indent=2))

    with open(out / "flux_vs_time.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "counts", "detector_flux_n_cm2_s", "detector_flux_stat_error",
                    "surface_flux_n_cm2_s", "surface_flux_stat_error"])
        for row in zip(t_mid, per_bin, flux_t, flux_t_err, flux_t * k, flux_t_err * k):
            w.writerow([f"{v:.6g}" for v in row])
    with open(out / "unfolded_spectrum.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["energy_low_MeV", "energy_high_MeV", "energy_center_MeV", "detector_flux_n_cm2_s_MeV",
                    "surface_flux_n_cm2_s_MeV", "surface_flux_std_n_cm2_s_MeV"])
        for lo, hi, c, v, s in zip(inc_edges[:-1], inc_edges[1:], inc_c, phi0 / inc_w, phi_std / inc_w):
            w.writerow([f"{x:.6g}" for x in (lo, hi, c, v, v * k, s * k)])

    # ---- plots
    label = f"{args.series} counter" + (f" ({win_series[0]:.2f}-{win_series[1]:.2f} MeV)" if np.isfinite(win_series[1])
                                         else f" (> {win_series[0]:.2f} MeV)")
    t_min = t_mid / 60.0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(t_min, flux_t * k, yerr=flux_t_err * k, fmt=".", ms=3, lw=0.7, color="C0", label=f"{args.bin_s:g} s bins")
    ax.axhline(mean_series * k, color="k", lw=1, label=f"run mean {mean_series * k:.3g}")
    ax.set(xlabel="runtime [min]", ylabel="flux at reactor surface [n/cm$^2$/s]",
           title=f"Flux vs runtime ({label}, {mv:g} mV/MeV)")
    sec = ax.secondary_yaxis("right", functions=(lambda y: y / k, lambda y: y * k))
    sec.set_ylabel("flux at detector [n/cm$^2$/s]")
    ax.legend(loc="best")
    grid(ax)
    fig.tight_layout(); fig.savefig(out / "plots" / "flux_vs_time.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = flux_t * k
    ax.hist(vals, bins=min(40, max(nbins // 4, 5)), density=True, color="C0", alpha=0.6, label="measured")
    mu, sd = float(vals.mean()), float(vals.std(ddof=1)) if nbins > 2 else 0.0
    sd_p = float(np.sqrt(np.mean(flux_t_err ** 2))) * k
    xs = np.linspace(max(mu - 5 * max(sd, sd_p), 0), mu + 5 * max(sd, sd_p), 400)
    if sd > 0:
        ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), "k", label=f"Gaussian, $\\sigma$={sd:.3g}")
    ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sd_p) ** 2) / (sd_p * np.sqrt(2 * np.pi)), "r--",
            label=f"Poisson expectation, $\\sigma$={sd_p:.3g}")
    ax.set(xlabel="flux at reactor surface [n/cm$^2$/s]", ylabel="frequency density [(n/cm$^2$/s)$^{-1}$]",
           title=f"Distribution of {args.bin_s:g} s flux values ({label})")
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / k, lambda x: x * k))
    sec.set_xlabel("flux at detector [n/cm$^2$/s]")
    ax.legend(); grid(ax)
    fig.tight_layout(); fig.savefig(out / "plots" / "flux_histogram.png", dpi=150); plt.close(fig)

    surf = phi0 / inc_w * k
    band = phi_std / inc_w * k
    fig, ax = plt.subplots(figsize=(10, 5.5))
    inset = ax.inset_axes([0.45, 0.45, 0.52, 0.5])
    for a in (ax, inset):
        a.step(inc_c, surf, where="mid", color="C1", label="unfolded from measurement")
        a.fill_between(inc_c, np.maximum(surf - band, 0 if a is ax else 1e-30), surf + band, step="mid", color="C1",
                       alpha=0.3, label="$\\pm1\\sigma$ (counts + matrix)")
        if field is not None:
            a.step(inc_c, field["rate"] / inc_w * k, where="mid", color="k", lw=1, label="simulated reactor field")
        grid(a)
    inset.set_yscale("log")
    inset.set_ylim(max(surf[surf > 0].min() * 0.5, 1e-3) if (surf > 0).any() else 1e-3, surf.max() * 3)
    ax.set(xlabel="incident neutron energy [MeV]", ylabel="flux at reactor surface [n/cm$^2$/s/MeV]",
           title=f"Unfolded spectrum (MLEM, {args.iterations} it., {args.replicas} replicas; pulses > {args.min_pulse_mev:g} MeV)", ylim=(0, None))
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out / "plots" / "unfolded_spectrum_surface.png", dpi=150); plt.close(fig)


    if not args.no_diagnostics:
        diagnostics(locals())
    print(json.dumps({k_: v for k_, v in summary.items() if k_ != "notes"}, indent=1))
    print("wrote results to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

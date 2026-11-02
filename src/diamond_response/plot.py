"""Plots of the response matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .response import ResponseMatrix

THRESHOLDS_MEV = (0.0, 0.5, 1.0, 5.0, 7.5)


_BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)
_ARROW = dict(arrowstyle="-", lw=0.4, color="grey")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def grid(ax) -> None:
    ax.minorticks_on()
    ax.grid(which="major", lw=0.8, alpha=0.6)
    ax.grid(which="minor", lw=0.4, alpha=0.35, ls=":")


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    _plt().close(fig)
    return path


def _label(threshold: float) -> str:
    return "all pulses" if threshold <= 0 else f"pulses > {threshold:g} MeV"


def efficiency_above(rm: ResponseMatrix, threshold: float):
    """Efficiency (cm^2) and its statistical error for pulses above a pulse-height threshold."""
    sel = rm.dep_centers > threshold if threshold > 0 else np.ones(len(rm.dep_centers), bool)
    return rm.matrix[sel].sum(axis=0), np.sqrt((rm.error[sel] ** 2).sum(axis=0))


def _log_inset(ax, x, ys, labels=None):
    inset = ax.inset_axes([0.62, 0.38, 0.35, 0.55])
    for y in ys:
        inset.step(x, np.where(y > 0, y, np.nan), where="mid", lw=0.8)
    inset.set_yscale("log")
    top = max((np.nanmax(y) for y in ys if np.any(y > 0)), default=0.0)
    if top > 0:
        inset.set_ylim(top * 1e-7, top * 3)
    inset.set_title("log scale", fontsize=8)
    inset.tick_params(labelsize=7)
    grid(inset)
    return inset


def plot_matrix(rm: ResponseMatrix, outdir: Path, folded: ResponseMatrix | None = None) -> list[Path]:
    from matplotlib.colors import LogNorm

    plt = _plt()
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    m = folded or rm
    data = np.where(m.matrix > 0, m.matrix, np.nan)
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(m.inc_edges, m.dep_edges, data, norm=LogNorm(), shading="flat")
    fig.colorbar(mesh, label="pulses per unit fluence per bin [cm$^2$]")
    ax.set_xlabel("incident neutron energy [MeV]")
    ax.set_ylabel("deposited (collected) energy [MeV]")
    grid(ax)
    paths.append(_save(fig, outdir / "response_matrix.png"))

    return paths


LINE_ENERGIES_MEV = (2.45, 13.8, 14.1, 14.5)


def column_at(rm: ResponseMatrix, energy_mev: float) -> np.ndarray:
    """Response column at an arbitrary energy: linear interpolation between neighbouring bin centres."""
    c = rm.inc_centers
    j = int(np.clip(np.searchsorted(c, energy_mev) - 1, 0, len(c) - 2))
    w = float(np.clip((energy_mev - c[j]) / (c[j + 1] - c[j]), 0.0, 1.0))
    return (1 - w) * rm.matrix[:, j] + w * rm.matrix[:, j + 1]


def source_average(rm: ResponseMatrix, probability: np.ndarray) -> np.ndarray:
    """Response per unit fluence of the unmoderated source spectrum (probability per incident bin, sums <= 1)."""
    total = probability.sum()
    return rm.matrix @ (probability / total) if total > 0 else np.zeros(rm.matrix.shape[0])


def _curves(rm, probability, energies):
    curves = [(f"{e:g} MeV", column_at(rm, e)) for e in energies]
    if probability is not None:
        curves.append(("unmoderated source (custom D-T)", source_average(rm, probability)))
    return curves


def plot_response_lines(rm: ResponseMatrix, outdir: Path, probability=None, energies=LINE_ENERGIES_MEV,
                        name="response_lines.png", title_suffix="") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = np.diff(rm.dep_edges)
    fig, ax = plt.subplots(figsize=(9, 5))
    curves = _curves(rm, probability, energies)
    for label, col in curves:
        ax.step(rm.dep_centers, col / width, where="mid", label=label)
    ax.set_xlabel("deposited (collected) energy [MeV]")
    ax.set_ylabel("pulses per unit fluence per MeV [cm$^2$/MeV]")
    ax.set_yscale("linear")
    ax.set_ylim(bottom=0)
    ax.set_title("Diamond response" + title_suffix)
    grid(ax)
    ax.legend(loc="upper left", fontsize=8)
    _log_inset(ax, rm.dep_centers, [c / width for _, c in curves])
    path = outdir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_source_response(rm: ResponseMatrix, outdir: Path, probability, name="response_custom_source.png") -> Path:
    """Pulse-height response to the unmoderated custom D-T source: linear y with a log-y inset."""
    plt = _plt()
    width = np.diff(rm.dep_edges)
    y = source_average(rm, probability) / width
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.step(rm.dep_centers, y, where="mid")
    ax.set_xlabel("deposited (collected) energy [MeV]")
    ax.set_ylabel("pulses per unit fluence per MeV [cm$^2$/MeV]")
    ax.set_title("Response to the unmoderated custom D-T source")
    ax.set_ylim(bottom=0)
    grid(ax)
    _log_inset(ax, rm.dep_centers, [y])
    return _save(fig, outdir / name)


def plot_efficiency_lines(rm: ResponseMatrix, outdir: Path, probability=None, energies=LINE_ENERGIES_MEV,
                          thresholds=THRESHOLDS_MEV) -> list[Path]:
    """Efficiency (pulses per unit fluence, cm^2) vs energy with error bars for several thresholds, and vs threshold."""
    plt = _plt()
    paths = []
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, t in enumerate(thresholds):
        eff, err = efficiency_above(rm, t)
        line = ax.errorbar(rm.inc_centers, eff, yerr=err, fmt="-", marker=".", ms=3, lw=1, elinewidth=0.7,
                           capsize=1.5, label=_label(t))
        color = line[0].get_color()
        for e in energies:
            y = float(np.interp(e, rm.inc_centers, eff))
            ax.plot([e], [y], "o", color=color, ms=4)
            ax.annotate(f"{y:.3g}", (e, y), xytext=(6, (2 - k) * 9), textcoords="offset points", fontsize=7,
                        color=color, bbox=_BOX, arrowprops=_ARROW)
    ax.set_yscale("linear")
    ax.set_ylim(bottom=0)
    for n, e in enumerate(energies):
        ax.axvline(e, color="grey", ls=":", lw=0.8)
        ax.annotate(f"{e:g}", (e, ax.get_ylim()[1]), ha="center", va="bottom", fontsize=8,
                    xytext=(0, 10 * (n % 2)), textcoords="offset points")
    ax.set_xlabel("incident neutron energy [MeV]")
    ax.set_ylabel("efficiency [cm$^2$]  (error bars: 1 sigma, statistical)")
    grid(ax)
    ax.legend()
    paths.append(_save(fig, outdir / "efficiency_vs_energy.png"))

    fig, ax = plt.subplots(figsize=(10, 6))
    edges = rm.dep_edges
    for k, (label, col) in enumerate(_curves(rm, probability, energies)):
        cum = np.cumsum(col[::-1])[::-1]
        line, = ax.step(edges[:-1], cum, where="post", label=label)
        full = np.append(cum, 0.0)
        for t in thresholds:
            y = float(np.interp(t, edges, full))
            ax.plot([t], [y], "o", color=line.get_color(), ms=4)
            ax.annotate(f"{y:.3g}", (t, y), xytext=(8, (2 - k) * 9), textcoords="offset points", fontsize=7,
                        color=line.get_color(), bbox=_BOX, arrowprops=_ARROW)
    for t in thresholds:
        ax.axvline(t, color="grey", ls=":", lw=0.8)
    ax.set_yscale("linear")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("pulse-height threshold [MeV]")
    ax.set_ylabel("efficiency for pulses above threshold [cm$^2$]")
    grid(ax)
    ax.legend()
    paths.append(_save(fig, outdir / "efficiency_vs_threshold.png"))
    return paths


def plot_efficiency_heatmap(rm: ResponseMatrix, outdir: Path, name="efficiency_heatmap.png") -> Path:
    """Two panels: efficiency per MeV of pulse height (log colour, 4 decades) and efficiency of pulses
    above a pulse-height threshold (linear colour). Dashed line: pulse height = incident energy."""
    from matplotlib.colors import LogNorm

    plt = _plt()
    width = np.diff(rm.dep_edges)[:, None]
    density = rm.matrix / width
    cumulative = np.cumsum(rm.matrix[::-1], axis=0)[::-1]
    top = float(density.max()) if density.max() > 0 else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    mesh = axes[0].pcolormesh(rm.inc_edges, rm.dep_edges, np.where(density > 0, density, np.nan),
                              norm=LogNorm(vmin=top * 1e-4, vmax=top), shading="flat", cmap="viridis")
    fig.colorbar(mesh, ax=axes[0], label="efficiency per MeV of pulse height [cm$^2$/MeV]")
    axes[0].set_title("Differential: pulses per unit fluence per MeV")
    mesh = axes[1].pcolormesh(rm.inc_edges, rm.dep_edges, cumulative, shading="flat", cmap="inferno")
    fig.colorbar(mesh, ax=axes[1], label="efficiency for pulses above this pulse height [cm$^2$]")
    axes[1].set_title("Integral: efficiency above pulse-height threshold")
    for ax in axes:
        ax.plot(rm.inc_edges, rm.inc_edges, "w--", lw=0.8)
        ax.set_xlim(rm.inc_edges[0], rm.inc_edges[-1])
        ax.set_ylim(rm.dep_edges[0], rm.dep_edges[-1])
        ax.set_xlabel("incident neutron energy [MeV]")
        grid(ax)
    axes[0].set_ylabel("pulse height (deposited energy) [MeV]")
    return _save(fig, outdir / name)


def plot_all(rm: ResponseMatrix, outdir: Path, smeared: ResponseMatrix | None = None, probability=None) -> list[Path]:
    """Line/efficiency/heatmap plots; the response panel is also drawn with resolution if ``smeared`` is given."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [plot_response_lines(rm, outdir, probability)]
    if smeared is not None:
        paths.append(plot_response_lines(smeared, outdir, probability, name="response_lines_with_resolution.png",
                                         title_suffix=" (with resolution)"))
    if probability is not None:
        paths.append(plot_source_response(rm, outdir, probability))
    paths += plot_efficiency_lines(rm, outdir, probability)
    paths.append(plot_efficiency_heatmap(rm, outdir))
    return paths

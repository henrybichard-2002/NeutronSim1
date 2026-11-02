"""Plots of the response matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .response import ResponseMatrix


def plot_matrix(rm: ResponseMatrix, outdir: Path, folded: ResponseMatrix | None = None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    m = folded or rm
    data = np.where(m.matrix > 0, m.matrix, np.nan)
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(m.inc_edges, m.dep_edges, data, norm=LogNorm(), shading="flat")
    fig.colorbar(mesh, label="pulses per unit fluence per bin [cm$^2$]")
    ax.set_xlabel("incident neutron energy [MeV]")
    ax.set_ylabel("deposited (collected) energy [MeV]")
    paths.append(outdir / "response_matrix.png")
    fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rm.inc_centers, rm.efficiency(0.5), label="> 0.5 MeV")
    ax.plot(rm.inc_centers, rm.efficiency(0.0), label="all pulses")
    ax.set_xlabel("incident neutron energy [MeV]")
    ax.set_ylabel("efficiency [cm$^2$]")
    ax.legend()
    paths.append(outdir / "efficiency.png")
    fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    near = int(np.argmin(np.abs(rm.inc_centers - 14.05)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(m.dep_centers, m.matrix[:, near], where="mid")
    ax.set_yscale("log")
    ax.set_xlabel("deposited energy [MeV]")
    ax.set_ylabel("pulses per fluence per bin [cm$^2$]")
    ax.set_title(f"Response column, incident {rm.inc_edges[near]:.2f}-{rm.inc_edges[near + 1]:.2f} MeV")
    paths.append(outdir / "response_14MeV.png")
    fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)
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
    for label, col in _curves(rm, probability, energies):
        ax.step(rm.dep_centers, col / width, where="mid", label=label)
    ax.set_xlabel("deposited (collected) energy [MeV]")
    ax.set_ylabel("pulses per unit fluence per MeV [cm$^2$/MeV]")
    ax.set_yscale("linear")
    ax.set_ylim(bottom=0)
    ax.set_title("Diamond response" + title_suffix)
    ax.legend()
    path = outdir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_efficiency_lines(rm: ResponseMatrix, outdir: Path, probability=None, energies=LINE_ENERGIES_MEV) -> list[Path]:
    """Efficiency (pulses per unit fluence, cm^2): vs energy with the line energies marked, and vs pulse-height threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rm.inc_centers, rm.efficiency(0.5), label="pulses > 0.5 MeV")
    ax.plot(rm.inc_centers, rm.efficiency(0.0), label="all pulses")
    ax.set_yscale("linear")
    ax.set_ylim(bottom=0)
    for n, e in enumerate(energies):
        ax.axvline(e, color="grey", ls=":", lw=0.8)
        ax.annotate(f"{e:g}", (e, ax.get_ylim()[1]), ha="center", va="bottom", fontsize=8,
                    xytext=(0, 10 * (n % 2)), textcoords="offset points")
    ax.set_xlabel("incident neutron energy [MeV]")
    ax.set_ylabel("efficiency [cm$^2$]")
    ax.legend()
    paths.append(outdir / "efficiency_vs_energy.png")
    fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, col in _curves(rm, probability, energies):
        above = np.cumsum(col[::-1])[::-1]
        ax.step(rm.dep_edges[:-1], above, where="post", label=label)
    ax.set_yscale("linear")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("pulse-height threshold [MeV]")
    ax.set_ylabel("efficiency for pulses above threshold [cm$^2$]")
    ax.legend()
    paths.append(outdir / "efficiency_vs_threshold.png")
    fig.savefig(paths[-1], dpi=150, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_efficiency_heatmap(rm: ResponseMatrix, outdir: Path, name="efficiency_heatmap.png") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    data = np.where(rm.matrix > 0, rm.matrix / np.diff(rm.dep_edges)[:, None], np.nan)
    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(rm.inc_edges, rm.dep_edges, data, norm=LogNorm(), shading="flat")
    fig.colorbar(mesh, label="efficiency [cm$^2$ per MeV of pulse height]")
    ax.set_xlabel("incident neutron energy [MeV]")
    ax.set_ylabel("pulse height (deposited energy) [MeV]")
    path = outdir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_all(rm: ResponseMatrix, outdir: Path, smeared: ResponseMatrix | None = None, probability=None) -> list[Path]:
    """Line/efficiency/heatmap plots; the response panel is also drawn with resolution if ``smeared`` is given."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [plot_response_lines(rm, outdir, probability)]
    if smeared is not None:
        paths.append(plot_response_lines(smeared, outdir, probability, name="response_lines_with_resolution.png",
                                         title_suffix=" (with resolution)"))
    paths += plot_efficiency_lines(rm, outdir, probability)
    paths.append(plot_efficiency_heatmap(rm, outdir))
    return paths

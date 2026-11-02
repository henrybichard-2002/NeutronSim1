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

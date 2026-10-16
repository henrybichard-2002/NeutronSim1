"""Terminal summary of neutron transport tallies from a completed run."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TextIO

from .config import SimulationConfig


def _means(tally, score: str) -> list[float]:
    return [float(value) for value in tally.get_values(scores=[score], value="mean").reshape(-1)]


def _spectrum_summary(values: list[float], edges_mev: tuple[float, ...]):
    if len(values) != len(edges_mev) - 1:
        raise ValueError(
            "Spectrum tally bin count does not match the configured energy bins"
        )
    total = sum(values)
    if total <= 0:
        return None
    mean_energy = sum(
        value * (left + right) / 2
        for value, left, right in zip(
            values, edges_mev[:-1], edges_mev[1:], strict=True
        )
    ) / total
    occupied_bins = [index for index, value in enumerate(values) if value > 0]
    maximum_bin_upper_edge = edges_mev[occupied_bins[-1] + 1]
    return mean_energy, maximum_bin_upper_edge


def _net_scatter_energy_change(tally, openmc) -> float:
    incoming_filter = next(
        item for item in tally.filters if isinstance(item, openmc.EnergyFilter)
    )
    outgoing_filter = next(
        item for item in tally.filters if isinstance(item, openmc.EnergyoutFilter)
    )
    incoming_bins = incoming_filter.bins
    outgoing_bins = outgoing_filter.bins
    rates = tally.get_values(scores=["scatter"], value="mean").reshape(
        len(incoming_bins), len(outgoing_bins)
    )
    return sum(
        (
            (float(incoming[0]) + float(incoming[1])
             - float(outgoing[0]) - float(outgoing[1]))
            / 2.0e6
        )
        * float(rates[in_index, out_index])
        for in_index, incoming in enumerate(incoming_bins)
        for out_index, outgoing in enumerate(outgoing_bins)
    )


def print_run_summary(
    statepoint_path: str | Path,
    config: SimulationConfig,
    *,
    file: TextIO = sys.stdout,
) -> None:
    """Print source-normalized spectra, leakage, secondary production, and reactions."""
    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError("OpenMC is required to read the run statepoint") from exc

    statepoint = openmc.StatePoint(str(statepoint_path))
    try:
        source_histories = config.settings.particles * config.settings.batches
        print("\nNeutron transport summary", file=file)
        print(f"  Source histories: {source_histories:,}", file=file)
        print(
            "  Spectral means are track-length weighted; maximum energies are "
            "the upper edge of the highest occupied 0.1 MeV bin.",
            file=file,
        )

        region_names = ["source", "source_cavity"]
        region_names.extend(layer.name for layer in config.geometry.layers)
        region_names.extend(
            component.name for component in config.geometry.components
        )
        edges_mev = config.tally.energy_bins_mev
        print("\n  Region neutron energies:", file=file)
        for region in region_names:
            tally = statepoint.get_tally(name=f"flux_spectrum_{region}")
            summary = _spectrum_summary(_means(tally, "flux"), edges_mev)
            if summary is None:
                print(f"    {region}: no scored neutron track length", file=file)
            else:
                mean_energy, maximum = summary
                print(
                    f"    {region}: mean {mean_energy:.3f} MeV; "
                    f"maximum-bin upper edge {maximum:.2f} MeV",
                    file=file,
                )

        leakage_tally = statepoint.get_tally(name="world_neutron_leakage_spectrum")
        leakage_spectrum = _means(leakage_tally, "current")
        leakage_per_source = sum(leakage_spectrum)
        expected_leakage = leakage_per_source * source_histories
        net_neutrons = expected_leakage - source_histories
        net_percent = 100 * (leakage_per_source - 1)
        print("\n  World-boundary neutron balance:", file=file)
        print(
            f"    Outward leakage: {leakage_per_source:.6g} neutrons/source "
            f"history (about {expected_leakage:,.1f} over this run)",
            file=file,
        )
        leakage_summary = _spectrum_summary(leakage_spectrum, edges_mev)
        if leakage_summary is not None:
            print(
                f"    Escaping neutron mean energy: {leakage_summary[0]:.3f} MeV; "
                f"maximum-bin upper edge {leakage_summary[1]:.2f} MeV",
                file=file,
            )
        print(
            f"    Escaped minus source primaries: {net_neutrons:+,.1f} "
            f"({net_percent:+.3f}%)",
            file=file,
        )

        production = statepoint.get_tally(name="secondary_particle_production")
        production_filter = next(
            item
            for item in production.filters
            if isinstance(item, openmc.ParticleProductionFilter)
        )
        production_values = _means(production, "events")
        production_by_particle: dict[str, float] = {}
        for (particle, _low, _high), value in zip(
            production_filter.bins, production_values, strict=True
        ):
            production_by_particle[particle] = (
                production_by_particle.get(particle, 0.0) + value
            )
        print("\n  Secondary particle production per source history:", file=file)
        for particle, value in sorted(production_by_particle.items()):
            if particle == "photon" and not config.physics.photon_transport:
                print("    photon: transport disabled in this configuration", file=file)
                continue
            print(f"    {particle}: {value:.6g}", file=file)

        interaction_scores = (
            "total",
            "scatter",
            "absorption",
            "fission",
            "nu-fission",
            "(n,2n)",
            "(n,3n)",
            "(n,gamma)",
        )
        material_names = [
            layer.name for layer in config.geometry.layers
        ] + [component.name for component in config.geometry.components]
        print("\n  Neutron interaction scores per source history:", file=file)
        for material_name in material_names:
            tally = statepoint.get_tally(
                name=f"neutron_interactions_{material_name}"
            )
            rates = [
                (score, _means(tally, score)[0]) for score in interaction_scores
            ]
            active = [(score, value) for score, value in rates if value > 0]
            if active:
                formatted = ", ".join(
                    f"{score}={value:.5g}" for score, value in active
                )
            else:
                formatted = "no scored neutron interactions"
            energy_transfer = statepoint.get_tally(
                name=f"neutron_energy_transfer_{material_name}"
            )
            energy_change_mev = _net_scatter_energy_change(
                energy_transfer, openmc
            )
            print(
                f"    {material_name}: {formatted}; "
                f"net scatter energy loss={energy_change_mev:.5g} "
                "MeV/source history (bin-midpoint estimate)",
                file=file,
            )
    finally:
        statepoint.close()

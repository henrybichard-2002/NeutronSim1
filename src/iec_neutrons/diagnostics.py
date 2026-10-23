"""Terminal summary of neutron transport tallies from a completed run."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import TextIO

from .config import SimulationConfig


@dataclass(frozen=True)
class FluxCalibration:
    target_flux_n_cm2_s: float
    simulated_flux_cm2_source: float
    scale_histories_s: float


def _means(tally, score: str) -> list[float]:
    return [float(value) for value in tally.get_values(scores=[score], value="mean").reshape(-1)]


def calculate_flux_calibration(
    statepoint, config: SimulationConfig
) -> FluxCalibration | None:
    """Return the source-history rate that matches the configured surface flux."""
    normalization = config.normalization
    if normalization is None:
        return None
    try:
        tally = statepoint.get_tally(name="flux_surface_normalization")
    except KeyError as exc:
        raise ValueError(
            "Statepoint is missing the surface-flux normalization tally; "
            "rerun with the current model."
        ) from exc
    values = _means(tally, "flux")
    if len(values) != 1:
        raise ValueError(
            "Surface-flux normalization tally must contain exactly one mesh bin"
        )
    width_cm = normalization.sample_width_cm
    simulated_flux = values[0] / width_cm**3
    if not math.isfinite(simulated_flux) or simulated_flux <= 0:
        raise ValueError(
            "Surface-flux normalization tally must have a positive finite mean"
        )
    scale = normalization.surface_flux_n_cm2_s / simulated_flux
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Surface-flux normalization produced an invalid scale")
    return FluxCalibration(
        target_flux_n_cm2_s=normalization.surface_flux_n_cm2_s,
        simulated_flux_cm2_source=simulated_flux,
        scale_histories_s=scale,
    )


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


def export_he3_spectra_csv(
    statepoint_path: str | Path,
    config: SimulationConfig,
    output_path: str | Path,
) -> Path | None:
    """Write all configured He-3 neutron spectra and capture responses to CSV."""
    detector_components = [
        component
        for component in config.geometry.components
        if component.material != "void"
        and "He3" in (config.materials[component.material].nuclides or {})
    ]
    if not detector_components:
        return None

    try:
        import numpy as np
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC and NumPy are required to export He-3 spectra"
        ) from exc

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = config.normalization is not None
    fieldnames = (
        "detector",
        "energy_min_mev",
        "energy_max_mev",
        "energy_midpoint_mev",
        (
            "neutron_flux_mean_n_cm-2_s-1_MeV-1"
            if normalized
            else "neutron_flux_mean_cm-2_source-1_MeV-1"
        ),
        (
            "neutron_flux_std_dev_n_cm-2_s-1_MeV-1"
            if normalized
            else "neutron_flux_std_dev_cm-2_source-1_MeV-1"
        ),
        (
            "he3_np_capture_mean_reactions_s-1_MeV-1"
            if normalized
            else "he3_np_capture_mean_reactions_source-1_MeV-1"
        ),
        (
            "he3_np_capture_std_dev_reactions_s-1_MeV-1"
            if normalized
            else "he3_np_capture_std_dev_reactions_source-1_MeV-1"
        ),
        *(("surface_flux_target_n_cm-2_s-1", "normalization_scale_histories_s-1") if normalized else ()),
    )

    with openmc.StatePoint(str(statepoint_path)) as statepoint:
        calibration = calculate_flux_calibration(statepoint, config)
        scale = calibration.scale_histories_s if calibration is not None else 1.0
        rows = []
        available_names = {
            tally.name for tally in statepoint.tallies.values()
        }
        for component in detector_components:
            tally_names = (
                f"he3_neutron_spectrum_{component.name}",
                f"he3_n_p_spectrum_{component.name}",
            )
            missing = set(tally_names) - available_names
            if missing:
                raise ValueError(
                    f"Statepoint is missing He-3 spectrum tallies for "
                    f"{component.name!r}: {', '.join(sorted(missing))}. "
                    "Rerun with the current model."
                )
            neutron_tally = statepoint.get_tally(
                name=tally_names[0]
            )
            response_tally = statepoint.get_tally(name=tally_names[1])
            neutron_filter = next(
                (
                    item
                    for item in neutron_tally.filters
                    if isinstance(item, openmc.EnergyFilter)
                ),
                None,
            )
            response_filter = next(
                (
                    item
                    for item in response_tally.filters
                    if isinstance(item, openmc.EnergyFilter)
                ),
                None,
            )
            if neutron_filter is None or response_filter is None:
                raise ValueError(
                    f"He-3 spectra for {component.name!r} must have energy filters"
                )
            neutron_bins = np.asarray(neutron_filter.bins, dtype=float)
            response_bins = np.asarray(response_filter.bins, dtype=float)
            if (
                neutron_bins.ndim != 2
                or neutron_bins.shape[1] != 2
                or response_bins.ndim != 2
                or response_bins.shape[1] != 2
                or not np.allclose(neutron_bins, response_bins)
            ):
                raise ValueError(
                    f"He-3 neutron and capture spectra for {component.name!r} "
                    "must have matching energy bins"
                )

            neutron_mean = np.asarray(
                neutron_tally.get_values(scores=["flux"], value="mean"),
                dtype=float,
            ).reshape(-1)
            neutron_std_dev = np.asarray(
                neutron_tally.get_values(scores=["flux"], value="std_dev"),
                dtype=float,
            ).reshape(-1)
            response_mean = np.asarray(
                response_tally.get_values(scores=["(n,p)"], value="mean"),
                dtype=float,
            ).reshape(-1)
            response_std_dev = np.asarray(
                response_tally.get_values(scores=["(n,p)"], value="std_dev"),
                dtype=float,
            ).reshape(-1)
            bin_count = neutron_bins.shape[0]
            if any(
                values.size != bin_count
                for values in (
                    neutron_mean,
                    neutron_std_dev,
                    response_mean,
                    response_std_dev,
                )
            ):
                raise ValueError(
                    f"He-3 spectra for {component.name!r} have an unexpected "
                    "number of energy bins"
                )

            if component.shape == "box":
                volume_cm3 = math.prod(component.dimensions_cm)
            else:
                radius_cm, length_cm = component.dimensions_cm
                volume_cm3 = math.pi * radius_cm**2 * length_cm
            energy_bins_mev = neutron_bins / 1.0e6
            for index, (lower, upper) in enumerate(energy_bins_mev):
                bin_width = float(upper - lower)
                rows.append(
                    {
                        "detector": component.name,
                        "energy_min_mev": float(lower),
                        "energy_max_mev": float(upper),
                        "energy_midpoint_mev": float((lower + upper) / 2),
                        fieldnames[4]: float(neutron_mean[index])
                        / volume_cm3
                        / bin_width
                        * scale,
                        fieldnames[5]: float(neutron_std_dev[index])
                        / volume_cm3
                        / bin_width
                        * scale,
                        fieldnames[6]: float(response_mean[index])
                        / bin_width
                        * scale,
                        fieldnames[7]: float(response_std_dev[index])
                        / bin_width
                        * scale,
                    }
                )
                if calibration is not None:
                    rows[-1].update(
                        {
                            "surface_flux_target_n_cm-2_s-1": calibration.target_flux_n_cm2_s,
                            "normalization_scale_histories_s-1": calibration.scale_histories_s,
                        }
                    )

    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


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
        calibration = calculate_flux_calibration(statepoint, config)
        scale = calibration.scale_histories_s if calibration is not None else 1.0
        print("\nNeutron transport summary", file=file)
        print(f"  Source histories: {source_histories:,}", file=file)
        if calibration is not None:
            print(
                "  Surface flux calibration: "
                f"{calibration.target_flux_n_cm2_s:,.6g} n cm-2 s-1 at "
                f"{config.normalization.surface_component} (positive-x surface, "
                "cathode midplane)",
                file=file,
            )
            print(
                "  Simulated reference flux: "
                f"{calibration.simulated_flux_cm2_source:.6g} cm-2/source history; "
                f"scale: {scale:.6g} source histories/s",
                file=file,
            )
        print(
            "  Spectral means are track-length weighted; maximum energies are "
            "the upper edge of the highest occupied 0.1 MeV bin.",
            file=file,
        )

        component_source_tally = (
            config.custom_source is not None and config.normalization is not None
        )
        region_names = (
            ["source"]
            if component_source_tally
            else ["source", "source_cavity"]
        )
        region_names.extend(layer.name for layer in config.geometry.layers)
        region_names.extend(
            component.name for component in config.geometry.components
        )
        region_names.extend(probe.name for probe in config.spectral_probes)
        edges_mev = config.tally.energy_bins_mev
        print("\n  Region neutron energies:", file=file)
        for region in region_names:
            is_probe = region in {probe.name for probe in config.spectral_probes}
            tally_name = (
                f"spectral_probe_{region}"
                if is_probe
                else f"flux_spectrum_{region}"
            )
            tally = statepoint.get_tally(name=tally_name)
            summary = _spectrum_summary(_means(tally, "flux"), edges_mev)
            if summary is None:
                print(f"    {region}: no scored neutron track length", file=file)
            else:
                mean_energy, maximum = summary
                display_name = (
                    "source (reactor vacuum cavity)"
                    if region == "source" and component_source_tally
                    else region
                )
                print(
                    f"    {display_name}: mean {mean_energy:.3f} MeV; "
                    f"maximum-bin upper edge {maximum:.2f} MeV",
                    file=file,
                )

        if config.custom_source is not None:
            source_spectrum_path = Path(statepoint_path).parent / "source_birth_spectrum.csv"
            if not source_spectrum_path.is_file():
                raise ValueError(
                    "Custom-source birth-energy data is missing beside the "
                    "statepoint; rerun with the current custom-source runner."
                )
            with source_spectrum_path.open(newline="", encoding="utf-8") as source_file:
                source_rows = list(csv.DictReader(source_file))
            if not source_rows:
                raise ValueError(
                    f"Custom-source birth-energy file is empty: {source_spectrum_path}"
                )
            probabilities = [
                float(row["probability_per_source_neutron"])
                for row in source_rows
            ]
            source_edges = tuple(
                [float(source_rows[0]["energy_lower_mev"])]
                + [float(row["energy_upper_mev"]) for row in source_rows]
            )
            if (
                len(probabilities) != len(edges_mev) - 1
                or source_edges != edges_mev
                or not math.isclose(sum(probabilities), 1.0)
            ):
                raise ValueError(
                    "Custom-source birth-energy data does not match the "
                    "configured tally bins."
                )
            summary = _spectrum_summary(probabilities, edges_mev)
            if summary is not None:
                print(
                    f"    custom source births: mean {summary[0]:.3f} MeV; "
                    f"maximum-bin upper edge {summary[1]:.2f} MeV",
                    file=file,
                )

        leakage_tally = statepoint.get_tally(name="world_neutron_leakage_spectrum")
        leakage_values = _means(leakage_tally, "current")
        if config.geometry.world_height_cm is None:
            leakage_spectrum = leakage_values
        else:
            surface_count = 3
            energy_count = len(edges_mev) - 1
            if len(leakage_values) != surface_count * energy_count:
                raise ValueError(
                    "Cylindrical world leakage tally must contain side, top, "
                    "and bottom surface bins for every energy bin"
                )
            by_surface = [
                leakage_values[index * energy_count : (index + 1) * energy_count]
                for index in range(surface_count)
            ]
            leakage_spectrum = [
                side + top - bottom
                for side, top, bottom in zip(
                    *by_surface, strict=True
                )
            ]
        leakage_per_source = sum(leakage_spectrum)
        expected_leakage = leakage_per_source * source_histories
        net_neutrons = expected_leakage - source_histories
        net_percent = 100 * (leakage_per_source - 1)
        print("\n  World-boundary neutron balance:", file=file)
        if calibration is None:
            print(
                f"    Outward leakage: {leakage_per_source:.6g} neutrons/source "
                f"history (about {expected_leakage:,.1f} over this run)",
                file=file,
            )
        else:
            leakage_rate = leakage_per_source * scale
            net_neutrons = (leakage_per_source - 1.0) * scale
            print(
                f"    Outward leakage: {leakage_rate:.6g} neutrons/s "
                f"({leakage_per_source:.6g} per source history)",
                file=file,
            )
        leakage_summary = _spectrum_summary(leakage_spectrum, edges_mev)
        if leakage_summary is not None:
            print(
                f"    Escaping neutron mean energy: {leakage_summary[0]:.3f} MeV; "
                f"maximum-bin upper edge {leakage_summary[1]:.2f} MeV",
                file=file,
            )
        if calibration is None:
            print(
                f"    Escaped minus source primaries: {net_neutrons:+,.1f} "
                f"({net_percent:+.3f}%)",
                file=file,
            )
        else:
            print(
                f"    Escaped minus source primaries: {net_neutrons:+,.6g} "
                f"neutrons/s ({net_percent:+.3f}%)",
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
        production_unit = (
            "particles/s"
            if calibration is not None
            else "particles/source history"
        )
        print(f"\n  Secondary particle production ({production_unit}):", file=file)
        for particle, value in sorted(production_by_particle.items()):
            if particle == "photon" and not config.physics.photon_transport:
                print("    photon: transport disabled in this configuration", file=file)
                continue
            print(f"    {particle}: {value * scale:.6g}", file=file)
        if calibration is not None:
            integrated_npr = (
                1.0 + production_by_particle.get("neutron", 0.0)
            ) * scale
            print(
                "  Total integrated NPR (primary + material-secondary neutrons): "
                f"{integrated_npr:.6g} neutrons/s",
                file=file,
            )

        for component in config.geometry.components:
            if component.material == "void":
                continue
            nuclides = config.materials[component.material].nuclides or {}
            if "He3" not in nuclides:
                continue
            response = statepoint.get_tally(
                name=f"he3_n_p_spectrum_{component.name}"
            )
            reaction_yield = sum(_means(response, "(n,p)"))
            print(
                f"\n  He-3 detector {component.name}: "
                f"(n,p) response={reaction_yield * scale:.6g} "
                + (
                    "reactions/s"
                    if calibration is not None
                    else "reactions/source history"
                ),
                file=file,
            )
            print(
                "    This transport response does not model the 1300 V charge "
                "collection or apply the nominal 50 cps/(n/cm^2/s) calibration.",
                file=file,
            )

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
        ] + [
            component.name
            for component in config.geometry.components
            if component.material != "void"
        ]
        interaction_unit = (
            "per second" if calibration is not None else "per source history"
        )
        print(f"\n  Neutron interaction scores {interaction_unit}:", file=file)
        for material_name in material_names:
            tally = statepoint.get_tally(
                name=f"neutron_interactions_{material_name}"
            )
            rates = [
                (score, _means(tally, score)[0] * scale)
                for score in interaction_scores
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
            ) * scale
            energy_unit = (
                "MeV/s" if calibration is not None else "MeV/source history"
            )
            print(
                f"    {material_name}: {formatted}; "
                f"net scatter energy loss={energy_change_mev:.5g} "
                f"{energy_unit} (bin-midpoint estimate)",
                file=file,
            )
    finally:
        statepoint.close()

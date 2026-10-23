"""Plot geometry, spatial flux, energy bins, detector spectra, and line profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import sys
import warnings

from .config import SimulationConfig
from .diagnostics import calculate_flux_calibration


@dataclass(frozen=True)
class PlotStyle:
    """Central defaults for plot scales, limits, and color schemes."""

    spatial_cmap: str = "gist_ncar"
    energy_cmap: str = "turbo"
    uncertainty_cmap: str = "gnuplot2"
    geometry_cmap: str = "tab20"
    spectrum_cmap: str = "tab10"
    spectrum_color: str = "tab:blue"
    spatial_xscale: str = "linear"
    spatial_yscale: str = "linear"
    spatial_cscale: str = "log"
    energy_cscale: str = "log"
    uncertainty_cscale: str = "log"
    heatmap_min_percentile: float = 0.1
    heatmap_max_percentile: float = 99.9
    spectrum_xscale: str = "linear"
    spectrum_yscale: str = "log"
    marginal_xscale: str = "linear"
    marginal_yscale: str = "log"
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    zlim: tuple[float, float] | None = None
    spatial_clim: tuple[float, float] | None = None
    energy_clim: tuple[float, float] | None = None
    uncertainty_clim: tuple[float, float] | None = None
    spectrum_xlim: tuple[float, float] | None = None
    spectrum_ylim: tuple[float, float] | None = None
    marginal_ylim: tuple[float, float] | None = None


# Edit this block to customize plot appearance. None limits retain the full
# world-radius axes; explicit limits zoom in. "zlim" controls the XZ vertical
# axis, and *_clim values set colorbar limits (vmin, vmax).
PLOT_STYLE = PlotStyle()


def plot_results(
    config: SimulationConfig, statepoint_path: str | Path, output_dir: str | Path
) -> list[Path]:
    """Write plots from a statepoint produced by the configured model."""
    try:
        import matplotlib
        import numpy as np
        import openmc
    except ImportError as exc:
        dependency = exc.name or "a plotting dependency"
        raise RuntimeError(
            f"Could not import {dependency} while plotting with "
            f"{sys.executable}: {exc}. Install the plotting "
            'extra into the active environment with: python -m pip install -e ".[plot]"'
        ) from exc

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Circle, Patch, Rectangle, Wedge

    statepoint_file = Path(statepoint_path).resolve()
    if not statepoint_file.is_file():
        raise FileNotFoundError(f"Statepoint file not found: {statepoint_file}")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    map_tally_names = {
        f"{kind}_{view}": f"{kind}_map_{view}"
        for kind in ("flux", "energy")
        for view in ("xy", "xz")
    }
    expected_names = set(map_tally_names.values())
    component_source_tally = (
        config.custom_source is not None and config.normalization is not None
    )
    expected_names.add(
        "flux_spectrum_source"
        if component_source_tally
        else "flux_spectrum_source_cavity"
    )
    expected_names.update(f"flux_spectrum_{layer.name}" for layer in config.geometry.layers)
    expected_names.update(
        f"survey_spectrum_{component.name}"
        for component in config.geometry.components
        if component.score_spectrum
    )
    expected_names.update(
        f"spectral_probe_{probe.name}" for probe in config.spectral_probes
    )
    if config.normalization is not None:
        expected_names.add("flux_surface_normalization")

    with openmc.StatePoint(statepoint_file) as statepoint:
        calibration = calculate_flux_calibration(statepoint, config)
        flux_scale = (
            calibration.scale_histories_s if calibration is not None else 1.0
        )
        available_names = {tally.name for tally in statepoint.tallies.values()}
        missing = expected_names - available_names
        if missing:
            raise ValueError(
                "Statepoint is missing required tallies: "
                f"{', '.join(sorted(missing))}. Run the current configuration first."
            )
        map_tallies = {
            name: statepoint.get_tally(name=tally_name)
            for name, tally_name in map_tally_names.items()
        }
        mesh_extents, slice_descriptions = _validate_mesh_tallies(
            map_tallies, config, np, openmc
        )

        map_cell_volumes_cm3 = {
            view: (
                (extent[1] - extent[0])
                / config.visualization.mesh_bins
                * (extent[3] - extent[2])
                / config.visualization.mesh_bins
                * config.visualization.slice_thickness_cm
            )
            for view, extent in mesh_extents.items()
        }
        flux_maps = {
            view: _mesh_values(map_tallies[f"flux_{view}"], view, 1, np)[..., 0]
            / map_cell_volumes_cm3[view]
            * flux_scale
            for view in ("xy", "xz")
        }
        flux_std_devs = {
            view: _mesh_values(
                map_tallies[f"flux_{view}"], view, 1, np, value="std_dev"
            )[..., 0]
            / map_cell_volumes_cm3[view]
            * flux_scale
            for view in ("xy", "xz")
        }
        energy_maps = {
            view: _mesh_values(
                map_tallies[f"energy_{view}"],
                view,
                len(config.visualization.energy_bins_mev) - 1,
                np,
            )
            / map_cell_volumes_cm3[view]
            * flux_scale
            for view in ("xy", "xz")
        }
        spectra = {}
        survey_spectra = {}
        he3_detector_spectra = {}
        he3_axial_capture_responses = {}
        source_birth_spectrum = None
        if component_source_tally:
            spectra["Reactor vacuum cavity (fusion source region)"] = (
                _spectrum_values(
                    statepoint.get_tally(name="flux_spectrum_source"), np
                )
                * flux_scale
            )
        elif "flux_spectrum_source" in available_names:
            spectra["Source volume"] = _spectrum_values(
                statepoint.get_tally(name="flux_spectrum_source"), np
            ) * flux_scale
            spectra["Source cavity outside source volume"] = _spectrum_values(
                statepoint.get_tally(name="flux_spectrum_source_cavity"), np
            ) * flux_scale
        else:
            spectra[
                "Source cavity (legacy tally; includes source volume)"
            ] = _spectrum_values(
                statepoint.get_tally(name="flux_spectrum_source_cavity"), np
            ) * flux_scale
            warnings.warn(
                "This statepoint predates the separate source-volume tally; "
                "its source spectrum cannot be separated from the cavity. "
                "Rerun `iec-neutrons run <config.yaml>` for a source-only spectrum.",
                UserWarning,
                stacklevel=2,
            )
        for layer in config.geometry.layers:
            spectra[layer.name] = _spectrum_values(
                statepoint.get_tally(name=f"flux_spectrum_{layer.name}"), np
            ) * flux_scale
        for component in config.geometry.components:
            if component.material == "void":
                continue
            material = config.materials[component.material]
            if "He3" not in (material.nuclides or {}):
                continue

            flux_tally_name = f"flux_spectrum_{component.name}"
            if flux_tally_name in available_names:
                spectra[f"{component.name} (He-3 gas)"] = _spectrum_values(
                    statepoint.get_tally(name=flux_tally_name), np
                ) * flux_scale

            neutron_tally_name = f"he3_neutron_spectrum_{component.name}"
            response_tally_name = f"he3_n_p_spectrum_{component.name}"
            missing_he3_tallies = {
                name
                for name in (neutron_tally_name, response_tally_name)
                if name not in available_names
            }
            if missing_he3_tallies:
                warnings.warn(
                    "Statepoint is missing He-3 spectrum tallies: "
                    f"{', '.join(sorted(missing_he3_tallies))}. Rerun the "
                    "simulation with the current model to generate the "
                    "combined spectrum with low-energy resolution.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            neutron_tally = statepoint.get_tally(name=neutron_tally_name)
            response_tally = statepoint.get_tally(name=response_tally_name)
            neutron_energy_filter = next(
                (
                    item
                    for item in neutron_tally.filters
                    if isinstance(item, openmc.EnergyFilter)
                ),
                None,
            )
            response_energy_filter = next(
                (
                    item
                    for item in response_tally.filters
                    if isinstance(item, openmc.EnergyFilter)
                ),
                None,
            )
            if neutron_energy_filter is None or response_energy_filter is None:
                raise ValueError(
                    f"He-3 spectrum tallies for {component.name!r} "
                    "must both have energy filters"
                )
            neutron_bins_ev = np.asarray(neutron_energy_filter.bins, dtype=float)
            response_bins_ev = np.asarray(response_energy_filter.bins, dtype=float)
            if (
                neutron_bins_ev.ndim != 2
                or neutron_bins_ev.shape[1] != 2
                or response_bins_ev.ndim != 2
                or response_bins_ev.shape[1] != 2
            ):
                raise ValueError(
                    f"He-3 spectrum tallies for {component.name!r} "
                    "have unexpected energy bins"
                )
            neutron_edges_mev = np.concatenate(
                (neutron_bins_ev[:1, 0], neutron_bins_ev[:, 1])
            ) / 1.0e6
            response_edges_mev = np.concatenate(
                (response_bins_ev[:1, 0], response_bins_ev[:, 1])
            ) / 1.0e6
            if not np.allclose(neutron_edges_mev, response_edges_mev):
                raise ValueError(
                    f"He-3 neutron and capture tallies for {component.name!r} "
                    "have different energy bins"
                )
            if not (
                np.isclose(neutron_edges_mev[0], 0.0)
                and np.isclose(neutron_edges_mev[-1], config.tally.energy_bins_mev[-1])
                and np.any(np.isclose(neutron_edges_mev, 1.0e-6))
                and np.any(np.isclose(neutron_edges_mev, 0.1))
            ):
                raise ValueError(
                    f"He-3 spectrum tallies for {component.name!r} do not "
                    "cover 1 eV through 0.1 MeV with the expected boundaries"
                )
            neutron_flux = np.asarray(
                neutron_tally.get_values(scores=["flux"], value="mean"),
                dtype=float,
            ).reshape(-1) * flux_scale
            capture_response = np.asarray(
                response_tally.get_values(scores=["(n,p)"], value="mean"),
                dtype=float,
            ).reshape(-1) * flux_scale
            if (
                neutron_flux.size != len(neutron_edges_mev) - 1
                or capture_response.size != len(response_edges_mev) - 1
            ):
                raise ValueError(
                    f"He-3 spectrum tallies for {component.name!r} have "
                    "unexpected numbers of energy bins"
                )
            he3_detector_spectra[component.name] = (
                neutron_edges_mev,
                neutron_flux,
                capture_response,
            )
            axial_tally_name = f"he3_n_p_axial_energy_{component.name}"
            if axial_tally_name not in available_names:
                warnings.warn(
                    f"Statepoint is missing {axial_tally_name!r}; rerun the "
                    "simulation with the current model to create the "
                    "position-resolved He-3 capture heatmap.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            axial_tally = statepoint.get_tally(name=axial_tally_name)
            axial_energy_filter = next(
                (
                    item
                    for item in axial_tally.filters
                    if isinstance(item, openmc.EnergyFilter)
                ),
                None,
            )
            axial_mesh_filter = next(
                (
                    item
                    for item in axial_tally.filters
                    if isinstance(item, openmc.MeshFilter)
                ),
                None,
            )
            if axial_energy_filter is None or axial_mesh_filter is None:
                raise ValueError(
                    f"Axial He-3 capture tally for {component.name!r} "
                    "must have energy and mesh filters"
                )
            axial_bins_ev = np.asarray(axial_energy_filter.bins, dtype=float)
            axial_edges_mev = np.concatenate(
                (axial_bins_ev[:1, 0], axial_bins_ev[:, 1])
            ) / 1.0e6
            axial_bin_count = int(axial_mesh_filter.mesh.dimension[2])
            center_cm = np.asarray(component.center_cm, dtype=float)
            half_extents_cm = np.asarray(component.half_extents_cm, dtype=float)
            if (
                not np.allclose(axial_edges_mev, response_edges_mev)
                or axial_bin_count != config.visualization.he3_axial_bins
                or not np.allclose(
                    axial_mesh_filter.mesh.lower_left,
                    center_cm - half_extents_cm,
                )
                or not np.allclose(
                    axial_mesh_filter.mesh.upper_right,
                    center_cm + half_extents_cm,
                )
            ):
                raise ValueError(
                    f"Axial He-3 capture tally for {component.name!r} has "
                    "unexpected energy, height, or detector bounds"
                )
            axial_capture = np.asarray(
                axial_tally.get_values(scores=["(n,p)"], value="mean"),
                dtype=float,
            ).reshape(-1) * flux_scale
            expected_size = axial_bin_count * (len(axial_edges_mev) - 1)
            if axial_capture.size != expected_size:
                raise ValueError(
                    f"Axial He-3 capture tally for {component.name!r} has "
                    "an unexpected number of bins"
                )
            z_edges_cm = np.linspace(
                -half_extents_cm[2],
                half_extents_cm[2],
                axial_bin_count + 1,
            )
            response_density = axial_capture.reshape(
                axial_bin_count, len(axial_edges_mev) - 1
            )
            response_density /= np.diff(z_edges_cm)[:, None]
            response_density /= np.diff(axial_edges_mev)[None, :]
            he3_axial_capture_responses[component.name] = (
                axial_edges_mev,
                z_edges_cm,
                response_density,
            )
        for component in config.geometry.components:
            if not component.score_spectrum:
                continue
            survey_tally = statepoint.get_tally(
                name=f"survey_spectrum_{component.name}"
            )
            energy_filter = next(
                (
                    item
                    for item in survey_tally.filters
                    if isinstance(item, openmc.EnergyFilter)
                ),
                None,
            )
            if energy_filter is None:
                raise ValueError(
                    f"Survey spectrum tally for {component.name!r} has no energy filter"
                )
            bins_ev = np.asarray(energy_filter.bins, dtype=float)
            edges_mev = np.concatenate((bins_ev[:1, 0], bins_ev[:, 1])) / 1.0e6
            if component.shape == "box":
                volume_cm3 = float(np.prod(component.dimensions_cm))
            else:
                component_radius_cm, component_length_cm = component.dimensions_cm
                volume_cm3 = np.pi * component_radius_cm**2 * component_length_cm
            values = np.asarray(
                survey_tally.get_values(scores=["flux"], value="mean"),
                dtype=float,
            ).reshape(-1)
            if values.size != len(edges_mev) - 1:
                raise ValueError(
                    f"Survey spectrum tally for {component.name!r} has an "
                    "unexpected number of energy bins"
                )
            survey_spectra[component.name] = (
                edges_mev,
                values / volume_cm3 / np.diff(edges_mev) * flux_scale,
            )
        for probe in config.spectral_probes:
            probe_tally = statepoint.get_tally(
                name=f"spectral_probe_{probe.name}"
            )
            energy_filter = next(
                (
                    item
                    for item in probe_tally.filters
                    if isinstance(item, openmc.EnergyFilter)
                ),
                None,
            )
            if energy_filter is None:
                raise ValueError(
                    f"Spectral probe {probe.name!r} has no energy filter"
                )
            bins_ev = np.asarray(energy_filter.bins, dtype=float)
            edges_mev = np.concatenate((bins_ev[:1, 0], bins_ev[:, 1])) / 1.0e6
            values = np.asarray(
                probe_tally.get_values(scores=["flux"], value="mean"),
                dtype=float,
            ).reshape(-1)
            volume_cm3 = float(np.prod(probe.size_cm))
            if values.size != len(edges_mev) - 1:
                raise ValueError(
                    f"Spectral probe {probe.name!r} has an unexpected "
                    "number of energy bins"
                )
            survey_spectra[probe.name] = (
                edges_mev,
                values / volume_cm3 / np.diff(edges_mev) * flux_scale,
            )
        if config.custom_source is not None:
            source_spectrum_path = (
                statepoint_file.parent / "source_birth_spectrum.csv"
            )
            if not source_spectrum_path.is_file():
                raise ValueError(
                    "Custom-source birth-energy data is missing beside the statepoint; "
                    "rerun with the current compiled-source runner."
                )
            with source_spectrum_path.open(
                newline="", encoding="utf-8"
            ) as source_file:
                source_rows = list(csv.DictReader(source_file))
            if not source_rows:
                raise ValueError(
                    f"Custom-source birth-energy file is empty: {source_spectrum_path}"
                )
            source_edges = np.asarray(
                [float(source_rows[0]["energy_lower_mev"])]
                + [float(row["energy_upper_mev"]) for row in source_rows],
                dtype=float,
            )
            source_probability = np.asarray(
                [float(row["probability_per_source_neutron"]) for row in source_rows],
                dtype=float,
            )
            if (
                source_probability.size != len(config.tally.energy_bins_mev) - 1
                or not np.allclose(source_edges, config.tally.energy_bins_mev)
                or np.any(~np.isfinite(source_probability))
                or np.any(source_probability < 0)
                or not np.isclose(source_probability.sum(), 1.0)
            ):
                raise ValueError(
                    "Custom-source birth-energy data does not match the configured "
                    "energy bins or is not normalized."
                )
            source_birth_spectrum = (
                source_edges,
                source_probability / np.diff(source_edges),
            )
        detector_available = "detector_energy_spectrum" in available_names
        detector_mean = None
        detector_std_dev = None
        if detector_available:
            detector_tally = statepoint.get_tally(name="detector_energy_spectrum")
            _validate_detector_tally(detector_tally, config, np, openmc)
            detector_volume_cm3 = (
                config.detector.width_x_cm
                * config.detector.width_y_cm
                * config.detector.thickness_z_cm
            )
            detector_mean = (
                _spectrum_values(detector_tally, np)
                / detector_volume_cm3
                * flux_scale
            )
            detector_std_dev = (
                _spectrum_values(detector_tally, np, value="std_dev")
                / detector_volume_cm3
                * flux_scale
            )
        elif config.detector.enabled:
            warnings.warn(
                "This statepoint has no detector_energy_spectrum tally. "
                "Spatial and geometry plots will mark the configured detector, "
                "but its spectrum requires rerunning the simulation.",
                UserWarning,
                stacklevel=2,
            )
    radius = config.geometry.world_radius_cm
    total_edges = np.asarray(config.tally.energy_bins_mev)
    map_edges = np.asarray(config.visualization.energy_bins_mev)
    normalized = calibration is not None
    flux_units = (
        "n cm$^{-2}$ s$^{-1}$"
        if normalized
        else "cm$^{-2}$ per source neutron"
    )
    track_length_units = (
        "cm s$^{-1}$" if normalized else "cm per source neutron"
    )
    paths: list[Path] = []

    for view in ("xy", "xz"):
        if view == "xy":
            extent = mesh_extents[view]
            horizontal_edges = _mesh_edges(
                (extent[1] - extent[0]) / 2, config.visualization.mesh_bins
            )
            vertical_edges = _mesh_edges(
                (extent[3] - extent[2]) / 2, config.visualization.mesh_bins
            )
            vertical_edges += (extent[2] + extent[3]) / 2
            horizontal_edges += (extent[0] + extent[1]) / 2
            slice_description = slice_descriptions[view]
        else:
            extent = mesh_extents[view]
            horizontal_edges = _mesh_edges(
                (extent[1] - extent[0]) / 2, config.visualization.mesh_bins
            )
            vertical_edges = _mesh_edges(
                (extent[3] - extent[2]) / 2, config.visualization.mesh_bins
            )
            vertical_edges += (extent[2] + extent[3]) / 2
            horizontal_edges += (extent[0] + extent[1]) / 2
            slice_description = slice_descriptions[view]
        orientation = "Top-down (XY)" if view == "xy" else "Side-on cross-section (XZ)"

        total_flux = flux_maps[view]
        relative_error = np.divide(
            flux_std_devs[view],
            total_flux,
            out=np.full_like(total_flux, np.nan, dtype=float),
            where=total_flux > 0,
        )
        paths.append(
            _plot_spatial_map(
                total_flux.T,
                extent,
                f"{orientation}: total flux, {slice_description}",
                f"Flux ({flux_units})",
                destination / f"flux_{view}.png",
                config,
                view,
                LogNorm,
                Circle,
                Rectangle,
                plt,
                PLOT_STYLE,
                Normalize,
                show_contours=True,
            )
        )
        paths.append(
            _plot_spatial_map(
                relative_error.T,
                extent,
                f"{orientation}: relative tally uncertainty",
                "Relative uncertainty (standard deviation / mean)",
                destination / f"relative_uncertainty_{view}.png",
                config,
                view,
                None,
                Circle,
                Rectangle,
                plt,
                PLOT_STYLE,
                Normalize,
                show_contours=False,
            )
        )
        paths.append(
            _plot_energy_map(
                energy_maps[view],
                map_edges,
                extent,
                orientation,
                slice_description,
                destination / f"energy_bins_{view}.png",
                config,
                view,
                LogNorm,
                Circle,
                Rectangle,
                plt,
                PLOT_STYLE,
                Normalize,
            )
        )
        paths.append(
            _plot_marginals(
                total_flux,
                horizontal_edges,
                vertical_edges,
                view,
                config,
                destination / f"marginals_{view}.png",
                plt,
                PLOT_STYLE,
                radius,
            )
        )

    paths.append(
        _plot_spectrum(
            spectra,
            total_edges,
            "Region-integrated energy spectra",
            f"Track length ({track_length_units})",
            destination / "energy_spectra_by_region.png",
            plt,
            PLOT_STYLE,
        )
    )
    if survey_spectra:
        paths.append(
            _plot_survey_spectra(
                survey_spectra,
                destination / "survey_region_neutron_spectra.png",
                plt,
                np,
                normalized,
                source_birth_spectrum,
            )
        )

    components_by_name = {
        component.name: component for component in config.geometry.components
    }
    for component_name, (edges, neutron_flux, capture_response) in (
        he3_detector_spectra.items()
    ):
        component = components_by_name[component_name]
        if component.shape == "box":
            component_volume_cm3 = float(np.prod(component.dimensions_cm))
        else:
            component_radius_cm, component_length_cm = component.dimensions_cm
            component_volume_cm3 = (
                np.pi * component_radius_cm**2 * component_length_cm
            )
        bin_widths = np.diff(edges)
        paths.append(
            _plot_he3_detector_spectrum(
                neutron_flux / component_volume_cm3 / bin_widths,
                capture_response / bin_widths,
                edges,
                component_name,
                "log",
                destination
                / f"he3_detector_spectrum_{component_name}_log.png",
                plt,
                PLOT_STYLE,
                normalized,
            )
        )
        if component_name in he3_axial_capture_responses:
            energy_edges, z_edges, response_density = (
                he3_axial_capture_responses[component_name]
            )
            paths.append(
                _plot_he3_capture_heatmap(
                    energy_edges,
                    z_edges,
                    response_density,
                    component_name,
                    destination
                    / f"he3_capture_response_{component_name}_height_energy.png",
                    plt,
                    LogNorm,
                    PLOT_STYLE,
                    normalized,
                )
            )

    if detector_mean is not None and detector_std_dev is not None:
        paths.append(
            _plot_detector_spectrum(
                detector_mean,
                detector_std_dev,
                total_edges,
                config,
                destination / "detector_energy_spectrum.png",
                plt,
                np,
                PLOT_STYLE,
            )
        )
    paths.append(
        _plot_geometry(
            config,
            "xy",
            destination / "geometry_xy.png",
            plt,
            Circle,
            Patch,
            Wedge,
            Rectangle,
            PLOT_STYLE,
        )
    )
    paths.append(
        _plot_geometry(
            config,
            "xz",
            destination / "geometry_xz.png",
            plt,
            Circle,
            Patch,
            Wedge,
            Rectangle,
            PLOT_STYLE,
        )
    )
    return paths


def _mesh_edges(half_width: float, bins: int):
    import numpy as np

    return np.linspace(-half_width, half_width, bins + 1)


def _mesh_values(tally, view: str, energy_groups: int, np, value: str = "mean"):
    data = tally.get_reshaped_data(value=value, expand_dims=True)
    if len(tally.filters) == 1:
        mesh_values = data[..., 0, 0]
        result = (
            mesh_values[:, :, 0, np.newaxis]
            if view == "xy"
            else mesh_values[:, 0, :, np.newaxis]
        )
    elif view == "xy":
        result = data[:, :, 0, :, 0, 0]
    else:
        result = data[:, 0, :, :, 0, 0]
    if result.shape[-1] != energy_groups:
        raise ValueError(
            f"Tally {tally.name!r} has {result.shape[-1]} energy groups; "
            f"expected {energy_groups}"
        )
    return np.asarray(result, dtype=float)


def _spectrum_values(tally, np, value: str = "mean"):
    data = tally.get_reshaped_data(value=value)
    if data.ndim != 4 or data.shape[0] != 1 or data.shape[2:] != (1, 1):
        raise ValueError(f"Tally {tally.name!r} does not have a single volume and energy filter")
    return np.asarray(data[0, :, 0, 0], dtype=float)


def _validate_mesh_tallies(
    tallies, config: SimulationConfig, np, openmc
) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, str]]:
    radius = config.geometry.world_radius_cm
    half_height = (
        config.geometry.world_height_cm / 2
        if config.geometry.world_height_cm is not None
        else radius
    )
    axis_limits = (radius, radius, half_height)
    half_slice = config.visualization.slice_thickness_cm / 2
    map_edges_ev = np.asarray(config.visualization.energy_bins_mev) * 1.0e6
    expected_energy_bins = np.column_stack((map_edges_ev[:-1], map_edges_ev[1:]))
    extents = {}
    slices = {}
    for view in ("xy", "xz"):
        expected_dimensions = (
            (config.visualization.mesh_bins, config.visualization.mesh_bins, 1)
            if view == "xy"
            else (config.visualization.mesh_bins, 1, config.visualization.mesh_bins)
        )
        view_mesh = None
        for kind in ("flux", "energy"):
            tally = tallies[f"{kind}_{view}"]
            mesh_filter = next(
                (item for item in tally.filters if isinstance(item, openmc.MeshFilter)),
                None,
            )
            if mesh_filter is None:
                raise ValueError(f"Statepoint tally {tally.name!r} has no mesh filter")
            mesh = mesh_filter.mesh
            if tuple(int(item) for item in mesh.dimension) != expected_dimensions:
                raise ValueError(f"Statepoint tally {tally.name!r} has unexpected mesh dimensions")
            lower = np.asarray(mesh.lower_left, dtype=float)
            upper = np.asarray(mesh.upper_right, dtype=float)
            plane_indices = (0, 1) if view == "xy" else (0, 2)
            if any(
                lower[index] >= 0
                or upper[index] <= 0
                or max(abs(lower[index]), abs(upper[index]))
                > axis_limits[index] + 1e-8
                or not np.isclose(lower[index], -upper[index])
                for index in plane_indices
            ):
                raise ValueError(
                    f"Statepoint tally {tally.name!r} mesh does not cover a centered "
                    "symmetric region within the configured world radius"
                )
            if view_mesh is not None and (
                not np.allclose(mesh.lower_left, view_mesh.lower_left)
                or not np.allclose(mesh.upper_right, view_mesh.upper_right)
            ):
                raise ValueError(
                    f"Statepoint {view.upper()} flux and energy meshes have different bounds"
                )
            view_mesh = mesh
            if kind == "energy":
                energy_filter = next(
                    (item for item in tally.filters if isinstance(item, openmc.EnergyFilter)),
                    None,
                )
                if energy_filter is None or not np.allclose(
                    energy_filter.bins, expected_energy_bins, rtol=1e-12, atol=1e-8
                ):
                    raise ValueError(f"Statepoint tally {tally.name!r} has unexpected energy bins")
        assert view_mesh is not None
        lower = np.asarray(view_mesh.lower_left, dtype=float)
        upper = np.asarray(view_mesh.upper_right, dtype=float)
        if view == "xy":
            extents[view] = (lower[0], upper[0], lower[1], upper[1])
            omitted_lower, omitted_upper = lower[2], upper[2]
            omitted_axis = "z"
        else:
            extents[view] = (lower[0], upper[0], lower[2], upper[2])
            omitted_lower, omitted_upper = lower[1], upper[1]
            omitted_axis = "y"
        thickness = omitted_upper - omitted_lower
        if np.isclose(thickness, 2 * half_slice) and np.isclose(
            omitted_lower, -half_slice
        ):
            slices[view] = (
                f"central {omitted_axis} slab, "
                f"{config.visualization.slice_thickness_cm:g} cm thick"
            )
        else:
            slices[view] = (
                f"{omitted_axis} mesh bin [{omitted_lower:g}, {omitted_upper:g}] cm"
            )
    return extents, slices


def _validate_detector_tally(tally, config: SimulationConfig, np, openmc) -> None:
    mesh_filter = next(
        (item for item in tally.filters if isinstance(item, openmc.MeshFilter)), None
    )
    energy_filter = next(
        (item for item in tally.filters if isinstance(item, openmc.EnergyFilter)), None
    )
    if mesh_filter is None or energy_filter is None:
        raise ValueError("Statepoint detector tally is missing its mesh or energy filter")
    mesh = mesh_filter.mesh
    lower = (
        config.detector.x_cm - config.detector.width_x_cm / 2,
        config.detector.y_cm - config.detector.width_y_cm / 2,
        config.detector.z_cm - config.detector.thickness_z_cm / 2,
    )
    upper = (
        config.detector.x_cm + config.detector.width_x_cm / 2,
        config.detector.y_cm + config.detector.width_y_cm / 2,
        config.detector.z_cm + config.detector.thickness_z_cm / 2,
    )
    if not (
        tuple(int(item) for item in mesh.dimension) == (1, 1, 1)
        and np.allclose(mesh.lower_left, lower)
        and np.allclose(mesh.upper_right, upper)
    ):
        raise ValueError("Statepoint detector does not match its configured position and dimensions")
    edges_ev = np.asarray(config.tally.energy_bins_mev) * 1.0e6
    expected_edges = np.column_stack((edges_ev[:-1], edges_ev[1:]))
    if not np.allclose(energy_filter.bins, expected_edges, rtol=1e-12, atol=1e-8):
        raise ValueError("Statepoint detector energy bins do not match the supplied configuration")


def _geometry_outlines(
    config: SimulationConfig, view: str, Circle, Rectangle, axis
):
    if view == "xz" and config.geometry.world_height_cm is not None:
        world_boundary = Rectangle(
            (-config.geometry.world_radius_cm, -config.geometry.world_height_cm / 2),
            2 * config.geometry.world_radius_cm,
            config.geometry.world_height_cm,
            fill=False,
            edgecolor="white",
            linewidth=1.0,
            linestyle=":",
        )
    else:
        world_boundary = Circle(
            (0, 0),
            config.geometry.world_radius_cm,
            fill=False,
            edgecolor="white",
            linewidth=1.0,
            linestyle=":",
        )
    if view == "xy":
        outlines = [
            Circle(
                (0, 0),
                config.geometry.cavity_radius_cm,
                fill=False,
                edgecolor="white",
                linewidth=1.0,
            )
        ]
        outlines.extend(
            Circle(
                (0, 0),
                layer.radius_cm,
                fill=False,
                edgecolor="white",
                linewidth=0.8,
            )
            for layer in config.geometry.layers
        )
        outlines.append(
            Circle(
                (0, 0),
                config.source.radius_cm,
                fill=False,
                edgecolor="cyan",
                linewidth=1.0,
                linestyle="--",
            )
        )
        outlines.extend(
            _component_patch(
                component,
                view,
                Circle,
                Rectangle,
                fill=False,
                edgecolor="white",
                linewidth=0.9,
            )
            for component in config.geometry.components
        )
        if config.geometry.air_sphere_radius_cm is not None:
            outlines.append(
                Circle(
                    (0, 0),
                    config.geometry.air_sphere_radius_cm,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.0,
                    linestyle="--",
                )
            )
        outlines.append(world_boundary)
        return outlines

    outlines = [
        Rectangle(
            (-config.geometry.cavity_radius_cm, -config.geometry.cavity_height_cm / 2),
            2 * config.geometry.cavity_radius_cm,
            config.geometry.cavity_height_cm,
            fill=False,
            edgecolor="white",
            linewidth=1.0,
        )
    ]
    for layer in config.geometry.layers:
        layer_outlines = (
            Rectangle(
                (-layer.radius_cm, -layer.height_cm / 2),
                layer.thickness_cm,
                layer.height_cm,
                fill=False,
                edgecolor="white",
                linewidth=0.7,
            ),
            Rectangle(
                (layer.radius_cm - layer.thickness_cm, -layer.height_cm / 2),
                layer.thickness_cm,
                layer.height_cm,
                fill=False,
                edgecolor="white",
                linewidth=0.7,
            ),
        )
        if config.geometry.air_sphere_radius_cm is not None:
            clip = Circle(
                (0, 0),
                config.geometry.air_sphere_radius_cm,
                transform=axis.transData,
            )
            for outline in layer_outlines:
                outline.set_clip_path(clip)
        outlines.extend(layer_outlines)
    outlines.append(
        Rectangle(
            (-config.source.radius_cm, -config.source.height_cm / 2),
            2 * config.source.radius_cm,
            config.source.height_cm,
            fill=False,
            edgecolor="cyan",
            linewidth=1.0,
            linestyle="--",
        )
    )
    outlines.extend(
        _component_patch(
            component,
            view,
            Circle,
            Rectangle,
            fill=False,
            edgecolor="white",
            linewidth=0.9,
        )
        for component in config.geometry.components
    )
    if config.geometry.air_sphere_radius_cm is not None:
        outlines.append(
            Circle(
                (0, 0),
                config.geometry.air_sphere_radius_cm,
                fill=False,
                edgecolor="cyan",
                linewidth=1.0,
                linestyle="--",
            )
        )
    outlines.append(world_boundary)
    return outlines

def _component_patch(component, view, Circle, Rectangle, **style):
    horizontal = 0
    vertical = 1 if view == "xy" else 2
    center = component.center_cm
    if component.shape == "box":
        width = component.dimensions_cm[horizontal]
        height = component.dimensions_cm[vertical]
        return Rectangle(
            (center[horizontal] - width / 2, center[vertical] - height / 2),
            width,
            height,
            **style,
        )

    radius, length = component.dimensions_cm
    axis_index = {"x": 0, "y": 1, "z": 2}[component.axis]
    if axis_index not in {horizontal, vertical}:
        return Circle((center[horizontal], center[vertical]), radius, **style)
    width = length if axis_index == horizontal else 2 * radius
    height = length if axis_index == vertical else 2 * radius
    return Rectangle(
        (center[horizontal] - width / 2, center[vertical] - height / 2),
        width,
        height,
        **style,
    )
def _add_dotted_contours(axis, data, extent, np) -> None:
    field = np.ma.masked_where(~np.isfinite(data) | (data <= 0), data)
    positive = field.compressed()
    if positive.size < 2:
        return
    lower, upper = np.percentile(positive, [10, 90])
    if not upper > lower:
        return
    levels = np.unique(np.geomspace(lower, upper, 5))[1:-1]
    if not levels.size:
        return
    x = np.linspace(extent[0], extent[1], data.shape[1])
    y = np.linspace(extent[2], extent[3], data.shape[0])
    contours = axis.contour(
        x,
        y,
        field,
        levels=levels,
        colors="white",
        linewidths=0.75,
        linestyles=":",
        alpha=0.9,
    )
    labels = axis.clabel(
        contours,
        fmt=lambda level: f"{level:.1e}",
        inline=True,
        fontsize=6,
        colors="black",
    )
    for label in labels:
        label.set_bbox(
            {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5}
        )


def _detector_overlay(axis, config: SimulationConfig, view: str, Rectangle) -> None:
    if not config.detector.enabled:
        return
    detector = config.detector
    if view == "xy":
        x, y = detector.x_cm, detector.y_cm
        width, height = detector.width_x_cm, detector.width_y_cm
        vertical_axis = "y"
    else:
        x, y = detector.x_cm, detector.z_cm
        width, height = detector.width_x_cm, detector.thickness_z_cm
        vertical_axis = "z"
    axis.add_patch(
        Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            fill=False,
            edgecolor="magenta",
            linewidth=2.0,
            linestyle="--",
            zorder=8,
        )
    )
    axis.scatter(
        [x],
        [y],
        marker="x",
        s=48,
        linewidths=1.8,
        color="magenta",
        zorder=9,
    )
    axis.annotate(
        f"Detector ({vertical_axis}={y:g} cm)",
        (x, y),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=7,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "magenta", "alpha": 0.85, "pad": 1.5},
        zorder=10,
    )


def _set_spatial_axes(
    axis,
    view: str,
    style: PlotStyle,
    world_radius: float,
    world_half_height: float | None = None,
) -> None:
    axis.set_xscale(style.spatial_xscale)
    axis.set_yscale(style.spatial_yscale)
    axis.set_xlim(*(style.xlim or (-world_radius, world_radius)))
    vertical_limit = style.ylim if view == "xy" else style.zlim
    axis.set_ylim(
        *(vertical_limit or (
            (-world_half_height, world_half_height)
            if view == "xz" and world_half_height is not None
            else (-world_radius, world_radius)
        ))
    )


def _plot_spatial_map(
    data,
    extent,
    title: str,
    colorbar_label: str,
    path: Path,
    config: SimulationConfig,
    view: str,
    LogNorm,
    Circle,
    Rectangle,
    plt,
    style: PlotStyle,
    Normalize,
    *,
    show_contours: bool,
) -> Path:
    import numpy as np

    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    masked = np.ma.masked_invalid(data)
    positive = masked.compressed()
    positive = positive[positive > 0]
    low = _heatmap_minimum(positive, style.heatmap_min_percentile, np)
    high = _heatmap_maximum(positive, style.heatmap_max_percentile, np) if positive.size else 1.0

    color_scale = style.spatial_cscale if LogNorm is not None else style.uncertainty_cscale
    color_limits = style.spatial_clim if LogNorm is not None else style.uncertainty_clim
    if color_limits is not None:
        low, high = color_limits
    if high <= low:
        high = low + 1.0
    color_map = style.spatial_cmap if LogNorm is not None else style.uncertainty_cmap
    options = {"cmap": color_map}
    if color_scale == "log" and LogNorm is not None and positive.size:
        options["norm"] = LogNorm(vmin=low, vmax=high)
    else:
        options["norm"] = Normalize(vmin=low, vmax=high)
    image = axis.imshow(masked, origin="lower", extent=extent, aspect="equal", **options)
    for outline in _geometry_outlines(config, view, Circle, Rectangle, axis):
        axis.add_patch(outline)
    if show_contours:
        _add_dotted_contours(axis, masked, extent, np)
    _detector_overlay(axis, config, view, Rectangle)
    axis.set_xlabel("x (cm)")
    axis.set_ylabel("y (cm)" if view == "xy" else "z (cm)")
    axis.set_title(title)
    _set_spatial_axes(
        axis,
        view,
        style,
        config.geometry.world_radius_cm,
        config.geometry.world_height_cm / 2
        if config.geometry.world_height_cm is not None
        else None,
    )
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.savefig(str(path), dpi=180)
    plt.close(figure)
    return path


def _plot_energy_map(
    data,
    edges,
    extent,
    orientation: str,
    slice_description: str,
    path: Path,
    config: SimulationConfig,
    view: str,
    LogNorm,
    Circle,
    Rectangle,
    plt,
    style: PlotStyle,
    Normalize,
) -> Path:
    import numpy as np

    groups = data.shape[-1]
    positives = data[np.isfinite(data) & (data > 0)]
    rows = (groups + 2) // 3
    figure, axes = plt.subplots(
        rows,
        3,
        figsize=(14, 4.2 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    norm = None
    if positives.size:
        low = _heatmap_minimum(positives, style.heatmap_min_percentile, np)
        high = _heatmap_maximum(positives, style.heatmap_max_percentile, np)
        if style.energy_clim is not None:
            low, high = style.energy_clim
        elif high <= low:
            high = low + 1.0
        norm = (
            LogNorm(vmin=low, vmax=high)
            if style.energy_cscale == "log" and LogNorm is not None
            else Normalize(vmin=low, vmax=high)
        )

    image = None
    for index, axis in enumerate(axes.flat):
        if index >= groups:
            axis.set_visible(False)
            continue
        field = np.ma.masked_invalid(data[:, :, index].T)
        if norm is not None:
            field = np.ma.masked_where(field <= 0, field)
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=style.energy_cmap,
            norm=norm,
        )
        for outline in _geometry_outlines(config, view, Circle, Rectangle, axis):
            axis.add_patch(outline)
        _detector_overlay(axis, config, view, Rectangle)
        _set_spatial_axes(
            axis,
            view,
            style,
            config.geometry.world_radius_cm,
            config.geometry.world_height_cm / 2
            if config.geometry.world_height_cm is not None
            else None,
        )
        axis.set_title(f"{edges[index]:g}-{edges[index + 1]:g} MeV")
        axis.set_xlabel("x (cm)")
        axis.set_ylabel("y (cm)" if view == "xy" else "z (cm)")

    figure.suptitle(f"{orientation}: flux by energy bin, {slice_description}")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label=(
                "Flux (n cm$^{-2}$ s$^{-1}$)"
                if config.normalization is not None
                else "Flux (cm$^{-2}$ per source neutron)"
            ),
            shrink=0.85,
        )
    figure.savefig(str(path), dpi=180)
    plt.close(figure)
    return path


def _heatmap_minimum(positive_values, percentile: float, np) -> float:
    if not 0 <= percentile < 100:
        raise ValueError("PLOT_STYLE.heatmap_min_percentile must be in [0, 100)")
    if not positive_values.size:
        return 0.0
    return float(np.percentile(positive_values, percentile))

def _heatmap_maximum(positive_values, percentile: float, np) -> float:
    if not 0 <= percentile < 100:
        raise ValueError("PLOT_STYLE.heatmap_max_percentile must be in [0, 100)")
    if not positive_values.size:
        return 1.0
    return float(np.percentile(positive_values, percentile))


def _nearest_index(edges, coordinate: float) -> int:
    import numpy as np

    centers = (edges[:-1] + edges[1:]) / 2
    return int(np.abs(centers - coordinate).argmin())


def _plot_marginals(
    data, horizontal_edges, vertical_edges, view, config, path, plt, style, world_radius
) -> Path:
    import numpy as np

    if view == "xy":
        first_index = _nearest_index(vertical_edges, config.visualization.marginal_y_cm)
        second_index = _nearest_index(horizontal_edges, config.visualization.marginal_x_cm)
        first_coordinate = (vertical_edges[first_index] + vertical_edges[first_index + 1]) / 2
        second_coordinate = (horizontal_edges[second_index] + horizontal_edges[second_index + 1]) / 2
        first_values = data[:, first_index]
        second_values = data[second_index, :]
        second_axis_label = "y (cm)"
        first_description = f"at y = {first_coordinate:g} cm"
        second_description = f"at x = {second_coordinate:g} cm"
    else:
        vertical_world_extent = (
            config.geometry.world_height_cm / 2
            if config.geometry.world_height_cm is not None
            else world_radius
        )
        first_index = _nearest_index(vertical_edges, config.visualization.marginal_z_cm)
        second_index = _nearest_index(horizontal_edges, config.visualization.marginal_x_cm)
        first_coordinate = (vertical_edges[first_index] + vertical_edges[first_index + 1]) / 2
        second_coordinate = (horizontal_edges[second_index] + horizontal_edges[second_index + 1]) / 2
        first_values = data[:, first_index]
        second_values = data[second_index, :]
        second_axis_label = "z (cm)"
        first_description = f"at z = {first_coordinate:g} cm"
        second_description = f"at x = {second_coordinate:g} cm"

    horizontal_centers = (horizontal_edges[:-1] + horizontal_edges[1:]) / 2
    vertical_centers = (vertical_edges[:-1] + vertical_edges[1:]) / 2
    vertical_world_extent = (
        config.geometry.world_height_cm / 2
        if view == "xz" and config.geometry.world_height_cm is not None
        else world_radius
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(horizontal_centers, first_values, color="tab:blue")
    axes[0].set_xlim(*(style.xlim or (-world_radius, world_radius)))
    axes[0].set_xscale(style.marginal_xscale)
    axes[0].set_yscale(style.marginal_yscale)
    if style.marginal_ylim is not None:
        axes[0].set_ylim(*style.marginal_ylim)
    axes[0].set_xlabel("x (cm)")
    flux_units = (
        "n cm$^{-2}$ s$^{-1}$"
        if config.normalization is not None
        else "cm$^{-2}$ per source neutron"
    )
    axes[0].set_ylabel(f"Flux ({flux_units})")
    axes[0].set_title(f"x profile {first_description}")
    if config.detector.enabled:
        axes[0].axvline(
            config.detector.x_cm,
            color="magenta",
            linestyle=":",
            linewidth=1,
            label="detector x",
        )
        axes[0].legend(fontsize="small")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].plot(vertical_centers, second_values, color="tab:orange")
    axes[1].set_xlim(*(style.ylim if view == "xy" and style.ylim else
                       style.zlim if view == "xz" and style.zlim else
                       (-vertical_world_extent, vertical_world_extent)))
    axes[1].set_xscale(style.marginal_xscale)
    axes[1].set_yscale(style.marginal_yscale)
    if style.marginal_ylim is not None:
        axes[1].set_ylim(*style.marginal_ylim)
    axes[1].set_xlabel(second_axis_label)
    axes[1].set_ylabel(f"Flux ({flux_units})")
    axes[1].set_title(f"{second_axis_label[0]} profile {second_description}")
    if config.detector.enabled:
        axes[1].axvline(
            config.detector.y_cm if view == "xy" else config.detector.z_cm,
            color="magenta",
            linestyle=":",
            linewidth=1,
            label=f"detector {second_axis_label[0]}",
        )
        axes[1].legend(fontsize="small")
    axes[1].grid(True, which="both", alpha=0.25)
    figure.suptitle(f"{'XY' if view == 'xy' else 'XZ'} flux line profiles")
    figure.savefig(str(path), dpi=180)
    plt.close(figure)
    return path


def _plot_survey_spectra(
    spectra,
    path,
    plt,
    np,
    normalized: bool,
    source_birth_spectrum=None,
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    inset = axis.inset_axes([0.08, 0.54, 0.4, 0.4])
    source_axis = axis.twinx() if source_birth_spectrum is not None else None
    source_inset = inset.twinx() if source_birth_spectrum is not None else None
    main_values = []
    inset_values = []
    colors = plt.get_cmap("tab10")
    for index, (name, (edges, values)) in enumerate(spectra.items()):
        color = colors(index % 10)
        main_start = int(np.searchsorted(edges, 1.0, side="left"))
        main_end = int(np.searchsorted(edges, 16.0, side="left"))
        main_slice = values[main_start:main_end]
        axis.stairs(
            main_slice,
            edges[main_start : main_end + 1],
            label=name,
            linewidth=1.6,
            color=color,
        )
        main_values.append(main_slice)

        inset_end = int(np.searchsorted(edges, 1.0, side="right")) - 1
        low_energy_slice = values[1:inset_end]
        inset.stairs(
            low_energy_slice,
            edges[1 : inset_end + 1],
            linewidth=1.2,
            color=color,
        )
        inset_values.append(low_energy_slice)
    if source_birth_spectrum is not None:
        source_edges, source_values = source_birth_spectrum
        main_start = int(np.searchsorted(source_edges, 1.0, side="left"))
        main_end = int(np.searchsorted(source_edges, 16.0, side="left"))
        source_inset_end = int(np.searchsorted(source_edges, 1.0, side="right")) - 1
        source_axis.stairs(
            source_values[main_start:main_end],
            source_edges[main_start : main_end + 1],
            label="Custom source birth spectrum",
            linewidth=1.8,
            color="black",
            linestyle="--",
        )
        source_inset.stairs(
            source_values[1 : source_inset_end],
            source_edges[1 : source_inset_end + 1],
            linewidth=1.4,
            color="black",
            linestyle="--",
        )
        source_axis.set_yscale("log")
        source_axis.set_ylabel("Birth probability per neutron per MeV")
        source_inset.set_yscale("log")
        source_inset.tick_params(axis="y", labelsize=6)
    axis.set_xscale("linear")
    _set_spectrum_yscale(axis, main_values, "log", np)
    axis.set_xlabel("Neutron energy (MeV)")
    axis.set_ylabel(
        "Neutron flux (n cm$^{-2}$ s$^{-1}$ MeV$^{-1}$)"
        if normalized
        else "Track length density (cm$^{-2}$ per source neutron per MeV)"
    )
    axis.set_xlim(1.0, 16.0)
    axis.set_title("Neutron spectra by region")
    axis.grid(True, which="both", alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if source_axis is not None:
        source_handles, source_labels = source_axis.get_legend_handles_labels()
        handles.extend(source_handles)
        labels.extend(source_labels)
    axis.legend(handles, labels)
    inset.set_xscale("log")
    inset.set_xlim(edges[1], 1.0)
    _set_spectrum_yscale(inset, inset_values, "log", np)
    inset.set_title("Below 1 MeV (log energy)", fontsize=8)
    inset.set_xlabel("Neutron energy (MeV)", fontsize=7)
    inset.tick_params(axis="both", labelsize=7)
    inset.grid(True, which="both", alpha=0.25)
    if not any(np.any(values > 0) for values in [*main_values, *inset_values]):
        axis.text(
            0.5, 0.5, "No positive tally values", transform=axis.transAxes,
            ha="center", va="center"
        )
    figure.savefig(str(path), dpi=180)
    plt.close(figure)
    return path


def _plot_spectrum(spectra, edges, title, ylabel, path, plt, style: PlotStyle) -> Path:
    import numpy as np

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.set_prop_cycle(
        color=[
            plt.get_cmap(style.spectrum_cmap)(index % 10)
            for index in range(max(len(spectra), 1))
        ]
    )
    for name, values in spectra.items():
        axis.stairs(values, edges, label=name, linewidth=1.5)
    has_positive_values = _set_spectrum_yscale(
        axis, [*spectra.values()], style.spectrum_yscale, np
    )
    axis.set_xscale(style.spectrum_xscale)
    axis.set_xlim(*(style.spectrum_xlim or (edges[0], edges[-1])))
    if style.spectrum_ylim is not None:
        axis.set_ylim(*style.spectrum_ylim)
    axis.set_xlabel("Neutron energy (MeV)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if not has_positive_values:
        axis.text(
            0.5,
            0.5,
            "No positive tally values",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)
    return path


def _plot_detector_spectrum(
    values, std_dev, edges, config, path, plt, np, style
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.stairs(values, edges, color=style.spectrum_color, linewidth=1.8)
    lower = np.maximum(values - std_dev, 0.0)
    upper = values + std_dev
    axis.fill_between(
        edges,
        np.append(lower, lower[-1]),
        np.append(upper, upper[-1]),
        step="post",
        color=style.spectrum_color,
        alpha=0.2,
        label="1 standard deviation",
    )
    has_positive_values = _set_spectrum_yscale(
        axis, [values, std_dev], style.spectrum_yscale, np
    )
    axis.set_xscale(style.spectrum_xscale)
    axis.set_xlim(*(style.spectrum_xlim or (edges[0], edges[-1])))
    if style.spectrum_ylim is not None:
        axis.set_ylim(*style.spectrum_ylim)
    axis.set_xlabel("Neutron energy (MeV)")
    axis.set_ylabel(
        "Volume-averaged flux (n cm$^{-2}$ s$^{-1}$)"
        if config.normalization is not None
        else "Volume-averaged flux (cm$^{-2}$ per source neutron)"
    )
    detector_dimensions = (
        config.detector.width_x_cm,
        config.detector.width_y_cm,
        config.detector.thickness_z_cm,
    )
    axis.set_title(
        "Detector energy spectrum: "
        f"{detector_dimensions[0]:g} × {detector_dimensions[1]:g} × "
        f"{detector_dimensions[2]:g} cm "
        f"at x={config.detector.x_cm:g}, y={config.detector.y_cm:g}, "
        f"z={config.detector.z_cm:g} cm"
    )
    if not has_positive_values:
        axis.text(
            0.5,
            0.5,
            "No positive detector flux sampled",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)
    return path


def _plot_he3_detector_spectrum(
    neutron_flux,
    capture_response,
    edges,
    component_name,
    yscale: str,
    path,
    plt,
    style: PlotStyle,
    normalized: bool = False,
) -> Path:
    import numpy as np

    if yscale not in {"linear", "log"}:
        raise ValueError("He-3 spectrum yscale must be 'linear' or 'log'")
    inset_min_mev = 1.0e-6
    inset_max_mev = 0.1
    inset_bins = (edges[:-1] >= inset_min_mev) & (edges[1:] <= inset_max_mev)
    if not np.any(inset_bins):
        raise ValueError(
            "He-3 spectrum tallies need energy bins spanning 1 eV to 0.1 MeV"
        )
    inset_start = int(np.flatnonzero(inset_bins)[0])
    inset_stop = int(np.flatnonzero(inset_bins)[-1]) + 1
    inset_edges = edges[inset_start : inset_stop + 1]
    inset_flux = neutron_flux[inset_start:inset_stop]
    inset_response = capture_response[inset_start:inset_stop]

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    capture_axis = axis.twinx()
    flux_color = style.spectrum_color
    capture_color = plt.get_cmap(style.spectrum_cmap)(1)
    flux_line = axis.stairs(
        neutron_flux,
        edges,
        color=flux_color,
        linewidth=1.5,
        label="Neutron flux in He-3 gas",
    )
    capture_line = capture_axis.stairs(
        capture_response,
        edges,
        color=capture_color,
        linewidth=1.5,
        label="He-3 (n,p) capture response",
    )
    axis.set_xscale(style.spectrum_xscale)
    axis.set_xlim(*(style.spectrum_xlim or (edges[0], edges[-1])))
    flux_has_positive_values = _set_he3_yscale(
        axis, yscale, neutron_flux, (1.0e-12, 1.0), np
    )
    capture_has_positive_values = _set_he3_yscale(
        capture_axis, yscale, capture_response, (1.0e-12, 1.0), np
    )
    axis.set_xlabel("Incident neutron energy (MeV)")
    axis.set_ylabel(
        (
            "Volume-averaged neutron flux "
            "(n cm$^{-2}$ s$^{-1}$ MeV$^{-1}$)"
            if normalized
            else "Volume-averaged neutron flux "
            "(cm$^{-2}$ source neutron$^{-1}$ MeV$^{-1}$)"
        ),
        color=flux_color,
    )
    capture_axis.set_ylabel(
        "He-3 (n,p) reactions s$^{-1}$ MeV$^{-1}$"
        if normalized
        else "He-3 (n,p) reactions per source neutron per MeV",
        color=capture_color,
    )
    axis.tick_params(axis="y", labelcolor=flux_color)
    capture_axis.tick_params(axis="y", labelcolor=capture_color)
    axis.set_title(
        f"He-3 detector neutron spectrum and capture response "
        f"({yscale} y scale): {component_name}"
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(
        handles=[flux_line, capture_line],
        loc="upper left",
        fontsize="small",
    )
    if not flux_has_positive_values:
        axis.text(
            0.5,
            0.5,
            "No positive neutron flux sampled",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color=flux_color,
        )
    if not capture_has_positive_values:
        capture_axis.text(
            0.5,
            0.45,
            "No positive capture response sampled",
            transform=capture_axis.transAxes,
            ha="center",
            va="center",
            color=capture_color,
        )

    inset = axis.inset_axes([0.48, 0.52, 0.48, 0.42])
    inset_capture_axis = inset.twinx()
    inset.stairs(inset_flux, inset_edges, color=flux_color, linewidth=1.0)
    inset_capture_axis.stairs(
        inset_response, inset_edges, color=capture_color, linewidth=1.0
    )
    inset.set_xscale("log")
    inset.set_xlim(inset_min_mev, inset_max_mev)
    inset_flux_has_positive_values = _set_he3_yscale(
        inset, yscale, inset_flux, axis.get_ylim(), np
    )
    inset_capture_has_positive_values = _set_he3_yscale(
        inset_capture_axis,
        yscale,
        inset_response,
        capture_axis.get_ylim(),
        np,
    )
    inset.set_title("1 eV–0.1 MeV detail", fontsize="small")
    inset.set_xlabel("Incident energy (MeV)", fontsize="small")
    inset.tick_params(axis="both", labelsize="x-small")
    inset_capture_axis.tick_params(axis="y", labelsize="x-small")
    inset.grid(True, which="both", alpha=0.2)
    if not inset_flux_has_positive_values:
        inset.text(
            0.04,
            0.05,
            "No neutron flux sampled",
            transform=inset.transAxes,
            fontsize="x-small",
        )
    if not inset_capture_has_positive_values:
        inset.text(
            0.04,
            0.15,
            "No capture events",
            transform=inset.transAxes,
            fontsize="x-small",
        )
    axis.indicate_inset_zoom(inset, edgecolor="0.4")

    figure.savefig(str(path), dpi=180)
    plt.close(figure)
    return path


def _set_he3_yscale(axis, yscale: str, values, fallback_limits, np) -> bool:
    has_positive_values = bool(np.any(np.isfinite(values) & (values > 0)))
    if yscale == "log" and not has_positive_values:
        axis.set_ylim(*fallback_limits)
    axis.set_yscale(yscale)
    return has_positive_values


def _plot_he3_capture_heatmap(
    energy_edges_mev,
    z_edges_cm,
    response_density,
    component_name: str,
    path: Path,
    plt,
    LogNorm,
    style: PlotStyle,
    normalized: bool = False,
) -> Path:
    import numpy as np

    energy_edges_mev = np.asarray(energy_edges_mev, dtype=float)
    z_edges_cm = np.asarray(z_edges_cm, dtype=float)
    response_density = np.asarray(response_density, dtype=float)
    if (
        energy_edges_mev.ndim != 1
        or z_edges_cm.ndim != 1
        or response_density.shape
        != (z_edges_cm.size - 1, energy_edges_mev.size - 1)
    ):
        raise ValueError("He-3 capture heatmap arrays have inconsistent bin shapes")

    first_visible_bin = int(energy_edges_mev[0] <= 0.0)
    visible_energy_edges = energy_edges_mev[first_visible_bin:]
    visible_response = response_density[:, first_visible_bin:]
    if visible_energy_edges.size < 2 or np.any(visible_energy_edges <= 0.0):
        raise ValueError("He-3 capture heatmap needs positive logarithmic energy bins")

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    positive = visible_response[
        np.isfinite(visible_response) & (visible_response > 0.0)
    ]
    if positive.size:
        minimum = float(np.percentile(positive, style.heatmap_min_percentile))
        maximum = float(np.percentile(positive, style.heatmap_max_percentile))
        minimum = max(minimum, float(positive.min()))
        maximum = min(maximum, float(positive.max()))
        if maximum <= minimum:
            maximum = minimum * (1.0 + 1.0e-6)
        color_mesh = axis.pcolormesh(
            visible_energy_edges,
            z_edges_cm,
            np.ma.masked_less_equal(visible_response, 0.0),
            cmap=style.energy_cmap,
            norm=LogNorm(vmin=minimum, vmax=maximum),
            shading="flat",
        )
        colorbar = figure.colorbar(color_mesh, ax=axis, pad=0.02)
        colorbar.set_label(
            (
                "He-3 capture rate density "
                "(reactions s$^{-1}$ MeV$^{-1}$ cm$^{-1}$)"
                if normalized
                else "He-3 capture response density "
                "(reactions source neutron$^{-1}$ MeV$^{-1}$ cm$^{-1}$)"
            )
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No positive He-3 capture response sampled",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )

    axis.set_xscale("log")
    axis.set_xlim(visible_energy_edges[0], visible_energy_edges[-1])
    axis.set_ylim(z_edges_cm[0], z_edges_cm[-1])
    axis.set_xlabel("Incident neutron energy (MeV)")
    axis.set_ylabel("Height in He-3 gas relative to detector center (cm)")
    axis.set_title(f"He-3 capture response by energy and height: {component_name}")
    axis.text(
        0.99,
        0.01,
        "Below 1e-5 eV omitted from log axis",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize="x-small",
        color="0.3",
    )
    axis.grid(True, which="both", alpha=0.2)
    figure.savefig(str(path), dpi=180)
    plt.close(figure)
    return path


def _set_spectrum_yscale(axis, datasets, requested_scale: str, np) -> bool:
    has_positive_values = any(np.any(values > 0) for values in datasets)
    scale = (
        "linear"
        if requested_scale == "log" and not has_positive_values
        else requested_scale
    )
    axis.set_yscale(scale)
    return has_positive_values


def _material_colors(config: SimulationConfig, style: PlotStyle):
    import matplotlib.pyplot as plt

    names = list(
        dict.fromkeys(
            [layer.material for layer in config.geometry.layers]
            + [component.material for component in config.geometry.components]
        )
    )
    palette = plt.get_cmap(style.geometry_cmap, max(len(names), 1))
    return {name: palette(index) for index, name in enumerate(names)}


def _geometry_legend(config: SimulationConfig, Patch, colors):
    entries = [
        Patch(
            facecolor="0.2",
            edgecolor="black",
            label=f"Source: {config.source.energy_mev:g} MeV, "
            f"Ø{2 * config.source.radius_cm:g} cm × {config.source.height_cm:g} cm",
        )
    ]
    entries.extend(
        Patch(
            facecolor=colors[material],
            edgecolor="black",
            label=f"Material: {material}",
        )
        for material in colors
    )
    entries.extend(
        (
            Patch(
                facecolor="none",
                edgecolor="magenta",
                linestyle="--",
                label="Configured detector volume",
            ),
            Patch(
                facecolor="none",
                edgecolor="black",
                linestyle=":",
                label="Vacuum boundary",
            ),
        )
    )
    if config.geometry.air_sphere_radius_cm is not None:
        entries.append(
            Patch(
                facecolor="none",
                edgecolor="cyan",
                linestyle="--",
                label="Air-sphere boundary",
            )
        )
    return entries


def _plot_geometry(
    config, view, path, plt, Circle, Patch, Wedge, Rectangle, style: PlotStyle
) -> Path:
    world_radius = config.geometry.world_radius_cm
    vertical_world_extent = (
        config.geometry.world_height_cm / 2
        if config.geometry.world_height_cm is not None
        else world_radius
    )
    colors = _material_colors(config, style)
    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    if view == "xy":
        for layer in reversed(config.geometry.layers):
            axis.add_patch(
                Wedge(
                    (0, 0),
                    layer.radius_cm,
                    0,
                    360,
                    width=layer.thickness_cm,
                    facecolor=colors[layer.material],
                    edgecolor="black",
                    linewidth=0.8,
                )
            )
        axis.add_patch(
            Circle((0, 0), config.source.radius_cm, facecolor="0.2", edgecolor="black")
        )
        _add_component_patches(axis, config, view, Circle, Rectangle, colors)
        axis.add_patch(
            Circle(
                (0, 0),
                world_radius,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
                linestyle=":",
                label="Vacuum boundary",
            )
        )
        if config.geometry.air_sphere_radius_cm is not None:
            axis.add_patch(
                Circle(
                    (0, 0),
                    config.geometry.air_sphere_radius_cm,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.0,
                    linestyle="--",
                    label="Air-sphere boundary",
                )
            )
        _detector_overlay(axis, config, view, Rectangle)
        _set_spatial_axes(axis, view, style, world_radius)
        axis.set_xlabel("x (cm)")
        axis.set_ylabel("y (cm)")
        axis.set_aspect("equal")
        axis.set_title("Top-down geometry (XY)")
    else:
        for layer in config.geometry.layers:
            for left in (-layer.radius_cm, layer.radius_cm - layer.thickness_cm):
                layer_patch = Rectangle(
                    (left, -layer.height_cm / 2),
                    layer.thickness_cm,
                    layer.height_cm,
                    facecolor=colors[layer.material],
                    edgecolor="black",
                    linewidth=0.6,
                )
                if config.geometry.air_sphere_radius_cm is not None:
                    layer_patch.set_clip_path(
                        Circle(
                            (0, 0),
                            config.geometry.air_sphere_radius_cm,
                            transform=axis.transData,
                        )
                    )
                axis.add_patch(layer_patch)
        axis.add_patch(
            Rectangle(
                (-config.source.radius_cm, -config.source.height_cm / 2),
                2 * config.source.radius_cm,
                config.source.height_cm,
                facecolor="0.2",
                edgecolor="black",
            )
        )
        _add_component_patches(axis, config, view, Circle, Rectangle, colors)
        if config.geometry.air_sphere_radius_cm is not None:
            axis.add_patch(
                Circle(
                    (0, 0),
                    config.geometry.air_sphere_radius_cm,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.0,
                    linestyle="--",
                    label="Air-sphere boundary",
                )
            )
        axis.add_patch(
            Rectangle(
                (-world_radius, -vertical_world_extent),
                2 * world_radius,
                2 * vertical_world_extent,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
                linestyle=":",
                label="Vacuum boundary",
            )
            if config.geometry.world_height_cm is not None
            else Circle(
                (0, 0),
                world_radius,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
                linestyle=":",
                label="Vacuum boundary",
            )
        )
        _detector_overlay(axis, config, view, Rectangle)
        _set_spatial_axes(axis, view, style, world_radius, vertical_world_extent)
        axis.set_xlabel("x (cm)")
        axis.set_ylabel("z (cm)")
        axis.set_aspect("equal")
        axis.set_title("Side-on geometry cross-section (XZ)")

    axis.grid(True, alpha=0.2)
    axis.legend(
        handles=_geometry_legend(config, Patch, colors),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize="small",
    )
    figure.savefig(str(path), dpi=180)
    plt.close(figure)
    return path


def _add_component_patches(axis, config, view, Circle, Rectangle, colors):
    for component in config.geometry.components:
        axis.add_patch(
            _component_patch(
                component,
                view,
                Circle,
                Rectangle,
                facecolor=colors[component.material],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.8,
            )
        )

"""Plot geometry, spatial flux, energy bins, detector spectra, and line profiles."""

from __future__ import annotations

from pathlib import Path

from .config import SimulationConfig


def plot_results(
    config: SimulationConfig, statepoint_path: str | Path, output_dir: str | Path
) -> list[Path]:
    """Write plots from a statepoint produced by the configured model."""
    try:
        import matplotlib
        import numpy as np
        import openmc
    except ImportError as exc:
        dependency = "OpenMC" if exc.name == "openmc" else "Matplotlib"
        raise RuntimeError(f"{dependency} is required to plot simulation results") from exc

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
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
    expected_names.add("detector_energy_spectrum")
    expected_names.add("flux_spectrum_source_cavity")
    expected_names.update(f"flux_spectrum_{layer.name}" for layer in config.geometry.layers)

    with openmc.StatePoint(statepoint_file) as statepoint:
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
        _validate_mesh_tallies(map_tallies, config, np, openmc)

        flux_maps = {
            view: _mesh_values(map_tallies[f"flux_{view}"], view, 1, np)[..., 0]
            for view in ("xy", "xz")
        }
        flux_std_devs = {
            view: _mesh_values(
                map_tallies[f"flux_{view}"], view, 1, np, value="std_dev"
            )[..., 0]
            for view in ("xy", "xz")
        }
        energy_maps = {
            view: _mesh_values(
                map_tallies[f"energy_{view}"],
                view,
                len(config.visualization.energy_bins_mev) - 1,
                np,
            )
            for view in ("xy", "xz")
        }
        spectra = {
            "Source cavity": _spectrum_values(
                statepoint.get_tally(name="flux_spectrum_source_cavity"), np
            )
        }
        for layer in config.geometry.layers:
            spectra[layer.name] = _spectrum_values(
                statepoint.get_tally(name=f"flux_spectrum_{layer.name}"), np
            )
        detector_tally = statepoint.get_tally(name="detector_energy_spectrum")
        _validate_detector_tally(detector_tally, config, np, openmc)
        detector_mean = _spectrum_values(detector_tally, np)
        detector_std_dev = _spectrum_values(detector_tally, np, value="std_dev")

    radius = config.geometry.layers[-1].radius_cm
    half_height = config.geometry.layers[-1].height_cm / 2
    flux_edges = _mesh_edges(radius, config.visualization.mesh_bins)
    z_edges = _mesh_edges(half_height, config.visualization.mesh_bins)
    total_edges = np.asarray(config.tally.energy_bins_mev)
    map_edges = np.asarray(config.visualization.energy_bins_mev)
    paths: list[Path] = []

    for view in ("xy", "xz"):
        if view == "xy":
            vertical_label = "y (cm)"
            vertical_edges = flux_edges
            vertical_radius = radius
            slice_description = (
                f"central z slab, {config.visualization.slice_thickness_cm:g} cm thick"
            )
        else:
            vertical_label = "z (cm)"
            vertical_edges = z_edges
            vertical_radius = half_height
            slice_description = (
                f"central y slab, {config.visualization.slice_thickness_cm:g} cm thick"
            )
        extent = (-radius, radius, -vertical_radius, vertical_radius)
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
                "Flux (cm$^{-2}$ per source neutron)",
                destination / f"flux_{view}.png",
                config,
                view,
                LogNorm,
                Circle,
                Rectangle,
                plt,
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
            )
        )
        paths.append(
            _plot_marginals(
                total_flux,
                flux_edges,
                vertical_edges,
                view,
                config,
                destination / f"marginals_{view}.png",
                plt,
            )
        )

    paths.append(
        _plot_spectrum(
            spectra,
            total_edges,
            "Region energy spectra",
            "Flux (cm$^{-2}$ per source neutron)",
            destination / "energy_spectra_by_region.png",
            plt,
        )
    )
    paths.append(
        _plot_detector_spectrum(
            detector_mean,
            detector_std_dev,
            total_edges,
            config,
            destination / "detector_energy_spectrum.png",
            plt,
            np,
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


def _validate_mesh_tallies(tallies, config: SimulationConfig, np, openmc) -> None:
    radius = config.geometry.layers[-1].radius_cm
    half_height = config.geometry.layers[-1].height_cm / 2
    half_slice = config.visualization.slice_thickness_cm / 2
    map_edges_ev = np.asarray(config.visualization.energy_bins_mev) * 1.0e6
    expected_energy_bins = np.column_stack((map_edges_ev[:-1], map_edges_ev[1:]))
    expected = {
        "xy": (
            (config.visualization.mesh_bins, config.visualization.mesh_bins, 1),
            (-radius, -radius, -half_slice),
            (radius, radius, half_slice),
        ),
        "xz": (
            (config.visualization.mesh_bins, 1, config.visualization.mesh_bins),
            (-radius, -half_slice, -half_height),
            (radius, half_slice, half_height),
        ),
    }
    for view in ("xy", "xz"):
        for kind in ("flux", "energy"):
            tally = tallies[f"{kind}_{view}"]
            mesh_filter = next(
                (item for item in tally.filters if isinstance(item, openmc.MeshFilter)),
                None,
            )
            if mesh_filter is None:
                raise ValueError(f"Statepoint tally {tally.name!r} has no mesh filter")
            mesh = mesh_filter.mesh
            dimensions, lower, upper = expected[view]
            if tuple(int(item) for item in mesh.dimension) != dimensions:
                raise ValueError(f"Statepoint tally {tally.name!r} has unexpected mesh dimensions")
            if not (
                np.allclose(mesh.lower_left, lower)
                and np.allclose(mesh.upper_right, upper)
            ):
                raise ValueError(f"Statepoint tally {tally.name!r} mesh bounds do not match config")
            if kind == "energy":
                energy_filter = next(
                    (item for item in tally.filters if isinstance(item, openmc.EnergyFilter)),
                    None,
                )
                if energy_filter is None or not np.allclose(
                    energy_filter.bins, expected_energy_bins, rtol=1e-12, atol=1e-8
                ):
                    raise ValueError(f"Statepoint tally {tally.name!r} has unexpected energy bins")


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


def _geometry_outlines(config: SimulationConfig, view: str, Circle, Rectangle):
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
        outlines.extend(
            (
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
        )
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
    return outlines


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
) -> Path:
    import numpy as np

    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    masked = np.ma.masked_invalid(data)
    positive = masked.compressed()
    positive = positive[positive > 0]
    if LogNorm is None:
        finite = masked.compressed()
        options = {
            "vmin": 0.0,
            "vmax": float(finite.max()) if finite.size and finite.max() > 0 else 1.0,
            "cmap": "magma",
        }
    else:
        options = {"cmap": "viridis"}
        if positive.size:
            low, high = float(positive.min()), float(positive.max())
            options["norm"] = LogNorm(vmin=low / 2 if low == high else low, vmax=high)
    image = axis.imshow(masked, origin="lower", extent=extent, aspect="equal", **options)
    for outline in _geometry_outlines(config, view, Circle, Rectangle):
        axis.add_patch(outline)
    axis.set_xlabel("x (cm)")
    axis.set_ylabel("y (cm)" if view == "xy" else "z (cm)")
    axis.set_title(title)
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
        low, high = float(positives.min()), float(positives.max())
        norm = LogNorm(vmin=low / 2 if low == high else low, vmax=high)

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
            cmap="viridis",
            norm=norm,
        )
        for outline in _geometry_outlines(config, view, Circle, Rectangle):
            axis.add_patch(outline)
        axis.set_title(f"{edges[index]:g}-{edges[index + 1]:g} MeV")
        axis.set_xlabel("x (cm)")
        axis.set_ylabel("y (cm)" if view == "xy" else "z (cm)")

    figure.suptitle(f"{orientation}: flux by energy bin, {slice_description}")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Flux (cm$^{-2}$ per source neutron)",
            shrink=0.85,
        )
    figure.savefig(str(path), dpi=180)
    plt.close(figure)
    return path


def _nearest_index(edges, coordinate: float) -> int:
    import numpy as np

    centers = (edges[:-1] + edges[1:]) / 2
    return int(np.abs(centers - coordinate).argmin())


def _plot_marginals(data, horizontal_edges, vertical_edges, view, config, path, plt) -> Path:
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
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(horizontal_centers, first_values, color="tab:blue")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("x (cm)")
    axes[0].set_ylabel("Flux (cm$^{-2}$ per source neutron)")
    axes[0].set_title(f"x profile {first_description}")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].plot(vertical_centers, second_values, color="tab:orange")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(second_axis_label)
    axes[1].set_ylabel("Flux (cm$^{-2}$ per source neutron)")
    axes[1].set_title(f"{second_axis_label[0]} profile {second_description}")
    axes[1].grid(True, which="both", alpha=0.25)
    figure.suptitle(f"{'XY' if view == 'xy' else 'XZ'} flux line profiles")
    figure.savefig(str(path), dpi=180)
    plt.close(figure)
    return path


def _plot_spectrum(spectra, edges, title, ylabel, path, plt) -> Path:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for name, values in spectra.items():
        axis.stairs(values, edges, label=name, linewidth=1.5)
    axis.set_yscale("log")
    axis.set_xlim(edges[0], edges[-1])
    axis.set_xlabel("Neutron energy (MeV)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)
    return path


def _plot_detector_spectrum(values, std_dev, edges, config, path, plt, np) -> Path:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.stairs(values, edges, color="tab:blue", linewidth=1.8)
    lower = np.maximum(values - std_dev, 0.0)
    upper = values + std_dev
    axis.fill_between(
        edges,
        np.append(lower, lower[-1]),
        np.append(upper, upper[-1]),
        step="post",
        color="tab:blue",
        alpha=0.2,
        label="1 standard deviation",
    )
    axis.set_yscale("log")
    axis.set_xlim(edges[0], edges[-1])
    axis.set_xlabel("Neutron energy (MeV)")
    axis.set_ylabel("Volume-averaged flux (cm$^{-2}$ per source neutron)")
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
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)
    return path


def _material_colors(config: SimulationConfig):
    import matplotlib.pyplot as plt

    palette = plt.get_cmap("tab10")
    names = list(dict.fromkeys(layer.material for layer in config.geometry.layers))
    return {name: palette(index % 10) for index, name in enumerate(names)}


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
            facecolor=colors[layer.material],
            edgecolor="black",
            label=(
                f"{index}. {layer.name}: {layer.material}, "
                f"{layer.thickness_cm * 10:g} mm"
            ),
        )
        for index, layer in enumerate(config.geometry.layers, start=1)
    )
    return entries


def _plot_geometry(config, view, path, plt, Circle, Patch, Wedge, Rectangle) -> Path:
    radius = config.geometry.layers[-1].radius_cm
    colors = _material_colors(config)
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
        axis.set_xlim(-radius * 1.05, radius * 1.05)
        axis.set_ylim(-radius * 1.05, radius * 1.05)
        axis.set_xlabel("x (cm)")
        axis.set_ylabel("y (cm)")
        axis.set_aspect("equal")
        axis.set_title("Top-down geometry (XY)")
    else:
        for layer in config.geometry.layers:
            for left in (-layer.radius_cm, layer.radius_cm - layer.thickness_cm):
                axis.add_patch(
                    Rectangle(
                        (left, -layer.height_cm / 2),
                        layer.thickness_cm,
                        layer.height_cm,
                        facecolor=colors[layer.material],
                        edgecolor="black",
                        linewidth=0.6,
                    )
                )
        axis.add_patch(
            Rectangle(
                (-config.source.radius_cm, -config.source.height_cm / 2),
                2 * config.source.radius_cm,
                config.source.height_cm,
                facecolor="0.2",
                edgecolor="black",
            )
        )
        axis.set_xlim(-radius * 1.1, radius * 1.1)
        axis.set_ylim(
            -config.geometry.layers[-1].height_cm / 2 * 1.05,
            config.geometry.layers[-1].height_cm / 2 * 1.05,
        )
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

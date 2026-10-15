"""Create spatial flux maps and energy-spectrum plots from OpenMC statepoints."""

from __future__ import annotations

from pathlib import Path

from .config import SimulationConfig


def plot_results(
    config: SimulationConfig, statepoint_path: str | Path, output_dir: str | Path
) -> list[Path]:
    """Write projection flux, energy-bin, uncertainty, and region-spectrum plots."""
    try:
        import matplotlib
        import numpy as np
        import openmc
    except ImportError as exc:
        package = "OpenMC" if exc.name == "openmc" else "Matplotlib"
        extra = "OpenMC per the README" if package == "OpenMC" else "the project plotting extra"
        raise RuntimeError(f"{package} is required to plot results; install {extra}") from exc

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle, Rectangle

    statepoint_file = Path(statepoint_path).resolve()
    if not statepoint_file.is_file():
        raise FileNotFoundError(f"Statepoint file not found: {statepoint_file}")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    names = ("xy", "xz")
    map_names = {
        f"{kind}_{view}": f"{kind}_map_{view}"
        for kind in ("flux", "energy")
        for view in names
    }
    required_names = set(map_names.values())
    required_names.add("flux_spectrum_source_cavity")
    required_names.update(
        f"flux_spectrum_{layer.name}" for layer in config.geometry.layers
    )
    with openmc.StatePoint(statepoint_file) as statepoint:
        available_names = {tally.name for tally in statepoint.tallies.values()}
        missing = required_names - available_names
        if missing:
            raise ValueError(
                "This statepoint does not contain the spatial mesh tallies required "
                f"for plotting ({', '.join(sorted(missing))}). Rerun the simulation "
                "with the current model using `iec-neutrons run <config.yaml>`."
            )
        tallies = {
            key: statepoint.get_tally(name=tally_name)
            for key, tally_name in map_names.items()
        }
        _validate_mesh_tallies(tallies, config, np)
        spectra = {
            "source cavity": statepoint.get_tally(name="flux_spectrum_source_cavity")
        }
        for layer in config.geometry.layers:
            spectra[layer.name] = statepoint.get_tally(name=f"flux_spectrum_{layer.name}")
        flux_data = {
            view: _mesh_array(tallies[f"flux_{view}"], view, 1)
            for view in names
        }
        flux_std_dev = {
            view: _mesh_array(tallies[f"flux_{view}"], view, 1, value="std_dev")
            for view in names
        }
        energy_data = {
            view: _mesh_array(
                tallies[f"energy_{view}"],
                view,
                len(config.tally.energy_bins_mev) - 1,
            )
            for view in names
        }
        spectrum_data = {
            name: np.asarray(tally.get_reshaped_data())[:, :, 0, 0].reshape(-1)
            for name, tally in spectra.items()
        }

    radius = config.geometry.world_radius_cm
    bin_edges = np.asarray(config.tally.energy_bins_mev)
    paths: list[Path] = []
    for view in names:
        flux = flux_data[view][..., 0]
        uncertainty = np.divide(
            flux_std_dev[view][..., 0],
            flux,
            out=np.full_like(flux, np.nan, dtype=float),
            where=flux > 0,
        )
        orientation = "Top-down (XY)" if view == "xy" else "Side-on cross-section (XZ)"
        label = "y (cm)" if view == "xy" else "z (cm)"
        paths.append(
            _plot_map(
                flux.T,
                radius,
                f"{orientation}: flux integrated over all neutron energies",
                "Flux (cm$^{-2}$ per source neutron)",
                destination / f"flux_{view}.png",
                config,
                label,
                view,
                LogNorm,
                Circle,
                Rectangle,
                plt,
            )
        )
        paths.append(
            _plot_map(
                uncertainty.T,
                radius,
                f"{orientation}: relative statistical uncertainty",
                "Relative uncertainty (standard deviation / mean)",
                destination / f"relative_uncertainty_{view}.png",
                config,
                label,
                view,
                None,
                Circle,
                Rectangle,
                plt,
            )
        )
        paths.append(
            _plot_energy_maps(
                energy_data[view],
                bin_edges,
                radius,
                orientation,
                destination / f"energy_bins_{view}.png",
                config,
                label,
                view,
                LogNorm,
                Circle,
                Rectangle,
                plt,
            )
        )

    spectrum_path = destination / "energy_spectra_by_region.png"
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for name, values in spectrum_data.items():
        axis.stairs(values, bin_edges, label=name, linewidth=1.8)
    axis.set_yscale("log")
    axis.set_xlim(bin_edges[0], bin_edges[-1])
    axis.set_xlabel("Neutron energy (MeV)")
    axis.set_ylabel("Binned flux (cm$^{-2}$ per source neutron)")
    axis.set_title("Energy-binned neutron flux by region")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.savefig(str(spectrum_path), dpi=180)
    plt.close(figure)
    paths.append(spectrum_path)
    return paths


def _mesh_array(tally, view: str, energy_groups: int, value: str = "mean"):
    import numpy as np

    data = tally.get_reshaped_data(value=value, expand_dims=True)
    if len(tally.filters) == 1:
        mesh_values = data[..., 0, 0]
        if view == "xy":
            result = mesh_values[:, :, 0, np.newaxis]
        else:
            result = mesh_values[:, 0, :, np.newaxis]
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


def _validate_mesh_tallies(tallies, config: SimulationConfig, np) -> None:
    import openmc

    radius = config.geometry.world_radius_cm
    edges_ev = np.asarray(config.tally.energy_bins_mev) * 1.0e6
    expected_energy_bins = np.column_stack((edges_ev[:-1], edges_ev[1:]))
    for view, dimensions in (
        ("xy", (config.visualization.mesh_bins, config.visualization.mesh_bins, 1)),
        ("xz", (config.visualization.mesh_bins, 1, config.visualization.mesh_bins)),
    ):
        for kind in ("flux", "energy"):
            name = f"{kind}_{view}"
            tally = tallies[name]
            mesh_filter = next(
                (item for item in tally.filters if isinstance(item, openmc.MeshFilter)),
                None,
            )
            if mesh_filter is None:
                raise ValueError(f"Statepoint tally {tally.name!r} has no mesh filter")
            mesh = mesh_filter.mesh
            actual_dimensions = tuple(int(value) for value in mesh.dimension)
            if actual_dimensions != dimensions:
                raise ValueError(
                    f"Statepoint tally {tally.name!r} uses mesh dimensions "
                    f"{actual_dimensions}, but the configuration expects {dimensions}"
                )
            if not (
                np.allclose(mesh.lower_left, (-radius, -radius, -radius))
                and np.allclose(mesh.upper_right, (radius, radius, radius))
            ):
                raise ValueError(
                    f"Statepoint tally {tally.name!r} mesh bounds do not match "
                    "geometry.world_radius_cm in the supplied configuration"
                )
            if kind == "energy":
                energy_filter = next(
                    (
                        item
                        for item in tally.filters
                        if isinstance(item, openmc.EnergyFilter)
                    ),
                    None,
                )
                if energy_filter is None or not np.allclose(
                    energy_filter.bins, expected_energy_bins, rtol=1e-12, atol=1e-8
                ):
                    raise ValueError(
                        f"Statepoint tally {tally.name!r} energy bins do not match "
                        "tally.energy_bins_mev in the supplied configuration"
                    )


def _geometry_outlines(config: SimulationConfig, view: str, Circle, Rectangle):
    if view == "xy":
        outlines = [
            Circle((0, 0), config.geometry.cavity_radius_cm, fill=False, color="white", linewidth=1.0)
        ]
        for layer in config.geometry.layers:
            outlines.append(Circle((0, 0), layer.radius_cm, fill=False, color="white", linewidth=1.0))
        outlines.append(
            Circle((0, 0), config.source.radius_cm, fill=False, color="cyan", linewidth=1.1, linestyle="--")
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
        outlines.append(
            Rectangle(
                (-layer.radius_cm, -layer.height_cm / 2),
                2 * layer.radius_cm,
                layer.height_cm,
                fill=False,
                edgecolor="white",
                linewidth=1.0,
            )
        )
    outlines.append(
        Rectangle(
            (-config.source.radius_cm, -config.source.height_cm / 2),
            2 * config.source.radius_cm,
            config.source.height_cm,
            fill=False,
            edgecolor="cyan",
            linewidth=1.1,
            linestyle="--",
        )
    )
    return outlines


def _plot_map(
    data,
    radius: float,
    title: str,
    colorbar_label: str,
    path: Path,
    config: SimulationConfig,
    vertical_label: str,
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
    if LogNorm is not None:
        if positive.size:
            low, high = float(positive.min()), float(positive.max())
            if low == high:
                low = high * 0.5
            image = axis.imshow(
                masked,
                origin="lower",
                extent=(-radius, radius, -radius, radius),
                aspect="equal",
                norm=LogNorm(vmin=low, vmax=high),
                cmap="viridis",
            )
        else:
            image = axis.imshow(
                masked,
                origin="lower",
                extent=(-radius, radius, -radius, radius),
                aspect="equal",
                cmap="viridis",
            )
    else:
        finite = masked.compressed()
        image = axis.imshow(
            masked,
            origin="lower",
            extent=(-radius, radius, -radius, radius),
            aspect="equal",
            vmin=0.0,
            vmax=float(finite.max()) if finite.size and finite.max() > 0 else 1.0,
            cmap="magma",
        )
    for outline in _geometry_outlines(config, view, Circle, Rectangle):
        axis.add_patch(outline)
    axis.set_xlabel("x (cm)")
    axis.set_ylabel(vertical_label)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.savefig(str(path), dpi=180)
    plt.close(figure)
    return path


def _plot_energy_maps(
    data,
    edges,
    radius: float,
    orientation: str,
    path: Path,
    config: SimulationConfig,
    vertical_label: str,
    view: str,
    LogNorm,
    Circle,
    Rectangle,
    plt,
) -> Path:
    import numpy as np

    groups = data.shape[-1]
    positive = data[np.isfinite(data) & (data > 0)]
    rows = int(np.ceil(groups / 3))
    figure, axes = plt.subplots(
        rows,
        3,
        figsize=(14, 4.2 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    if positive.size:
        low, high = float(positive.min()), float(positive.max())
        if low == high:
            low = high * 0.5
        norm = LogNorm(vmin=low, vmax=high)
    else:
        norm = None

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
            extent=(-radius, radius, -radius, radius),
            aspect="equal",
            cmap="viridis",
            norm=norm,
        )
        for outline in _geometry_outlines(config, view, Circle, Rectangle):
            axis.add_patch(outline)
        axis.set_title(f"{edges[index]:g}-{edges[index + 1]:g} MeV")
        axis.set_xlabel("x (cm)")
        axis.set_ylabel(vertical_label)

    figure.suptitle(f"{orientation}: flux by energy bin (shared color scale)")
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

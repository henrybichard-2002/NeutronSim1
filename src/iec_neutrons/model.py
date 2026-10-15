"""Translate validated configuration into an OpenMC model."""

from __future__ import annotations

from pathlib import Path

from .config import SimulationConfig


def build_model(config: SimulationConfig):
    """Construct an OpenMC model with a volumetric cylindrical D-T source."""
    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC is required to build or run a model; install it using the instructions in README.md"
        ) from exc

    materials: dict[str, openmc.Material] = {}
    for name, specification in config.materials.items():
        material = openmc.Material(name=name)
        material.set_density("g/cm3", specification.density_g_cm3)
        composition = specification.elements if specification.elements is not None else specification.nuclides
        assert composition is not None
        add_component = material.add_element if specification.elements is not None else material.add_nuclide
        for component, amount in composition.items():
            add_component(component, amount)
        for table in specification.thermal_scattering:
            material.add_s_alpha_beta(table)
        materials[name] = material

    def cylinder_region(radius_cm: float, height_cm: float):
        cylinder = openmc.ZCylinder(r=radius_cm)
        lower = openmc.ZPlane(z0=-height_cm / 2)
        upper = openmc.ZPlane(z0=height_cm / 2)
        return -cylinder & +lower & -upper

    cavity_region = cylinder_region(
        config.geometry.cavity_radius_cm, config.geometry.cavity_height_cm
    )
    source_region = cylinder_region(config.source.radius_cm, config.source.height_cm)
    source_cell = openmc.Cell(name="source volume", region=source_region)
    cavity_cell = openmc.Cell(
        name="source cavity outside source volume",
        region=cavity_region & ~source_region,
    )
    cells = [source_cell, cavity_cell]
    previous_region = cavity_region
    tally_cells: list[tuple[str, openmc.Cell]] = [
        ("source", source_cell),
        ("source_cavity", cavity_cell),
    ]
    for layer in config.geometry.layers:
        layer_region = cylinder_region(layer.radius_cm, layer.height_cm)
        cell = openmc.Cell(
            name=layer.name,
            fill=materials[layer.material],
            region=layer_region & ~previous_region,
        )
        cells.append(cell)
        tally_cells.append((layer.name, cell))
        previous_region = layer_region

    boundary = openmc.Sphere(r=config.geometry.world_radius_cm, boundary_type="vacuum")
    cells.append(
        openmc.Cell(name="external vacuum", region=-boundary & ~previous_region)
    )
    geometry = openmc.Geometry(openmc.Universe(cells=cells))

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = config.settings.particles
    settings.batches = config.settings.batches
    settings.inactive = config.settings.inactive
    if config.settings.seed is not None:
        settings.seed = config.settings.seed
    source_space = openmc.stats.CylindricalIndependent(
        r=openmc.stats.PowerLaw(a=0.0, b=config.source.radius_cm, n=1.0),
        phi=openmc.stats.Uniform(a=0.0, b=2.0 * 3.141592653589793),
        z=openmc.stats.Uniform(a=-config.source.height_cm / 2, b=config.source.height_cm / 2),
    )
    settings.source = openmc.IndependentSource(
        space=source_space,
        energy=openmc.stats.Discrete([config.source.energy_mev * 1.0e6], [1.0]),
        particle="neutron",
    )

    energy_edges_ev = [edge * 1.0e6 for edge in config.tally.energy_bins_mev]
    map_energy_edges_ev = [
        edge * 1.0e6 for edge in config.visualization.energy_bins_mev
    ]

    def make_energy_filter(edges_ev: list[float]):
        return openmc.EnergyFilter(edges_ev)

    tallies = openmc.Tallies()
    for name, cell in tally_cells:
        tally = openmc.Tally(name=f"flux_spectrum_{name}")
        tally.filters = [openmc.CellFilter(cell), make_energy_filter(energy_edges_ev)]
        tally.scores = ["flux"]
        tallies.append(tally)

    mesh_bins = config.visualization.mesh_bins
    radius = config.geometry.world_radius_cm
    half_slice = config.visualization.slice_thickness_cm / 2
    for view, dimensions, lower_left, upper_right in (
        (
            "xy",
            (mesh_bins, mesh_bins, 1),
            (-radius, -radius, -half_slice),
            (radius, radius, half_slice),
        ),
        (
            "xz",
            (mesh_bins, 1, mesh_bins),
            (-radius, -half_slice, -radius),
            (radius, half_slice, radius),
        ),
    ):
        mesh = openmc.RegularMesh(name=f"mesh_{view}")
        mesh.dimension = dimensions
        mesh.lower_left = lower_left
        mesh.upper_right = upper_right

        flux_tally = openmc.Tally(name=f"flux_map_{view}")
        flux_tally.filters = [openmc.MeshFilter(mesh)]
        flux_tally.scores = ["flux"]
        tallies.append(flux_tally)

        spectrum_tally = openmc.Tally(name=f"energy_map_{view}")
        spectrum_tally.filters = [
            openmc.MeshFilter(mesh),
            make_energy_filter(map_energy_edges_ev),
        ]
        spectrum_tally.scores = ["flux"]
        tallies.append(spectrum_tally)

    detector_mesh = openmc.RegularMesh(name="spectrum_detector")
    detector_mesh.dimension = (1, 1, 1)
    detector_mesh.lower_left = (
        config.detector.x_cm - config.detector.width_x_cm / 2,
        config.detector.y_cm - config.detector.width_y_cm / 2,
        config.detector.z_cm - config.detector.thickness_z_cm / 2,
    )
    detector_mesh.upper_right = (
        config.detector.x_cm + config.detector.width_x_cm / 2,
        config.detector.y_cm + config.detector.width_y_cm / 2,
        config.detector.z_cm + config.detector.thickness_z_cm / 2,
    )
    detector_tally = openmc.Tally(name="detector_energy_spectrum")
    detector_tally.filters = [
        openmc.MeshFilter(detector_mesh),
        make_energy_filter(energy_edges_ev),
    ]
    detector_tally.scores = ["flux"]
    tallies.append(detector_tally)

    return openmc.Model(geometry=geometry, materials=list(materials.values()), settings=settings, tallies=tallies)


def export_model(config: SimulationConfig, output_dir: str | Path) -> Path:
    """Build the model and export OpenMC XML input files to the output directory."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    build_model(config).export_to_xml(directory=str(destination))
    return destination


def run_model(config: SimulationConfig, output_dir: str | Path) -> Path:
    """Run OpenMC and return the generated statepoint path."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    statepoint = build_model(config).run(cwd=str(destination))
    return Path(statepoint)

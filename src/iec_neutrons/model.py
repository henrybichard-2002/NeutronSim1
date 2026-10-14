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
    cells = [openmc.Cell(name="source cavity", region=cavity_region)]
    previous_region = cavity_region
    tally_cells: list[tuple[str, openmc.Cell]] = [("source_cavity", cells[0])]
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

    energy_filter = openmc.EnergyFilter(
        [edge * 1.0e6 for edge in config.tally.energy_bins_mev]
    )
    tallies = openmc.Tallies()
    for name, cell in tally_cells:
        tally = openmc.Tally(name=f"flux_spectrum_{name}")
        tally.filters = [openmc.CellFilter(cell), energy_filter]
        tally.scores = ["flux"]
        tallies.append(tally)

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

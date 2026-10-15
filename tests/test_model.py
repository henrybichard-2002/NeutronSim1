from pathlib import Path

import pytest

from iec_neutrons.config import load_config
from iec_neutrons.model import build_model


CONFIG_DIR = Path(__file__).parents[1] / "config"


@pytest.mark.parametrize(
    "filename", ["reactorpath1.yaml", "reactorpath2.yaml", "reactorpath3.yaml"]
)
def test_source_flux_tally_and_spatial_mesh_cover_world(filename: str) -> None:
    openmc = pytest.importorskip("openmc")
    config = load_config(CONFIG_DIR / filename)
    model = build_model(config)
    tallies = {tally.name: tally for tally in model.tallies}

    source_cell = next(
        cell
        for cell in model.geometry.get_all_cells().values()
        if cell.name == "source volume"
    )
    source_tally = tallies["flux_spectrum_source"]
    source_cell_filter = next(
        item for item in source_tally.filters if isinstance(item, openmc.CellFilter)
    )
    assert source_cell.id in source_cell_filter.bins

    for view in ("xy", "xz"):
        flux_tally = tallies[f"flux_map_{view}"]
        mesh_filter = next(
            item for item in flux_tally.filters if isinstance(item, openmc.MeshFilter)
        )
        radius = config.geometry.world_radius_cm
        if view == "xy":
            assert tuple(mesh_filter.mesh.lower_left) == pytest.approx(
                (-radius, -radius, -config.visualization.slice_thickness_cm / 2)
            )
            assert tuple(mesh_filter.mesh.upper_right) == pytest.approx(
                (radius, radius, config.visualization.slice_thickness_cm / 2)
            )
        else:
            assert tuple(mesh_filter.mesh.lower_left) == pytest.approx(
                (-radius, -config.visualization.slice_thickness_cm / 2, -radius)
            )
            assert tuple(mesh_filter.mesh.upper_right) == pytest.approx(
                (radius, config.visualization.slice_thickness_cm / 2, radius)
            )


def test_external_geometry_and_physics_tallies_are_constructed() -> None:
    openmc = pytest.importorskip("openmc")
    config = load_config(CONFIG_DIR / "he3_detector.yaml")
    model = build_model(config)
    tallies = {tally.name: tally for tally in model.tallies}
    cells = {cell.name: cell for cell in model.geometry.get_all_cells().values()}

    assert model.settings.photon_transport
    assert all(
        name in cells
        for name in ("hdpe_moderator", "he3_tube_wall", "he3_gas")
    )
    assert "he3_n_p_spectrum_he3_gas" in tallies
    assert "photon_heating_he3_gas" in tallies
    assert "voxel_flux" not in tallies
    assert "voxel_photon_heating" not in tallies

    transfer_tally = tallies["neutron_energy_transfer_he3_gas"]
    assert any(isinstance(item, openmc.EnergyoutFilter) for item in transfer_tally.filters)
    assert "secondary_particle_production" in tallies


def test_neutron_only_configs_do_not_add_photon_tallies() -> None:
    openmc = pytest.importorskip("openmc")
    config = load_config(CONFIG_DIR / "reactorpath1.yaml")
    model = build_model(config)
    names = {tally.name for tally in model.tallies}

    assert not model.settings.photon_transport
    assert "voxel_flux" not in names
    assert "voxel_photon_heating" not in names

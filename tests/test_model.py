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
    assert "he3_neutron_spectrum_he3_gas" in tallies
    he3_response = tallies["he3_n_p_spectrum_he3_gas"]
    assert he3_response.scores == ["(n,p)"]
    assert he3_response.nuclides == ["He3"]
    response_energy_filter = next(
        item
        for item in he3_response.filters
        if isinstance(item, openmc.EnergyFilter)
    )
    neutron_spectrum = tallies["he3_neutron_spectrum_he3_gas"]
    neutron_energy_filter = next(
        item
        for item in neutron_spectrum.filters
        if isinstance(item, openmc.EnergyFilter)
    )
    assert response_energy_filter.bins.tolist() == neutron_energy_filter.bins.tolist()
    he3_edges_ev = [
        neutron_energy_filter.bins[0, 0],
        *neutron_energy_filter.bins[:, 1],
    ]
    assert 1.0 in he3_edges_ev
    assert 1.0e5 in he3_edges_ev
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


def test_gen_reactor_geometry_and_custom_source_zones() -> None:
    openmc = pytest.importorskip("openmc")
    config = load_config(CONFIG_DIR / "gen_reactor.yaml")
    model = build_model(config)
    assert config.source.radius_cm == pytest.approx(3.8)
    assert config.source.height_cm == pytest.approx(35.0)
    assert config.geometry.cavity_height_cm == pytest.approx(70.0)
    assert [layer.thickness_cm for layer in config.geometry.layers] == pytest.approx(
        [1.4, 1.5, 0.3, 50.0]
    )
    assert [layer.height_cm for layer in config.geometry.layers] == pytest.approx(
        [70.0, 70.0, 70.0, 100.0]
    )
    assert config.geometry.layers[-1].radius_cm == pytest.approx(57.0)
    assert config.geometry.world_radius_cm == 80.0
    components = {item.name: item for item in config.geometry.components}
    assert components["detector_12cm_hdpe"].carve_from_base
    assert not components["detector_12cm_lead"].carve_from_base
    assert not components["detector_12cm_aluminium_wall"].carve_from_base
    assert not components["detector_12cm_he3_gas"].carve_from_base
    assert components["aluminium_cap_top"].carve_from_base
    assert components["detector_12cm_he3_gas"].center_cm == (-25.0, 0.0, 0.0)
    assert components["detector_12cm_aluminium_wall"].dimensions_cm == pytest.approx(
        (1.27, 31.0)
    )
    assert components["detector_12cm_he3_gas"].dimensions_cm == pytest.approx(
        (1.17, 30.8)
    )
    assert components["detector_12cm_lead"].dimensions_cm == pytest.approx((1.57, 31.6))
    assert components["detector_12cm_hdpe"].dimensions_cm == pytest.approx((13.57, 31.6))
    detector_shields = {
        10: ("detector_10cm", (25.0, 0.0, 0.0)),
        11: ("detector_11cm", (0.0, 25.0, 0.0)),
        12: ("detector_12cm", (-25.0, 0.0, 0.0)),
        13: ("detector_13cm", (0.0, -25.0, 0.0)),
    }
    for thickness_cm, (prefix, center) in detector_shields.items():
        assert components[f"{prefix}_hdpe"].center_cm == center
        assert components[f"{prefix}_hdpe"].dimensions_cm == pytest.approx(
            (1.57 + thickness_cm, 31.6)
        )
        assert components[f"{prefix}_lead"].dimensions_cm == pytest.approx(
            (1.57, 31.6)
        )
        assert components[f"{prefix}_aluminium_wall"].dimensions_cm == pytest.approx(
            (1.27, 31.0)
        )
        assert components[f"{prefix}_he3_gas"].center_cm == center
        assert components[f"{prefix}_he3_gas"].dimensions_cm == pytest.approx(
            (1.17, 30.8)
        )
        assert not components[f"{prefix}_lead"].carve_from_base
        assert not components[f"{prefix}_aluminium_wall"].carve_from_base
        assert not components[f"{prefix}_he3_gas"].carve_from_base
        assert f"he3_neutron_spectrum_{prefix}_he3_gas" in {
            tally.name for tally in model.tallies
        }
        assert f"he3_n_p_spectrum_{prefix}_he3_gas" in {
            tally.name for tally in model.tallies
        }
    assert components["detector_12cm_he3_gas"].center_cm == (-25.0, 0.0, 0.0)
    assert components["detector_10cm_he3_gas"].center_cm == (25.0, 0.0, 0.0)
    assert components["detector_11cm_he3_gas"].center_cm == (0.0, 25.0, 0.0)
    assert components["detector_13cm_he3_gas"].center_cm == (0.0, -25.0, 0.0)
    assert components["aluminium_cap_top"].center_cm == (0.0, 0.0, 35.7)
    assert components["aluminium_cap_bottom"].center_cm == (0.0, 0.0, -35.7)
    assert config.custom_source is not None
    assert config.custom_source.tritium_fraction == pytest.approx(0.5)
    assert (
        config.custom_source.zone_beam_target,
        config.custom_source.zone_beam_wall,
        config.custom_source.zone_beam_beam,
    ) == (1.0, 1.0, 1.0)
    assert config.tally.energy_bins_mev[0] == 0.0
    assert config.tally.energy_bins_mev[-1] == 20.0
    matching_cylinders = [
        surface
        for surface in model.geometry.get_all_surfaces().values()
        if isinstance(surface, openmc.ZCylinder) and surface.r == 3.8
    ]
    assert len(matching_cylinders) == 3


def test_gen_dd_reactor_geometry_and_detector_shields() -> None:
    openmc = pytest.importorskip("openmc")
    config = load_config(CONFIG_DIR / "genDD_reactor.yaml")
    model = build_model(config)
    tallies = {tally.name: tally for tally in model.tallies}

    assert config.custom_source is not None
    assert config.custom_source.tritium_fraction == 0.0
    assert [layer.thickness_cm for layer in config.geometry.layers] == pytest.approx(
        [1.4, 1.5, 0.3, 50.0]
    )
    components = {item.name: item for item in config.geometry.components}
    detector_layout = {
        2: (25.0, 0.0, 0.0),
        3: (0.0, 25.0, 0.0),
        4: (-25.0, 0.0, 0.0),
        5: (0.0, -25.0, 0.0),
    }
    for thickness_cm, center in detector_layout.items():
        prefix = f"detector_{thickness_cm}cm"
        assert components[f"{prefix}_hdpe"].center_cm == center
        assert components[f"{prefix}_hdpe"].dimensions_cm == pytest.approx(
            (1.57 + thickness_cm, 31.6)
        )
        assert components[f"{prefix}_lead"].dimensions_cm == pytest.approx(
            (1.57, 31.6)
        )
        assert components[f"{prefix}_aluminium_wall"].dimensions_cm == pytest.approx(
            (1.27, 31.0)
        )
        assert components[f"{prefix}_he3_gas"].dimensions_cm == pytest.approx(
            (1.17, 30.8)
        )
        assert f"he3_neutron_spectrum_{prefix}_he3_gas" in {
            tally.name for tally in model.tallies
        }
        assert f"he3_n_p_spectrum_{prefix}_he3_gas" in {
            tally.name for tally in model.tallies
        }
        axial_tally = tallies[f"he3_n_p_axial_energy_{prefix}_he3_gas"]
        mesh_filter = next(
            item for item in axial_tally.filters if isinstance(item, openmc.MeshFilter)
        )
        energy_filter = next(
            item
            for item in axial_tally.filters
            if isinstance(item, openmc.EnergyFilter)
        )
        assert tuple(mesh_filter.mesh.dimension) == (1, 1, 40)
        assert any(
            value == pytest.approx(1.0e-5) for value in energy_filter.bins[:, 0]
        )
        assert any(value == pytest.approx(1.0) for value in energy_filter.bins[:, 0])
        assert any(
            value == pytest.approx(1.0e5) for value in energy_filter.bins[:, 1]
        )
    assert components["aluminium_cap_top"].center_cm == (0.0, 0.0, 35.7)
    assert components["aluminium_cap_bottom"].center_cm == (0.0, 0.0, -35.7)


def test_world_neutron_leakage_spectrum_uses_outward_surface_current() -> None:
    openmc = pytest.importorskip("openmc")
    model = build_model(load_config(CONFIG_DIR / "gen_reactor.yaml"))
    tallies = {tally.name: tally for tally in model.tallies}

    leakage = tallies["world_neutron_leakage_spectrum"]
    assert leakage.scores == ["current"]
    assert any(isinstance(item, openmc.SurfaceFilter) for item in leakage.filters)
    assert any(isinstance(item, openmc.EnergyFilter) for item in leakage.filters)
    assert any(isinstance(item, openmc.ParticleFilter) for item in leakage.filters)
    for component_name in (
        "detector_12cm_he3_gas",
        "detector_10cm_he3_gas",
        "detector_11cm_he3_gas",
        "detector_13cm_he3_gas",
    ):
        assert f"he3_n_p_spectrum_{component_name}" in tallies
        assert f"he3_neutron_spectrum_{component_name}" in tallies

    for position, expected_cell in (
        ((0.0, 0.0, 35.5), "aluminium_cap_top"),
        ((0.0, 0.0, -35.5), "aluminium_cap_bottom"),
    ):
        assert model.geometry.find(position)[-1].name == expected_cell
    for prefix, center, direction in (
        ("detector_10cm", (25.0, 0.0), (1.0, 0.0)),
        ("detector_11cm", (0.0, 25.0), (0.0, 1.0)),
        ("detector_12cm", (-25.0, 0.0), (-1.0, 0.0)),
        ("detector_13cm", (0.0, -25.0), (0.0, -1.0)),
    ):
        for distance, suffix in (
            (0.0, "he3_gas"),
            (1.22, "aluminium_wall"),
            (1.4, "lead"),
            (2.0, "hdpe"),
        ):
            position = (
                center[0] + direction[0] * distance,
                center[1] + direction[1] * distance,
                0.0,
            )
            assert model.geometry.find(position)[-1].name == f"{prefix}_{suffix}"

from pathlib import Path

import pytest

from iec_neutrons.config import ConfigError, load_config


EXAMPLE = Path(__file__).parents[1] / "config" / "example.yaml"
CONFIG_DIR = EXAMPLE.parent


def test_example_configuration_loads() -> None:
    config = load_config(EXAMPLE)

    assert config.source.energy_mev == 14.1
    assert [layer.name for layer in config.geometry.layers] == ["lead", "polyethylene"]
    assert config.materials["polyethylene"].thermal_scattering == ("c_H_in_CH2",)
    assert config.tally.energy_bins_mev[0] == 0.0
    assert config.visualization.mesh_bins == 100
    assert config.visualization.voxel_bins == 64
    assert config.visualization.he3_axial_bins == 40
    assert not config.physics.photon_transport


def test_visualization_settings_default_for_existing_configs(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8").split("\nvisualization:", maxsplit=1)[0]
    path = tmp_path / "existing.yaml"
    path.write_text(text, encoding="utf-8")

    assert load_config(path).visualization.mesh_bins == 100


def test_missing_layer_material_is_rejected(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8").replace(
        "material: polyethylene", "material: missing"
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="undefined geometry materials"):
        load_config(path)


def test_non_nested_layer_geometry_is_rejected(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8").replace(
        "radius_cm: 15.0", "radius_cm: 14.0"
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="preceding cylinder radius"):
        load_config(path)


def test_invalid_visualization_mesh_resolution_is_rejected(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8").split("\nvisualization:", maxsplit=1)[0]
    text += "\nvisualization:\n  mesh_bins: 0\n"
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="visualization.mesh_bins"):
        load_config(path)


def test_invalid_he3_axial_resolution_is_rejected(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8").split("\nvisualization:", maxsplit=1)[0]
    text += "\nvisualization:\n  he3_axial_bins: 0\n"
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="visualization.he3_axial_bins"):
        load_config(path)


def test_marginal_coordinates_can_cover_the_world_radius(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    text += "\nvisualization:\n  marginal_x_cm: 30.0\n  marginal_y_cm: -30.0\n  marginal_z_cm: 30.0\n"
    path = tmp_path / "world_marginals.yaml"
    path.write_text(text, encoding="utf-8")

    config = load_config(path)

    assert config.visualization.marginal_x_cm == 30.0
    assert config.visualization.marginal_y_cm == -30.0
    assert config.visualization.marginal_z_cm == 30.0


@pytest.mark.parametrize(
    (
        "filename",
        "expected_thicknesses_mm",
        "expected_layer_heights",
        "energy_min_mev",
        "energy_max_mev",
    ),
    [
        ("reactorpath1.yaml", [7.3, 2.9, 30.6, 11.0, 15.0, 3.0, 120.0], [60.0] * 6 + [100.0], 0.1, 16.0),
        ("reactorpath2.yaml", [38.0, 14.0, 1.5, 1.5, 7.0, 1.5, 3.5, 3.0, 80.0], [60.0] * 8 + [70.0], 0.1, 14.2),
        ("reactorpath3.yaml", [40.5, 25.5, 1.0, 3.0, 30.0], [60.0] * 5, 1.0, 16.0),
    ],
)
def test_reactorpath_configs(
    filename: str,
    expected_thicknesses_mm: list[float],
    expected_layer_heights: list[float],
    energy_min_mev: float,
    energy_max_mev: float,
) -> None:
    config = load_config(CONFIG_DIR / filename)

    assert [layer.thickness_cm * 10 for layer in config.geometry.layers] == pytest.approx(
        expected_thicknesses_mm
    )
    assert [layer.height_cm for layer in config.geometry.layers] == expected_layer_heights
    assert config.source.radius_cm == 0.5
    assert config.source.height_cm == 35.0
    assert config.source.energy_mev == 14.1
    assert config.tally.energy_bins_mev[0] == pytest.approx(energy_min_mev)
    assert config.tally.energy_bins_mev[-1] == pytest.approx(energy_max_mev)
    assert len(config.tally.energy_bins_mev) == pytest.approx(
        (energy_max_mev - energy_min_mev) / 0.1 + 1
    )
    assert config.tally.energy_bins_mev[1] - config.tally.energy_bins_mev[0] == pytest.approx(0.1)
    assert config.settings.batches == 10
    assert config.detector.width_x_cm == 0.4
    assert config.detector.width_y_cm == 0.4
    assert config.detector.thickness_z_cm == 0.05


def test_gen_dd_reactor_loads_as_pure_deuterium() -> None:
    config = load_config(CONFIG_DIR / "genDD_reactor.yaml")

    assert config.custom_source is not None
    assert config.custom_source.tritium_fraction == 0.0
    assert (
        config.custom_source.zone_beam_target,
        config.custom_source.zone_beam_wall,
        config.custom_source.zone_beam_beam,
    ) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("filename", "expected_centers"),
    [
        ("genDT_borated_jacket.yaml", ((0.0, 0.0),)),
        ("genDT_dual_borated_jacket.yaml", ((0.0, 0.0), (100.0, 0.0))),
    ],
)
def test_borated_jacket_configurations(
    filename: str, expected_centers: tuple[tuple[float, float], ...]
) -> None:
    config = load_config(CONFIG_DIR / filename)

    assert config.custom_source is not None
    assert config.custom_source.reactor_centers_cm == expected_centers
    assert not config.detector.enabled
    assert [layer.thickness_cm for layer in config.geometry.layers] == pytest.approx(
        [1.4, 1.5, 0.3, 15.0, 5.0, 0.3, 50.0]
    )
    assert [layer.height_cm for layer in config.geometry.layers] == [
        70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 100.0
    ]
    assert [layer.material for layer in config.geometry.layers[3:6]] == [
        "hdpe", "borated_polyethylene", "lead"
    ]
    gap = next(
        component for component in config.geometry.components
        if component.name in {"jacket_gap", "jacket_1_gap"}
    )
    assert gap.center_cm[0] == pytest.approx(13.65)
    assert gap.dimensions_cm == pytest.approx((27.3, 14.0, 90.0))
    assert gap.carve_from_layers == (
        "hdpe_jacket", "borated_polyethylene_jacket", "lead_jacket"
    )
    if filename == "genDT_dual_borated_jacket.yaml":
        second_gap = next(
            component for component in config.geometry.components
            if component.name == "jacket_2_gap"
        )
        assert second_gap.center_cm[0] == pytest.approx(86.35)
        assert second_gap.carve_from_components == (
            "reactor_2_hdpe",
            "reactor_2_borated_polyethylene",
            "reactor_2_lead",
        )
    borated = config.materials["borated_polyethylene"]
    assert borated.elements == pytest.approx(
        {"B": 0.0222626, "C": 0.325893, "H": 0.6518444}
    )
    assert len(
        [component for component in config.geometry.components if component.score_spectrum]
    ) == 4


def test_beam_beam_source_rejects_pure_tritium(tmp_path: Path) -> None:
    text = (CONFIG_DIR / "genDD_reactor.yaml").read_text(encoding="utf-8")
    path = tmp_path / "pure_tritium.yaml"
    path.write_text(text.replace("tritium_fraction: 0.0", "tritium_fraction: 1.0"))

    with pytest.raises(ConfigError, match="zero is valid for pure D-D"):
        load_config(path)


def test_external_he3_detector_example_uses_provisional_components() -> None:
    config = load_config(CONFIG_DIR / "he3_detector.yaml")

    assert config.physics.photon_transport
    assert config.detector.x_cm == 30.0
    assert [component.name for component in config.geometry.components] == [
        "hdpe_moderator",
        "he3_tube_wall",
        "he3_gas",
    ]
    assert config.geometry.components[0].dimensions_cm == (8.0, 8.0, 40.0)
    assert config.geometry.components[1].axis == "z"
    assert config.materials["helium3_gas"].nuclides == {"He3": 1.0}


def test_detector_may_be_outside_reactor_but_inside_world(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8").replace(
        "  world_radius_cm: 35.0", "  world_radius_cm: 70.0"
    )
    text += (
        "\ndetector:\n"
        "  x_cm: 30.0\n"
        "  y_cm: 0.0\n"
        "  z_cm: 0.0\n"
        "  width_x_cm: 0.4\n"
        "  width_y_cm: 0.4\n"
        "  thickness_z_cm: 0.05\n"
    )
    path = tmp_path / "external_detector.yaml"
    path.write_text(text, encoding="utf-8")

    assert load_config(path).detector.x_cm == 30.0

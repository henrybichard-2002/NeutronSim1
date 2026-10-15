from pathlib import Path

import pytest

from iec_neutrons.config import ConfigError, load_config


EXAMPLE = Path(__file__).parents[1] / "config" / "example.yaml"
CONFIG_DIR = EXAMPLE.parent


def test_example_configuration_loads() -> None:
    config = load_config(EXAMPLE)

    assert config.source.energy_mev == 14.1
    assert [layer.name for layer in config.geometry.layers] == ["polyethylene", "lead"]
    assert config.materials["polyethylene"].thermal_scattering == ("c_H_in_CH2",)
    assert config.tally.energy_bins_mev[0] == 0.0
    assert config.visualization.mesh_bins == 100


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

    with pytest.raises(ConfigError, match="undefined layer materials"):
        load_config(path)


def test_non_nested_layer_geometry_is_rejected(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8").replace(
        "radius_cm: 20.0", "radius_cm: 14.0"
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
    ("filename", "expected_thicknesses_mm"),
    [
        ("reactorpath1.yaml", [7.3, 2.9, 30.6, 11.0, 15.0, 3.0, 30.0]),
        ("reactorpath2.yaml", [38.0, 14.0, 1.5, 1.5, 7.0, 1.5, 3.5, 3.0, 30.0]),
        ("reactorpath3.yaml", [40.5, 25.5, 1.0, 3.0, 30.0]),
    ],
)
def test_reactorpath_configs(
    filename: str, expected_thicknesses_mm: list[float]
) -> None:
    config = load_config(CONFIG_DIR / filename)

    assert [layer.thickness_cm * 10 for layer in config.geometry.layers] == pytest.approx(
        expected_thicknesses_mm
    )
    assert all(layer.height_cm == 60.0 for layer in config.geometry.layers)
    assert config.source.radius_cm == 0.5
    assert config.source.height_cm == 35.0
    assert config.source.energy_mev == 14.1
    assert len(config.tally.energy_bins_mev) == 151
    assert config.tally.energy_bins_mev[0] == pytest.approx(1.0)
    assert config.tally.energy_bins_mev[-1] == pytest.approx(16.0)
    assert config.tally.energy_bins_mev[1] - config.tally.energy_bins_mev[0] == pytest.approx(0.1)
    assert config.settings.batches == 10
    assert config.detector.width_x_cm == 0.4
    assert config.detector.width_y_cm == 0.4
    assert config.detector.thickness_z_cm == 0.05

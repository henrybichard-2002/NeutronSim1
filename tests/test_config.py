from pathlib import Path

import pytest

from iec_neutrons.config import ConfigError, load_config


EXAMPLE = Path(__file__).parents[1] / "config" / "example.yaml"


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
    text = EXAMPLE.read_text(encoding="utf-8").replace("mesh_bins: 100", "mesh_bins: 0")
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="visualization.mesh_bins"):
        load_config(path)

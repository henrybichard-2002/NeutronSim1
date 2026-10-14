"""Load and validate YAML simulation configurations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a simulation configuration is invalid."""


@dataclass(frozen=True)
class SourceConfig:
    radius_cm: float
    height_cm: float
    energy_mev: float


@dataclass(frozen=True)
class LayerConfig:
    name: str
    radius_cm: float
    height_cm: float
    material: str


@dataclass(frozen=True)
class GeometryConfig:
    cavity_radius_cm: float
    cavity_height_cm: float
    layers: tuple[LayerConfig, ...]
    world_radius_cm: float


@dataclass(frozen=True)
class MaterialConfig:
    density_g_cm3: float
    elements: dict[str, float] | None
    nuclides: dict[str, float] | None
    thermal_scattering: tuple[str, ...]


@dataclass(frozen=True)
class SettingsConfig:
    particles: int
    batches: int
    inactive: int
    seed: int | None


@dataclass(frozen=True)
class TallyConfig:
    energy_bins_mev: tuple[float, ...]


@dataclass(frozen=True)
class SimulationConfig:
    source: SourceConfig
    geometry: GeometryConfig
    materials: dict[str, MaterialConfig]
    settings: SettingsConfig
    tally: TallyConfig


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _keys(data: dict[str, Any], required: set[str], optional: set[str], path: str) -> None:
    missing = required - data.keys()
    unknown = data.keys() - required - optional
    if missing:
        raise ConfigError(f"{path} is missing required keys: {', '.join(sorted(missing))}")
    if unknown:
        raise ConfigError(f"{path} has unknown keys: {', '.join(sorted(unknown))}")


def _number(value: Any, path: str, *, minimum: float, exclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or (result <= minimum if exclusive else result < minimum):
        comparison = "greater than" if exclusive else "at least"
        raise ConfigError(f"{path} must be {comparison} {minimum}")
    return result


def _positive_int(value: Any, path: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{path} must be an integer of at least {minimum}")
    return value


def _composition(value: Any, path: str) -> dict[str, float]:
    raw = _mapping(value, path)
    if not raw:
        raise ConfigError(f"{path} must contain at least one entry")
    return {
        str(name): _number(amount, f"{path}.{name}", minimum=0, exclusive=True)
        for name, amount in raw.items()
    }


def load_config(path: str | Path) -> SimulationConfig:
    """Read a YAML file and return a validated immutable configuration."""
    try:
        raw_config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML: {exc}") from exc
    root = _mapping(raw_config, "configuration")
    _keys(root, {"source", "geometry", "materials", "settings", "tally"}, set(), "configuration")

    source_data = _mapping(root["source"], "source")
    _keys(source_data, {"radius_cm", "height_cm", "energy_mev"}, set(), "source")
    source = SourceConfig(
        radius_cm=_number(source_data["radius_cm"], "source.radius_cm", minimum=0, exclusive=True),
        height_cm=_number(source_data["height_cm"], "source.height_cm", minimum=0, exclusive=True),
        energy_mev=_number(source_data["energy_mev"], "source.energy_mev", minimum=0, exclusive=True),
    )

    geometry_data = _mapping(root["geometry"], "geometry")
    _keys(geometry_data, {"cavity", "layers", "world_radius_cm"}, set(), "geometry")
    cavity_data = _mapping(geometry_data["cavity"], "geometry.cavity")
    _keys(cavity_data, {"radius_cm", "height_cm"}, set(), "geometry.cavity")
    cavity_radius = _number(cavity_data["radius_cm"], "geometry.cavity.radius_cm", minimum=0, exclusive=True)
    cavity_height = _number(cavity_data["height_cm"], "geometry.cavity.height_cm", minimum=0, exclusive=True)

    raw_layers = geometry_data["layers"]
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ConfigError("geometry.layers must be a non-empty list")
    layers: list[LayerConfig] = []
    names: set[str] = set()
    previous_radius = cavity_radius
    previous_height = cavity_height
    for index, raw_layer in enumerate(raw_layers):
        layer_path = f"geometry.layers[{index}]"
        layer_data = _mapping(raw_layer, layer_path)
        _keys(layer_data, {"name", "radius_cm", "height_cm", "material"}, set(), layer_path)
        name = layer_data["name"]
        material = layer_data["material"]
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{layer_path}.name must be a non-empty string")
        if name in names:
            raise ConfigError(f"geometry layer name {name!r} is duplicated")
        if not isinstance(material, str) or not material.strip():
            raise ConfigError(f"{layer_path}.material must be a non-empty string")
        radius = _number(layer_data["radius_cm"], f"{layer_path}.radius_cm", minimum=0, exclusive=True)
        height = _number(layer_data["height_cm"], f"{layer_path}.height_cm", minimum=0, exclusive=True)
        if radius <= previous_radius:
            raise ConfigError(f"{layer_path}.radius_cm must exceed the preceding cylinder radius")
        if height < previous_height:
            raise ConfigError(f"{layer_path}.height_cm must be at least the preceding cylinder height")
        layers.append(LayerConfig(name, radius, height, material))
        names.add(name)
        previous_radius = radius
        previous_height = height

    world_radius = _number(
        geometry_data["world_radius_cm"], "geometry.world_radius_cm", minimum=0, exclusive=True
    )
    outermost = layers[-1]
    if world_radius <= math.hypot(outermost.radius_cm, outermost.height_cm / 2):
        raise ConfigError("geometry.world_radius_cm must enclose the outermost cylinder")
    if source.radius_cm >= cavity_radius or source.height_cm > cavity_height:
        raise ConfigError("the source cylinder must fit inside geometry.cavity")
    geometry = GeometryConfig(cavity_radius, cavity_height, tuple(layers), world_radius)

    materials_data = _mapping(root["materials"], "materials")
    if not materials_data:
        raise ConfigError("materials must contain at least one material")
    materials: dict[str, MaterialConfig] = {}
    for name, raw_material in materials_data.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("material names must be non-empty strings")
        material_path = f"materials.{name}"
        material_data = _mapping(raw_material, material_path)
        _keys(
            material_data,
            {"density_g_cm3"},
            {"elements", "nuclides", "thermal_scattering"},
            material_path,
        )
        if ("elements" in material_data) == ("nuclides" in material_data):
            raise ConfigError(f"{material_path} must define exactly one of elements or nuclides")
        elements = _composition(material_data["elements"], f"{material_path}.elements") if "elements" in material_data else None
        nuclides = _composition(material_data["nuclides"], f"{material_path}.nuclides") if "nuclides" in material_data else None
        thermal = material_data.get("thermal_scattering", [])
        if not isinstance(thermal, list) or any(not isinstance(item, str) or not item.strip() for item in thermal):
            raise ConfigError(f"{material_path}.thermal_scattering must be a list of non-empty strings")
        materials[name] = MaterialConfig(
            density_g_cm3=_number(
                material_data["density_g_cm3"], f"{material_path}.density_g_cm3", minimum=0, exclusive=True
            ),
            elements=elements,
            nuclides=nuclides,
            thermal_scattering=tuple(thermal),
        )
    missing_materials = {layer.material for layer in layers} - materials.keys()
    if missing_materials:
        raise ConfigError(f"undefined layer materials: {', '.join(sorted(missing_materials))}")

    settings_data = _mapping(root["settings"], "settings")
    _keys(settings_data, {"particles", "batches", "inactive"}, {"seed"}, "settings")
    batches = _positive_int(settings_data["batches"], "settings.batches")
    inactive = _positive_int(settings_data["inactive"], "settings.inactive", minimum=0)
    if inactive >= batches:
        raise ConfigError("settings.inactive must be less than settings.batches")
    seed = settings_data.get("seed")
    if seed is not None:
        seed = _positive_int(seed, "settings.seed")
    settings = SettingsConfig(
        particles=_positive_int(settings_data["particles"], "settings.particles"),
        batches=batches,
        inactive=inactive,
        seed=seed,
    )

    tally_data = _mapping(root["tally"], "tally")
    _keys(tally_data, {"energy_bins_mev"}, set(), "tally")
    raw_bins = tally_data["energy_bins_mev"]
    if not isinstance(raw_bins, list) or len(raw_bins) < 2:
        raise ConfigError("tally.energy_bins_mev must contain at least two energy boundaries")
    bins = tuple(
        _number(edge, f"tally.energy_bins_mev[{index}]", minimum=0)
        for index, edge in enumerate(raw_bins)
    )
    if any(right <= left for left, right in zip(bins, bins[1:])):
        raise ConfigError("tally.energy_bins_mev must be strictly increasing")
    tally = TallyConfig(bins)

    return SimulationConfig(source, geometry, materials, settings, tally)

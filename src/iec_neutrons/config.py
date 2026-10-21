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
    thickness_cm: float
    height_cm: float
    material: str


@dataclass(frozen=True)
class ComponentConfig:
    name: str
    shape: str
    center_cm: tuple[float, float, float]
    dimensions_cm: tuple[float, ...]
    material: str
    axis: str | None = None
    carve_from_base: bool = True
    score_spectrum: bool = False

    @property
    def half_extents_cm(self) -> tuple[float, float, float]:
        if self.shape == "box":
            size_x, size_y, size_z = self.dimensions_cm
            return size_x / 2, size_y / 2, size_z / 2
        if self.shape == "cylinder_shell":
            _, radius, length = self.dimensions_cm
        else:
            radius, length = self.dimensions_cm

        if self.axis == "x":
            return length / 2, radius, radius
        if self.axis == "y":
            return radius, length / 2, radius
        return radius, radius, length / 2


@dataclass(frozen=True)
class DetectorConfig:
    x_cm: float
    y_cm: float
    z_cm: float
    width_x_cm: float
    width_y_cm: float
    thickness_z_cm: float
    enabled: bool = True


@dataclass(frozen=True)
class GeometryConfig:
    cavity_radius_cm: float
    cavity_height_cm: float
    layers: tuple[LayerConfig, ...]
    components: tuple[ComponentConfig, ...]
    world_radius_cm: float
    background_material: str | None = None


@dataclass(frozen=True)
class PhysicsConfig:
    photon_transport: bool = False


@dataclass(frozen=True)
class CustomSourceConfig:
    cathode_radius_cm: float
    voltage_kev: float
    neutral_temperature_k: float
    tritium_fraction: float
    zone_beam_target: float
    zone_beam_wall: float
    zone_beam_beam: float
    source_strength: float
    neutral_profile: tuple[tuple[float, float], ...]
    reactor_centers_cm: tuple[tuple[float, float], ...]


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
class VisualizationConfig:
    mesh_bins: int = 100
    voxel_bins: int = 64
    he3_axial_bins: int = 40
    slice_thickness_cm: float = 0.4
    energy_bins_mev: tuple[float, ...] = ()
    marginal_x_cm: float = 0.0
    marginal_y_cm: float = 0.0
    marginal_z_cm: float = 0.0


@dataclass(frozen=True)
class SimulationConfig:
    source: SourceConfig
    detector: DetectorConfig
    geometry: GeometryConfig
    materials: dict[str, MaterialConfig]
    settings: SettingsConfig
    tally: TallyConfig
    visualization: VisualizationConfig
    physics: PhysicsConfig
    custom_source: CustomSourceConfig | None = None


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
    _keys(
        root,
        {"source", "geometry", "materials", "settings", "tally"},
        {"visualization", "detector", "physics", "custom_source"},
        "configuration",
    )

    source_data = _mapping(root["source"], "source")
    _keys(source_data, {"radius_cm", "height_cm", "energy_mev"}, set(), "source")
    source = SourceConfig(
        radius_cm=_number(source_data["radius_cm"], "source.radius_cm", minimum=0, exclusive=True),
        height_cm=_number(source_data["height_cm"], "source.height_cm", minimum=0, exclusive=True),
        energy_mev=_number(source_data["energy_mev"], "source.energy_mev", minimum=0, exclusive=True),
    )

    geometry_data = _mapping(root["geometry"], "geometry")
    _keys(
        geometry_data,
        {"cavity", "layers", "world_radius_cm"},
        {"components", "background_material"},
        "geometry",
    )
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
        _keys(
            layer_data,
            {"name", "height_cm", "material"},
            {"radius_cm", "thickness_mm"},
            layer_path,
        )
        if ("radius_cm" in layer_data) == ("thickness_mm" in layer_data):
            raise ConfigError(
                f"{layer_path} must define exactly one of radius_cm or thickness_mm"
            )
        name = layer_data["name"]
        material = layer_data["material"]
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{layer_path}.name must be a non-empty string")
        if name in names:
            raise ConfigError(f"geometry layer name {name!r} is duplicated")
        if not isinstance(material, str) or not material.strip():
            raise ConfigError(f"{layer_path}.material must be a non-empty string")
        if "thickness_mm" in layer_data:
            thickness_cm = _number(
                layer_data["thickness_mm"],
                f"{layer_path}.thickness_mm",
                minimum=0,
                exclusive=True,
            ) / 10.0
            radius = previous_radius + thickness_cm
        else:
            radius = _number(
                layer_data["radius_cm"],
                f"{layer_path}.radius_cm",
                minimum=0,
                exclusive=True,
            )
            thickness_cm = radius - previous_radius
        height = _number(layer_data["height_cm"], f"{layer_path}.height_cm", minimum=0, exclusive=True)
        if radius <= previous_radius:
            raise ConfigError(f"{layer_path}.radius_cm must exceed the preceding cylinder radius")
        if height < previous_height:
            raise ConfigError(f"{layer_path}.height_cm must be at least the preceding cylinder height")
        layers.append(LayerConfig(name, radius, thickness_cm, height, material))
        names.add(name)
        previous_radius = radius
        previous_height = height

    world_radius = _number(
        geometry_data["world_radius_cm"], "geometry.world_radius_cm", minimum=0, exclusive=True
    )
    background_material = geometry_data.get("background_material")
    if background_material is not None and (
        not isinstance(background_material, str) or not background_material.strip()
    ):
        raise ConfigError("geometry.background_material must be a material name")
    outermost = layers[-1]
    if world_radius <= math.hypot(outermost.radius_cm, outermost.height_cm / 2):
        raise ConfigError("geometry.world_radius_cm must enclose the outermost cylinder")
    if source.radius_cm > cavity_radius or source.height_cm > cavity_height:
        raise ConfigError("the source cylinder must fit inside geometry.cavity")

    raw_components = geometry_data.get("components", [])
    if not isinstance(raw_components, list):
        raise ConfigError("geometry.components must be a list")
    components: list[ComponentConfig] = []
    for index, raw_component in enumerate(raw_components):
        component_path = f"geometry.components[{index}]"
        component_data = _mapping(raw_component, component_path)
        _keys(
            component_data,
            {"name", "shape", "center_cm", "material"},
            {"size_cm", "radius_cm", "length_cm", "axis", "carve_from_base", "score_spectrum"},
            component_path,
        )
        name = component_data["name"]
        shape = component_data["shape"]
        material = component_data["material"]
        carve_from_base = component_data.get("carve_from_base", True)
        score_spectrum = component_data.get("score_spectrum", False)
        if not isinstance(carve_from_base, bool):
            raise ConfigError(f"{component_path}.carve_from_base must be a boolean")
        if not isinstance(score_spectrum, bool):
            raise ConfigError(f"{component_path}.score_spectrum must be a boolean")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{component_path}.name must be a non-empty string")
        if name in names:
            raise ConfigError(f"geometry component name {name!r} is duplicated")
        if not isinstance(shape, str) or shape not in {"box", "cylinder", "cylinder_shell"}:
            raise ConfigError(
                f"{component_path}.shape must be 'box', 'cylinder' or 'cylinder_shell'"
            )
        if not isinstance(material, str) or not material.strip():
            raise ConfigError(f"{component_path}.material must be a non-empty string")
        center_data = component_data["center_cm"]
        if not isinstance(center_data, list) or len(center_data) != 3:
            raise ConfigError(f"{component_path}.center_cm must contain x, y and z")
        center = (
            _number(center_data[0], f"{component_path}.center_cm[x]", minimum=-world_radius),
            _number(center_data[1], f"{component_path}.center_cm[y]", minimum=-world_radius),
            _number(center_data[2], f"{component_path}.center_cm[z]", minimum=-world_radius),
        )
        if shape == "box":
            _keys(
                component_data,
                {"name", "shape", "center_cm", "material", "size_cm"},
                {"carve_from_base", "score_spectrum"},
                component_path,
            )
            size_data = component_data["size_cm"]
            if not isinstance(size_data, list) or len(size_data) != 3:
                raise ConfigError(f"{component_path}.size_cm must contain x, y and z sizes")
            dimensions = tuple(
                _number(value, f"{component_path}.size_cm[{axis}]", minimum=0, exclusive=True)
                for axis, value in zip(("x", "y", "z"), size_data, strict=True)
            )
            axis = None
            half_extents = tuple(value / 2 for value in dimensions)
        else:
            shell = shape == "cylinder_shell"
            _keys(
                component_data,
                {
                    "name", "shape", "center_cm", "material", "radius_cm",
                    "length_cm",
                } | ({"inner_radius_cm"} if shell else set()),
                {"axis", "carve_from_base", "score_spectrum"},
                component_path,
            )
            axis = component_data.get("axis", "z")
            if not isinstance(axis, str) or axis not in {"x", "y", "z"}:
                raise ConfigError(f"{component_path}.axis must be 'x', 'y' or 'z'")
            radius = _number(
                component_data["radius_cm"],
                f"{component_path}.radius_cm",
                minimum=0,
                exclusive=True,
            )
            length = _number(
                component_data["length_cm"],
                f"{component_path}.length_cm",
                minimum=0,
                exclusive=True,
            )
            if shell:
                inner_radius = _number(
                    component_data["inner_radius_cm"],
                    f"{component_path}.inner_radius_cm",
                    minimum=0,
                    exclusive=True,
                )
                if inner_radius >= radius:
                    raise ConfigError(
                        f"{component_path}.inner_radius_cm must be less than radius_cm"
                    )
                dimensions = (inner_radius, radius, length)
            else:
                dimensions = (radius, length)
            half_extents = tuple(
                length / 2 if coordinate == axis else radius for coordinate in ("x", "y", "z")
            )
        if math.sqrt(
            sum((abs(position) + extent) ** 2 for position, extent in zip(center, half_extents, strict=True))
        ) >= world_radius:
            raise ConfigError(f"{component_path} must fit inside geometry.world_radius_cm")
        components.append(
            ComponentConfig(
                name, shape, center, dimensions, material, axis, carve_from_base,
                score_spectrum
            )
        )
        names.add(name)

    geometry = GeometryConfig(
        cavity_radius, cavity_height, tuple(layers), tuple(components), world_radius,
        background_material
    )

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
    missing_materials = (
        {layer.material for layer in layers}
        | {component.material for component in components}
        | ({background_material} if background_material is not None else set())
    ) - materials.keys() - {"void"}
    if missing_materials:
        raise ConfigError(f"undefined geometry materials: {', '.join(sorted(missing_materials))}")

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
    range_keys = {"energy_min_mev", "energy_max_mev", "energy_bin_width_mev"}
    explicit_edges = "energy_bins_mev" in tally_data
    if explicit_edges:
        _keys(tally_data, {"energy_bins_mev"}, set(), "tally")
        raw_bins = tally_data["energy_bins_mev"]
        if not isinstance(raw_bins, list) or len(raw_bins) < 2:
            raise ConfigError("tally.energy_bins_mev must contain at least two energy boundaries")
        bins = tuple(
            _number(edge, f"tally.energy_bins_mev[{index}]", minimum=0)
            for index, edge in enumerate(raw_bins)
        )
    else:
        _keys(tally_data, range_keys, set(), "tally")
        energy_min = _number(tally_data["energy_min_mev"], "tally.energy_min_mev", minimum=0)
        energy_max = _number(
            tally_data["energy_max_mev"],
            "tally.energy_max_mev",
            minimum=energy_min,
            exclusive=True,
        )
        bin_width = _number(
            tally_data["energy_bin_width_mev"],
            "tally.energy_bin_width_mev",
            minimum=0,
            exclusive=True,
        )
        bin_count = round((energy_max - energy_min) / bin_width)
        if bin_count < 1 or not math.isclose(
            energy_min + bin_count * bin_width, energy_max, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ConfigError(
                "tally energy range must be an exact multiple of energy_bin_width_mev"
            )
        bins = tuple(
            energy_min + index * bin_width for index in range(bin_count)
        ) + (energy_max,)
    if any(right <= left for left, right in zip(bins, bins[1:])):
        raise ConfigError("tally.energy_bins_mev must be strictly increasing")
    tally = TallyConfig(bins)

    visualization_data = _mapping(root.get("visualization", {}), "visualization")
    _keys(
        visualization_data,
        set(),
        {
            "mesh_bins",
            "voxel_bins",
            "he3_axial_bins",
            "slice_thickness_cm",
            "energy_bins_mev",
            "marginal_x_cm",
            "marginal_y_cm",
            "marginal_z_cm",
        },
        "visualization",
    )
    map_edges_raw = visualization_data.get("energy_bins_mev", list(bins))
    if not isinstance(map_edges_raw, list) or len(map_edges_raw) < 2:
        raise ConfigError("visualization.energy_bins_mev must contain at least two boundaries")
    map_edges = tuple(
        _number(edge, f"visualization.energy_bins_mev[{index}]", minimum=0)
        for index, edge in enumerate(map_edges_raw)
    )
    if any(right <= left for left, right in zip(map_edges, map_edges[1:])):
        raise ConfigError("visualization.energy_bins_mev must be strictly increasing")
    if map_edges[0] < bins[0] or map_edges[-1] > bins[-1]:
        raise ConfigError("visualization.energy_bins_mev must fit within the tally energy range")
    visualization = VisualizationConfig(
        mesh_bins=_positive_int(
            visualization_data.get("mesh_bins", 100), "visualization.mesh_bins"
        ),
        voxel_bins=_positive_int(
            visualization_data.get("voxel_bins", 64), "visualization.voxel_bins"
        ),
        he3_axial_bins=_positive_int(
            visualization_data.get("he3_axial_bins", 40),
            "visualization.he3_axial_bins",
        ),
        slice_thickness_cm=_number(
            visualization_data.get("slice_thickness_cm", 0.4),
            "visualization.slice_thickness_cm",
            minimum=0,
            exclusive=True,
        ),
        energy_bins_mev=map_edges,
        marginal_x_cm=_number(
            visualization_data.get("marginal_x_cm", 0.0),
            "visualization.marginal_x_cm",
            minimum=-world_radius,
        ),
        marginal_y_cm=_number(
            visualization_data.get("marginal_y_cm", 0.0),
            "visualization.marginal_y_cm",
            minimum=-world_radius,
        ),
        marginal_z_cm=_number(
            visualization_data.get("marginal_z_cm", 0.0),
            "visualization.marginal_z_cm",
            minimum=-world_radius,
        ),
    )
    if abs(visualization.marginal_x_cm) > world_radius:
        raise ConfigError("visualization.marginal_x_cm must lie within the world radius")
    if abs(visualization.marginal_y_cm) > world_radius:
        raise ConfigError("visualization.marginal_y_cm must lie within the world radius")
    if abs(visualization.marginal_z_cm) > world_radius:
        raise ConfigError("visualization.marginal_z_cm must lie within the world radius")

    detector_data = _mapping(root.get("detector", {}), "detector")
    _keys(
        detector_data,
        set(),
        {"x_cm", "y_cm", "z_cm", "width_x_cm", "width_y_cm", "thickness_z_cm", "enabled"},
        "detector",
    )
    detector = DetectorConfig(
        x_cm=_number(detector_data.get("x_cm", 0.0), "detector.x_cm", minimum=-world_radius),
        y_cm=_number(detector_data.get("y_cm", 0.0), "detector.y_cm", minimum=-world_radius),
        z_cm=_number(detector_data.get("z_cm", 0.0), "detector.z_cm", minimum=-world_radius),
        width_x_cm=_number(
            detector_data.get("width_x_cm", 0.4),
            "detector.width_x_cm",
            minimum=0,
            exclusive=True,
        ),
        width_y_cm=_number(
            detector_data.get("width_y_cm", 0.4),
            "detector.width_y_cm",
            minimum=0,
            exclusive=True,
        ),
        thickness_z_cm=_number(
            detector_data.get("thickness_z_cm", 0.05),
            "detector.thickness_z_cm",
            minimum=0,
            exclusive=True,
        ),
        enabled=detector_data.get("enabled", True),
    )
    if not isinstance(detector.enabled, bool):
        raise ConfigError("detector.enabled must be a boolean")
    detector_extent = (
        detector.width_x_cm / 2,
        detector.width_y_cm / 2,
        detector.thickness_z_cm / 2,
    )
    if detector.enabled and math.sqrt(
        sum(
            (abs(position) + extent) ** 2
            for position, extent in zip(
                (detector.x_cm, detector.y_cm, detector.z_cm),
                detector_extent,
                strict=True,
            )
        )
    ) >= world_radius:
        raise ConfigError("detector volume must fit inside geometry.world_radius_cm")

    physics_data = _mapping(root.get("physics", {}), "physics")
    _keys(physics_data, set(), {"photon_transport"}, "physics")
    photon_transport = physics_data.get("photon_transport", False)
    if not isinstance(photon_transport, bool):
        raise ConfigError("physics.photon_transport must be a boolean")
    physics = PhysicsConfig(photon_transport=photon_transport)

    custom_source = None
    if "custom_source" in root:
        custom_data = _mapping(root["custom_source"], "custom_source")
        _keys(
            custom_data,
            {
                "cathode_radius_cm",
                "voltage_kev",
                "neutral_temperature_k",
                "tritium_fraction",
                "zone_beam_target",
                "zone_beam_wall",
                "zone_beam_beam",
                "source_strength",
                "neutral_profile",
            },
            {"reactor_centers_cm"},
            "custom_source",
        )
        cathode_radius = _number(
            custom_data["cathode_radius_cm"],
            "custom_source.cathode_radius_cm",
            minimum=0,
            exclusive=True,
        )
        if cathode_radius >= geometry.cavity_radius_cm:
            raise ConfigError(
                "custom_source.cathode_radius_cm must be smaller than "
                "geometry.cavity.radius_cm"
            )
        if not math.isclose(source.radius_cm, geometry.cavity_radius_cm):
            raise ConfigError(
                "source.radius_cm must match geometry.cavity.radius_cm when "
                "custom_source is configured"
            )
        tritium_fraction = _number(
            custom_data["tritium_fraction"],
            "custom_source.tritium_fraction",
            minimum=0,
        )
        if tritium_fraction > 1:
            raise ConfigError("custom_source.tritium_fraction must not exceed 1")
        zone_weights = {
            key: _number(custom_data[key], f"custom_source.{key}", minimum=0)
            for key in ("zone_beam_target", "zone_beam_wall", "zone_beam_beam")
        }
        if not any(zone_weights.values()):
            raise ConfigError("custom_source must enable at least one reaction zone")
        if zone_weights["zone_beam_beam"] > 0 and tritium_fraction >= 1:
            raise ConfigError(
                "custom_source.tritium_fraction must be less than 1 when "
                "the beam-beam zone is enabled; zero is valid for pure D-D"
            )
        raw_profile = custom_data["neutral_profile"]
        if not isinstance(raw_profile, list) or len(raw_profile) < 2:
            raise ConfigError(
                "custom_source.neutral_profile must contain at least two "
                "[radius_fraction, relative_density] points"
            )
        profile = tuple(
            (
                _number(
                    point[0],
                    f"custom_source.neutral_profile[{index}][0]",
                    minimum=0,
                ),
                _number(
                    point[1],
                    f"custom_source.neutral_profile[{index}][1]",
                    minimum=0,
                ),
            )
            for index, point in enumerate(raw_profile)
            if isinstance(point, list) and len(point) == 2
        )
        if len(profile) != len(raw_profile):
            raise ConfigError(
                "each custom_source.neutral_profile point must be a pair"
            )
        if (
            profile[0][0] != 0
            or profile[-1][0] != 1
            or any(
                right[0] <= left[0]
                for left, right in zip(profile, profile[1:])
            )
        ):
            raise ConfigError(
                "custom_source.neutral_profile radius fractions must increase "
                "from 0 to 1"
            )
        custom_source = CustomSourceConfig(
            cathode_radius_cm=cathode_radius,
            voltage_kev=_number(
                custom_data["voltage_kev"],
                "custom_source.voltage_kev",
                minimum=0,
                exclusive=True,
            ),
            neutral_temperature_k=_number(
                custom_data["neutral_temperature_k"],
                "custom_source.neutral_temperature_k",
                minimum=0,
            ),
            tritium_fraction=tritium_fraction,
            **zone_weights,
            source_strength=_number(
                custom_data["source_strength"],
                "custom_source.source_strength",
                minimum=0,
                exclusive=True,
            ),
            neutral_profile=profile,
            reactor_centers_cm=tuple(
                (
                    _number(point[0], f"custom_source.reactor_centers_cm[{index}][0]", minimum=-world_radius),
                    _number(point[1], f"custom_source.reactor_centers_cm[{index}][1]", minimum=-world_radius),
                )
                for index, point in enumerate(custom_data.get("reactor_centers_cm", [[0.0, 0.0]]))
                if isinstance(point, list) and len(point) == 2
            ),
        )
        raw_centers = custom_data.get("reactor_centers_cm", [[0.0, 0.0]])
        if not isinstance(raw_centers, list) or not raw_centers or len(custom_source.reactor_centers_cm) != len(raw_centers):
            raise ConfigError("custom_source.reactor_centers_cm must contain [x_cm, y_cm] pairs")
        if any(
            math.sqrt(
                (abs(x_cm) + source.radius_cm) ** 2
                + (abs(y_cm) + source.radius_cm) ** 2
                + (source.height_cm / 2) ** 2
            )
            >= world_radius
            for x_cm, y_cm in custom_source.reactor_centers_cm
        ):
            raise ConfigError(
                "custom_source reactor source volumes must fit inside geometry.world_radius_cm"
            )

    return SimulationConfig(
        source=source,
        detector=detector,
        geometry=geometry,
        materials=materials,
        settings=settings,
        tally=tally,
        visualization=visualization,
        physics=physics,
        custom_source=custom_source,
    )

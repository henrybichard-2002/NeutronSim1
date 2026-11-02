"""YAML configuration for the diamond detector response simulation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class ConfigError(ValueError):
    """Raised for invalid configuration files."""


@dataclass
class SupportConfig:
    """Optional slab behind the diamond (PCB, ceramic, holder) that scatters neutrons back."""

    elements: dict[str, float] = field(default_factory=lambda: {"Al": 2.0, "O": 3.0})
    density_g_cm3: float = 3.95
    thickness_mm: float = 1.0
    lateral_mm: tuple[float, float] = (10.0, 10.0)
    gap_um: float = 0.0


@dataclass
class DetectorConfig:
    width_mm: tuple[float, float] = (4.0, 4.0)
    thickness_um: float = 500.0
    density_g_cm3: float = 3.515
    # Parts of the diamond (metal contacts, damaged surface layers) that do not collect charge.
    dead_layer_front_um: float = 0.0
    dead_layer_back_um: float = 0.0
    dead_layer_lateral_um: float = 0.0
    # Angle between the beam and the detector normal (rotation about the y axis).
    incidence_angle_deg: float = 0.0
    support: SupportConfig | None = None


@dataclass
class SourceConfig:
    # Uniform parallel beam. The response is expressed per unit neutron fluence.
    beam_margin_mm: float = 3.0
    distance_cm: float = 5.0


@dataclass
class GridSegment:
    start_mev: float
    stop_mev: float
    step_mev: float


@dataclass
class IncidentGridConfig:
    segments: list[GridSegment] = field(
        default_factory=lambda: [
            GridSegment(1.0, 11.0, 1.0),
            GridSegment(11.0, 13.0, 0.25),
            GridSegment(13.0, 15.5, 0.1),
            GridSegment(15.5, 17.0, 0.25),
        ]
    )
    # "uniform": incident energy sampled uniformly inside each bin (bin-averaged response).
    # "center": mono-energetic at the bin centre.
    sampling: str = "uniform"


@dataclass
class DepositedConfig:
    max_mev: float = 16.0
    bin_kev: float = 10.0


@dataclass
class StatisticsConfig:
    # One OpenMC batch per chunk: particle ids repeat between batches, so histories could not be
    # told apart. The source bank limits a chunk to a few 1e7 histories.
    histories_per_chunk: float = 2.0e7
    min_chunks: int = 4
    max_chunks: int = 200
    # Stop adding chunks to a column once the relative uncertainty of the pulse count above
    # roi_min_mev falls below this value.
    target_rel_error: float = 0.01
    roi_min_mev: float = 0.5


@dataclass
class PhysicsConfig:
    cross_sections: str | None = None
    temperature_k: float = 294.0
    energy_cutoff_ev: float = 1.0e3
    # Probability of each 9Be level in 12C(n,alpha)9Be* (excitation energy in MeV). Levels that
    # are energetically closed at a given neutron energy are dropped and the rest renormalised.
    # JEFF-4.0 only provides the summed (n,alpha) cross section, so these are an assumption.
    alpha_levels: dict[float, float] = field(
        default_factory=lambda: {0.0: 0.60, 1.684: 0.12, 2.429: 0.18, 3.049: 0.05, 4.704: 0.05}
    )
    nuclear_stopping: bool = True
    pulse_window_ns: float = 10.0


@dataclass
class ResolutionConfig:
    # FWHM(E) = sqrt(noise^2 + (relative * E)^2)
    noise_kev: float = 40.0
    relative: float = 0.008


@dataclass
class SimulationConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    incident: IncidentGridConfig = field(default_factory=IncidentGridConfig)
    deposited: DepositedConfig = field(default_factory=DepositedConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    seed: int = 1


_NESTED = {
    "detector": DetectorConfig,
    "source": SourceConfig,
    "incident": IncidentGridConfig,
    "deposited": DepositedConfig,
    "statistics": StatisticsConfig,
    "physics": PhysicsConfig,
    "resolution": ResolutionConfig,
    "support": SupportConfig,
}


def _build(cls, data: Any, path: str):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping")
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"unknown key(s) in {path}: {', '.join(sorted(map(str, unknown)))}")
    kwargs = {}
    for name, value in data.items():
        where = f"{path}.{name}"
        if name in _NESTED:
            kwargs[name] = None if value is None else _build(_NESTED[name], value, where)
        elif name == "segments":
            if not isinstance(value, list) or not value:
                raise ConfigError(f"{where} must be a non-empty list")
            kwargs[name] = [_build(GridSegment, item, f"{where}[{i}]") for i, item in enumerate(value)]
        elif name == "alpha_levels":
            kwargs[name] = {float(k): float(v) for k, v in dict(value).items()}
        elif name == "histories_per_chunk":
            try:
                kwargs[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{where} must be a number") from exc
        elif name in {"width_mm", "lateral_mm"}:
            kwargs[name] = tuple(float(v) for v in value)
        else:
            kwargs[name] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def validate(config: SimulationConfig) -> None:
    d = config.detector
    if len(d.width_mm) != 2 or min(d.width_mm) <= 0 or d.thickness_um <= 0 or d.density_g_cm3 <= 0:
        raise ConfigError("detector dimensions and density must be positive")
    if d.dead_layer_front_um + d.dead_layer_back_um >= d.thickness_um:
        raise ConfigError("dead layers consume the whole detector thickness")
    if not -89.0 <= d.incidence_angle_deg <= 89.0:
        raise ConfigError("detector.incidence_angle_deg must be within +-89 degrees")
    if config.incident.sampling not in {"uniform", "center"}:
        raise ConfigError("incident.sampling must be 'uniform' or 'center'")
    segs = config.incident.segments
    for seg in segs:
        if not 0 < seg.start_mev < seg.stop_mev <= 20.0 or seg.step_mev <= 0:
            raise ConfigError(f"invalid grid segment {seg}")
    for i, a in enumerate(segs):
        for b in segs[i + 1 :]:
            if a.start_mev < b.stop_mev and b.start_mev < a.stop_mev:
                raise ConfigError("incident grid segments overlap")
    if config.deposited.bin_kev <= 0 or config.deposited.max_mev <= 0:
        raise ConfigError("deposited-energy binning must be positive")
    s = config.statistics
    if s.histories_per_chunk < 1e4 or s.min_chunks < 2 or s.max_chunks < s.min_chunks:
        raise ConfigError("statistics: need histories_per_chunk >= 1e4 and 2 <= min_chunks <= max_chunks")
    if s.histories_per_chunk > 5e7:
        raise ConfigError("statistics.histories_per_chunk above 5e7 would exhaust memory; use more chunks")
    levels = config.physics.alpha_levels
    if not levels or min(levels.values()) < 0 or sum(levels.values()) <= 0:
        raise ConfigError("physics.alpha_levels needs non-negative weights with a positive sum")


def incident_edges_mev(config: SimulationConfig) -> np.ndarray:
    """Sorted incident-energy bin edges (MeV) from the configured segments."""
    edges: list[float] = []
    for seg in config.incident.segments:
        span = seg.stop_mev - seg.start_mev
        count = int(round(span / seg.step_mev))
        if count < 1 or abs(count * seg.step_mev - span) > 1e-9:
            raise ConfigError(f"segment {seg} is not an integer number of steps")
        edges.extend(seg.start_mev + seg.step_mev * np.arange(count + 1))
    return np.unique(np.round(np.asarray(edges), 9))


def deposited_edges_mev(config: SimulationConfig) -> np.ndarray:
    n = int(round(config.deposited.max_mev * 1000.0 / config.deposited.bin_kev))
    return np.linspace(0.0, n * config.deposited.bin_kev / 1000.0, n + 1)


def load_config(path: str | Path) -> SimulationConfig:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    config = _build(SimulationConfig, data, "config")
    validate(config)
    return config

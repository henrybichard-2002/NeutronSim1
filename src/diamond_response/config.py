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
class ReactorConfig:
    """Place the diamond next to an iec_neutrons reactor model and fold the response with its neutron field."""

    config: str = ""                      # iec_neutrons YAML, relative to this file or the working directory
    distance_from_surface_cm: float = 5.0  # radial distance from the normalisation surface to the diamond centre
    z_cm: float = 0.0                     # axial position (0 = middle of the cathode)
    particles: float = 1.0e6              # source histories per batch
    batches: int = 100
    ring_thickness_cm: float = 0.5        # radial thickness of the axisymmetric flux-averaging ring
    ring_height_cm: float = 2.0
    angle_box_cm: float = 3.0             # side of the box used to measure the incidence-angle distribution
    angle_edges_deg: list = field(default_factory=lambda: [0.0, 20.0, 40.0, 90.0])
    angle_energy_edges_mev: list = field(default_factory=lambda: [1.0, 5.0, 10.0, 13.0, 13.8, 14.4, 15.0, 17.0])
    matrix_target_rel_error: float = 0.01
    cross_sections: str | None = None  # default: config/cross_sections/<reactor yaml stem>.xml


@dataclass
class SimulationConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    incident: IncidentGridConfig = field(default_factory=IncidentGridConfig)
    deposited: DepositedConfig = field(default_factory=DepositedConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    reactor: ReactorConfig | None = None
    seed: int = 1
    base_dir: Path | None = field(default=None, repr=False)


_NESTED = {
    "detector": DetectorConfig,
    "source": SourceConfig,
    "incident": IncidentGridConfig,
    "deposited": DepositedConfig,
    "statistics": StatisticsConfig,
    "physics": PhysicsConfig,
    "resolution": ResolutionConfig,
    "support": SupportConfig,
    "reactor": ReactorConfig,
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
        elif name in {"histories_per_chunk", "particles"}:
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
    r = config.reactor
    if r is not None:
        edges = list(r.angle_edges_deg)
        if not r.config or r.particles < 1 or r.batches < 2 or min(r.ring_thickness_cm, r.ring_height_cm, r.angle_box_cm) <= 0:
            raise ConfigError("reactor: need config, particles >= 1, batches >= 2 and positive tally sizes")
        if len(edges) < 2 or edges[0] != 0.0 or edges[-1] != 90.0 or any(b <= a for a, b in zip(edges, edges[1:])):
            raise ConfigError("reactor.angle_edges_deg must increase from 0 to 90")
        if len(r.angle_energy_edges_mev) < 2 or any(b <= a for a, b in zip(r.angle_energy_edges_mev, r.angle_energy_edges_mev[1:])):
            raise ConfigError("reactor.angle_energy_edges_mev must be increasing")
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
    config.base_dir = Path(path).resolve().parent
    validate(config)
    return config

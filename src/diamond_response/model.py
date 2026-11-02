"""OpenMC model: parallel neutron beam on a (possibly tilted) diamond plate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import SimulationConfig
from .nucdata import default_cross_sections
from .transport import local_frame

DIAMOND_MATERIAL_ID = 1
SUPPORT_MATERIAL_ID = 2


def cross_section_path(config: SimulationConfig) -> Path:
    if config.physics.cross_sections:
        return Path(config.physics.cross_sections).expanduser()
    return default_cross_sections()


def beam_half_widths_cm(config: SimulationConfig) -> tuple[float, float]:
    det = config.detector
    th = np.radians(det.incidence_angle_deg)
    w, h = det.width_mm[0] / 20.0, det.width_mm[1] / 20.0
    t = det.thickness_um * 1e-4 / 2.0
    margin = config.source.beam_margin_mm / 10.0
    hx = w * abs(np.cos(th)) + t * abs(np.sin(th)) + margin
    return float(hx), float(h + margin)


def beam_area_cm2(config: SimulationConfig) -> float:
    hx, hy = beam_half_widths_cm(config)
    return 4.0 * hx * hy


def collision_limit(config: SimulationConfig, histories: int) -> int:
    fraction = 0.12 if config.detector.support is not None else 0.04
    return int(histories * fraction) + 10_000


def build_model(config: SimulationConfig, energy_lo_mev: float, energy_hi_mev: float,
                histories: int, seed: int):
    import openmc
    import openmc.stats

    det = config.detector
    ex, ey, nrm = local_frame(det.incidence_angle_deg)
    hw, hh = det.width_mm[0] / 20.0, det.width_mm[1] / 20.0
    ht = det.thickness_um * 1e-4 / 2.0

    dia = openmc.Material(material_id=DIAMOND_MATERIAL_ID, name="sCVD diamond")
    dia.add_element("C", 1.0)
    dia.set_density("g/cm3", det.density_g_cm3)
    materials = [dia]

    def plane(axis, value):
        return openmc.Plane(a=axis[0], b=axis[1], c=axis[2], d=value)

    def box(half_x, half_y, n_lo, n_hi):
        return (+plane(ex, -half_x) & -plane(ex, half_x) & +plane(ey, -half_y) & -plane(ey, half_y)
                & +plane(nrm, n_lo) & -plane(nrm, n_hi))

    diamond_region = box(hw, hh, -ht, ht)
    cells = [openmc.Cell(name="diamond", fill=dia, region=diamond_region)]
    solid = diamond_region

    sup = det.support
    if sup is not None:
        smat = openmc.Material(material_id=SUPPORT_MATERIAL_ID, name="support")
        for element, fraction in sup.elements.items():
            smat.add_element(element, fraction)
        smat.set_density("g/cm3", sup.density_g_cm3)
        materials.append(smat)
        n0 = ht + sup.gap_um * 1e-4
        support_region = box(sup.lateral_mm[0] / 20.0, sup.lateral_mm[1] / 20.0, n0, n0 + sup.thickness_mm / 10.0)
        cells.append(openmc.Cell(name="support", fill=smat, region=support_region))
        solid = solid | support_region

    hx, hy = beam_half_widths_cm(config)
    distance = config.source.distance_cm
    outer = openmc.Sphere(r=distance + 10.0, boundary_type="vacuum")
    cells.append(openmc.Cell(name="void", region=-outer & ~solid))
    geometry = openmc.Geometry(cells)

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = 1
    settings.particles = int(histories)
    settings.seed = int(seed)
    settings.output = {"summary": False, "tallies": False}
    settings.cutoff = {"energy_neutron": config.physics.energy_cutoff_ev}
    settings.collision_track = {"max_collisions": collision_limit(config, histories)}
    if config.incident.sampling == "center" or energy_hi_mev - energy_lo_mev < 1e-9:
        energy = openmc.stats.Discrete([0.5 * (energy_lo_mev + energy_hi_mev) * 1e6], [1.0])
    else:
        energy = openmc.stats.Uniform(energy_lo_mev * 1e6, energy_hi_mev * 1e6)
    space = openmc.stats.CartesianIndependent(
        openmc.stats.Uniform(-hx, hx), openmc.stats.Uniform(-hy, hy), openmc.stats.Discrete([-distance], [1.0])
    )
    settings.source = openmc.IndependentSource(
        space=space, energy=energy, angle=openmc.stats.Monodirectional((0.0, 0.0, 1.0)), particle="neutron"
    )
    return openmc.Model(geometry, openmc.Materials(materials), settings)

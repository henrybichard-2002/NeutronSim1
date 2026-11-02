"""Nuclear data access: discrete-level energies and partial cross sections for C12/C13."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Absorption-type reactions that are resolved by sampling from partial cross sections.
ABSORPTION_MTS = (102, 103, 104, 107)


@dataclass
class NuclideData:
    name: str
    awr: float
    level_ex_mev: dict[int, float]  # MT 51..90 -> excitation energy (MeV)
    energy_ev: np.ndarray
    xs: dict[int, np.ndarray]  # MT -> cross section on energy_ev grid (barn)

    def absorption_xs(self, energy_mev: np.ndarray) -> dict[int, np.ndarray]:
        out = {}
        for mt in ABSORPTION_MTS:
            if mt in self.xs:
                out[mt] = np.interp(energy_mev * 1e6, self.energy_ev, self.xs[mt])
        return out


def default_cross_sections() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config" / "cross_sections" / "diamond_dt.xml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("config/cross_sections/diamond_dt.xml not found; set physics.cross_sections")


def load_nuclide(name: str, cross_sections: str | Path, temperature_k: float = 294.0) -> NuclideData:
    import openmc.data

    library = openmc.data.DataLibrary.from_xml(str(cross_sections))
    entry = library.get_by_material(name, data_type="neutron")
    if entry is None:
        raise KeyError(f"{name} not present in {cross_sections}")
    nuc = openmc.data.IncidentNeutron.from_hdf5(entry["path"])
    temp = min(nuc.temperatures, key=lambda t: abs(float(t.rstrip("K")) - temperature_k))
    energy = nuc.energy[temp]
    xs = {}
    levels = {}
    for mt, rx in nuc.reactions.items():
        if mt in ABSORPTION_MTS or mt == 2:
            xs[mt] = rx.xs[temp](energy)
        if 51 <= mt <= 90:
            levels[mt] = -rx.q_value / 1e6
    return NuclideData(name, nuc.atomic_weight_ratio, levels, energy, xs)


def load_carbon(cross_sections: str | Path, temperature_k: float = 294.0) -> dict[int, NuclideData]:
    """Return data keyed by OpenMC ZA-style nuclide id (1000060120 for C12)."""
    return {
        1000060120: load_nuclide("C12", cross_sections, temperature_k),
        1000060130: load_nuclide("C13", cross_sections, temperature_k),
    }

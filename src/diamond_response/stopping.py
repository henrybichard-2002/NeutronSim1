"""Ion stopping in diamond: catima electronic stopping plus ZBL nuclear stopping."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

N_A = 6.02214076e23
A_C = 12.011
Z_C = 6


def zbl_nuclear_stopping(a1: float, z1: int, energy_mev: np.ndarray, density_g_cm3: float) -> np.ndarray:
    """ZBL universal nuclear stopping power in carbon (MeV/cm)."""
    e_kev = np.asarray(energy_mev, float) * 1e3
    zz = z1**0.23 + Z_C**0.23
    eps = 32.53 * A_C * e_kev / (z1 * Z_C * (a1 + A_C) * zz)
    low = np.log1p(1.1383 * eps) / (2.0 * (eps + 0.01321 * eps**0.21226 + 0.19593 * np.sqrt(eps)))
    high = np.log(np.maximum(eps, 1.0)) / (2.0 * np.maximum(eps, 1.0))
    sn = np.where(eps <= 30.0, low, high)
    ev_per_1e15 = 8.462 * z1 * Z_C * a1 * sn / ((a1 + A_C) * zz)
    atoms_per_cm3 = density_g_cm3 * N_A / A_C
    return ev_per_1e15 * atoms_per_cm3 * 1e-15 * 1e-6


def electronic_stopping(a1: int, z1: int, energy_mev: np.ndarray, density_g_cm3: float) -> np.ndarray:
    import pycatima as catima

    material = catima.Material(A_C, Z_C, density_g_cm3)
    out = np.empty(len(energy_mev))
    for i, e in enumerate(energy_mev):
        proj = catima.Projectile(a1, z1)
        proj.T(e / a1)
        out[i] = catima.dedx(proj, material) * density_g_cm3  # catima gives MeV cm^2/g
    return out


class IonTable:
    """Range and collected-energy tables for one ion species in diamond.

    ``collected(E)`` is the electronic energy loss accumulated while slowing from E to rest.
    Nuclear (lattice) losses are not collected as charge.
    """

    def __init__(self, a1: int, z1: int, density_g_cm3: float, nuclear: bool = True,
                 emax_mev: float = 40.0, npoints: int = 500):
        self.energy = np.geomspace(1e-3, emax_mev, npoints)
        s_el = electronic_stopping(a1, z1, self.energy, density_g_cm3)
        s_n = zbl_nuclear_stopping(a1, z1, self.energy, density_g_cm3) if nuclear else 0.0 * s_el
        s_tot = s_el + s_n
        e = self.energy
        # Below the first point the stopping power scales as sqrt(E).
        r0 = 2.0 * e[0] / s_tot[0]
        inv = 1.0 / s_tot
        self.range_cm = r0 + np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(e))])
        frac = s_el / s_tot
        self.collected = e[0] * frac[0] + np.concatenate(
            [[0.0], np.cumsum(0.5 * (frac[1:] + frac[:-1]) * np.diff(e))]
        )

    def range_of(self, energy):
        return np.interp(energy, self.energy, self.range_cm)

    def energy_at_range(self, remaining_range):
        return np.interp(remaining_range, self.range_cm, self.energy)

    def collected_of(self, energy):
        return np.interp(energy, self.energy, self.collected)


@lru_cache(maxsize=None)
def get_table(a1: int, z1: int, density_g_cm3: float, nuclear: bool) -> IonTable:
    return IonTable(a1, z1, density_g_cm3, nuclear)

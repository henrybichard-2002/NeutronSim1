"""D-T neutron energies and folding of a source spectrum with the response matrix."""

from __future__ import annotations

import numpy as np

from .kinematics import MASS

M_T = 3.01604928 * 931.49410242 - 0.51099895  # triton, from the atomic mass


def dt_neutron_energy(deuteron_mev: float, angle_deg: float) -> float:
    """Neutron kinetic energy (MeV) of d + t -> n + alpha at lab angle to the deuteron beam (triton at rest)."""
    md, mn, ma = MASS["d"], MASS["n"], MASS["a"]
    e_d = deuteron_mev + md
    p_d = np.sqrt(e_d**2 - md**2)
    w = e_d + M_T
    s = md**2 + M_T**2 + 2.0 * M_T * e_d
    a = 0.5 * (s + mn**2 - ma**2)
    pc = p_d * np.cos(np.radians(angle_deg))
    qa = w**2 - pc**2
    roots = np.roots([qa, -2.0 * w * a, a**2 + pc**2 * mn**2])
    roots = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real >= mn]
    ok = [r for r in roots if (w * r - a) * np.cos(np.radians(angle_deg)) >= -1e-9]
    if not ok:
        raise ValueError("no physical neutron solution")
    return float(min(ok) - mn)


def gaussian_fluence_fractions(edges_mev: np.ndarray, mean_mev: float, fwhm_mev: float) -> np.ndarray:
    """Fraction of a Gaussian source spectrum falling in each incident-energy bin."""
    from scipy.special import ndtr

    sigma = fwhm_mev / 2.354820045
    cdf = ndtr((edges_mev - mean_mev) / sigma)
    return np.diff(cdf)


def tabulated_fluence_fractions(edges_mev: np.ndarray, energy_mev, fluence) -> np.ndarray:
    """Bin a tabulated differential fluence (piecewise-linear) onto the incident grid; the result sums to the covered fraction."""
    e = np.asarray(energy_mev, float)
    f = np.asarray(fluence, float)
    fine = np.linspace(edges_mev[0], edges_mev[-1], 20 * (len(edges_mev) - 1) + 1)
    phi = np.interp(fine, e, f, left=0.0, right=0.0)
    total = np.trapezoid(np.interp(np.linspace(e[0], e[-1], 20001), e, f), np.linspace(e[0], e[-1], 20001))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (phi[1:] + phi[:-1]) * np.diff(fine))])
    at_edges = np.interp(edges_mev, fine, cum)
    return np.diff(at_edges) / total


def fold(matrix: np.ndarray, error: np.ndarray, fractions: np.ndarray):
    """Pulse-height spectrum per unit total fluence (pulses per cm^-2 per deposited bin) and its error."""
    return matrix @ fractions, np.sqrt((error**2) @ fractions**2)

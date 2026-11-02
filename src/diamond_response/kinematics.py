"""Relativistic kinematics helpers and nuclear masses (all energies in MeV)."""

from __future__ import annotations

import numpy as np

U_MEV = 931.49410242
# Nuclear (bare) masses in u.
M_U = {
    "n": 1.008664916,
    "p": 1.007276467,
    "d": 2.013553213,
    "a": 4.001506179,
    "9Be": 9.009987,
    "10Be": 10.011338,
    "11B": 11.006560,
    "12B": 12.011608,
    "13B": 13.015034,
    "12C": 11.996706,
    "13C": 13.000061,
}
MASS = {k: v * U_MEV for k, v in M_U.items()}
# 8Be ground state lies 91.84 keV above two alphas.
MASS["8Be"] = 2.0 * MASS["a"] + 0.09184
E_3ALPHA_12C = 7.2747  # 12C excitation energy at the 3-alpha threshold
E_N_9BE = 1.6654  # 9Be neutron separation energy (n + 8Be)

# (mass number, charge) of each tracked ion.
ION_AZ = {"p": (1, 1), "d": (2, 1), "a": (4, 2), "9Be": (9, 4), "10Be": (10, 4),
          "11B": (11, 5), "12B": (12, 5), "13B": (13, 5), "12C": (12, 6), "13C": (13, 6)}


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def isotropic(rng: np.random.Generator, n: int) -> np.ndarray:
    mu = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([s * np.cos(phi), s * np.sin(phi), mu], axis=1)


def boost(p_star: np.ndarray, e_star: np.ndarray, p_sys: np.ndarray, e_sys: np.ndarray, m_sys):
    """Boost a CM-frame 4-momentum to the lab frame, where the system has (e_sys, p_sys)."""
    beta = p_sys / e_sys[:, None]
    b2 = np.sum(beta * beta, axis=1)
    gamma = e_sys / m_sys
    with np.errstate(invalid="ignore", divide="ignore"):
        coef = np.where(b2 > 1e-20, (gamma - 1.0) * np.sum(p_star * beta, axis=1) / b2, 0.0)
    p_lab = p_star + (coef + gamma * e_star)[:, None] * beta
    e_lab = gamma * (e_star + np.sum(beta * p_star, axis=1))
    return p_lab, e_lab


def decay(e_sys: np.ndarray, p_sys: np.ndarray, m1, m2, rng: np.random.Generator):
    """Isotropic two-body decay of a moving system.

    Returns ((p1, e1), (p2, e2)) total energies and lab momenta, plus a boolean mask of
    decays that are kinematically allowed.
    """
    e_sys = np.asarray(e_sys, float)
    m1 = np.broadcast_to(np.asarray(m1, float), e_sys.shape)
    m2 = np.broadcast_to(np.asarray(m2, float), e_sys.shape)
    m_sys = np.sqrt(np.maximum(e_sys**2 - np.sum(p_sys**2, axis=1), 0.0))
    ok = m_sys > m1 + m2
    ms = np.where(ok, m_sys, m1 + m2 + 1.0)
    q = np.sqrt((ms**2 - (m1 + m2) ** 2) * (ms**2 - (m1 - m2) ** 2)) / (2.0 * ms)
    n = isotropic(rng, len(e_sys))
    p1s = q[:, None] * n
    e1s = np.sqrt(q**2 + m1**2)
    e2s = np.sqrt(q**2 + m2**2)
    p1, e1 = boost(p1s, e1s, p_sys, e_sys, ms)
    p2, e2 = boost(-p1s, e2s, p_sys, e_sys, ms)
    return (p1, e1), (p2, e2), ok


def recoil_four_momentum(mass: float, kinetic: np.ndarray, direction: np.ndarray):
    e = mass + kinetic
    p = np.sqrt(np.maximum(e * e - mass * mass, 0.0))
    return e, p[:, None] * direction

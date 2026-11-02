"""Convert OpenMC neutron collision records in diamond into charged-ion starting conditions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import kinematics as kin
from .config import PhysicsConfig
from .nucdata import NuclideData

BEAM_DIRECTION = np.array([0.0, 0.0, 1.0])
NUCLIDE_MASS_KEY = {1000060120: "12C", 1000060130: "13C"}
# (n,p), (n,d), (n,alpha) product pairs on each nuclide: (light ion, residual).
CHANNELS = {
    1000060120: {103: ("p", "12B"), 104: ("d", "11B"), 107: ("a", "9Be")},
    1000060130: {103: ("p", "13B"), 104: ("d", "12B"), 107: ("a", "10Be")},
}


@dataclass
class IonSet:
    """Charged ions to be transported; ``record`` indexes the originating collision record."""

    species: list[str] = field(default_factory=list)
    record: list[np.ndarray] = field(default_factory=list)
    energy: list[np.ndarray] = field(default_factory=list)
    position: list[np.ndarray] = field(default_factory=list)
    direction: list[np.ndarray] = field(default_factory=list)

    def add(self, species: str, record, energy, position, direction) -> None:
        if len(record) == 0:
            return
        self.species.append(species)
        self.record.append(np.asarray(record))
        self.energy.append(np.asarray(energy, float))
        self.position.append(np.asarray(position, float))
        self.direction.append(np.asarray(direction, float))

    def by_species(self):
        merged: dict[str, list[int]] = {}
        for i, name in enumerate(self.species):
            merged.setdefault(name, []).append(i)
        for name, idx in merged.items():
            yield name, tuple(
                np.concatenate([getattr(self, attr)[i] for i in idx])
                for attr in ("record", "energy", "position", "direction")
            )


def _ke(p: np.ndarray, mass: float) -> np.ndarray:
    pm = np.linalg.norm(p, axis=1)
    return pm * pm / (np.sqrt(pm * pm + mass * mass) + mass)


def incoming_directions(rec) -> tuple[np.ndarray, np.ndarray]:
    """Incoming neutron direction for every collision record and a validity mask.

    The first collision of a particle uses the beam direction; later collisions use the
    outgoing direction of the preceding record of the same particle.
    """
    parent = rec["parent_id"].astype(np.int64)
    progeny = rec["progeny_id"].astype(np.int64)
    ncol = rec["n_collision"].astype(np.int64)
    order = np.lexsort((ncol, progeny, parent))
    ps, gs, ns = parent[order], progeny[order], ncol[order]
    if np.any((ps[1:] == ps[:-1]) & (gs[1:] == gs[:-1]) & (ns[1:] == ns[:-1])):
        raise RuntimeError("duplicate (particle id, collision number) in collision track")
    u = np.stack([rec["u"]["x"], rec["u"]["y"], rec["u"]["z"]], axis=1)[order]
    d_sorted = np.tile(BEAM_DIRECTION, (len(order), 1))
    valid_sorted = ns == 1
    follows = np.nonzero((ps[1:] == ps[:-1]) & (gs[1:] == gs[:-1]) & (ns[1:] == ns[:-1] + 1))[0]
    d_sorted[follows + 1] = u[follows]
    valid_sorted[follows + 1] = True
    d_in = np.empty_like(d_sorted)
    valid = np.empty(len(order), bool)
    d_in[order] = d_sorted
    valid[order] = valid_sorted
    return d_in, valid


def generate_ions(rec, nuclides: dict[int, NuclideData], physics: PhysicsConfig,
                  diamond_material_id: int, rng: np.random.Generator):
    """Build charged-ion starting conditions for all neutron collisions in the diamond."""
    d_in, valid = incoming_directions(rec)
    mt = rec["event_mt"].astype(int)
    e_in = rec["E"] / 1e6
    d_e = rec["dE"] / 1e6
    u_out = np.stack([rec["u"]["x"], rec["u"]["y"], rec["u"]["z"]], axis=1)
    pos = np.stack([rec["r"]["x"], rec["r"]["y"], rec["r"]["z"]], axis=1)
    za = rec["nuclide_id"].astype(np.int64)
    in_diamond = rec["material_id"] == diamond_material_id
    mn = kin.MASS["n"]

    p_in = np.sqrt(e_in**2 + 2.0 * e_in * mn)
    e_out = np.maximum(e_in - d_e, 0.0)
    p_out = np.sqrt(e_out**2 + 2.0 * e_out * mn)
    p_rec = p_in[:, None] * d_in - p_out[:, None] * u_out
    rng_iso = kin.isotropic(rng, len(mt))

    ions = IonSet()
    stats: Counter = Counter()
    unhandled: Counter = Counter()
    handled = {2, 101} | set(range(51, 92))

    for za_id, data in nuclides.items():
        sel = in_diamond & (za == za_id)
        key = NUCLIDE_MASS_KEY[za_id]
        mass = kin.MASS[key]
        for kind, m in (("elastic", sel & (mt == 2)), ("inelastic", sel & (mt >= 51) & (mt <= 91))):
            idx = np.nonzero(m)[0]
            if len(idx) == 0:
                continue
            pr = p_rec[idx]
            direction = np.where(valid[idx, None], kin.unit(pr + 1e-30), rng_iso[idx])
            if kind == "elastic":
                e_r = d_e[idx].copy()
                ex = np.zeros(len(idx))
            else:
                e_mom = np.sum(pr * pr, axis=1) / (2.0 * mass)
                ex = np.array([data.level_ex_mev.get(int(t), np.nan) for t in mt[idx]])
                cont = np.isnan(ex)
                ex[cont] = np.maximum(d_e[idx][cont] - e_mom[cont], 0.0)
                e_r = np.maximum(d_e[idx] - ex, 0.0)
            stats[f"{key}:{kind}"] += len(idx)
            breakup = (ex > kin.E_3ALPHA_12C + 0.0918) if key == "12C" else np.zeros(len(idx), bool)
            keep = ~breakup
            ions.add(key, idx[keep], e_r[keep], pos[idx][keep], direction[keep])
            if np.any(breakup):
                b = np.nonzero(breakup)[0]
                _breakup_12c(ions, idx[b], e_r[b], ex[b], direction[b], pos, rng)
                stats["12C:3alpha_breakup"] += len(b)

        idx = np.nonzero(sel & (mt == 101))[0]
        if len(idx):
            _absorption(ions, idx, za_id, data, e_in, d_in, pos, physics, rng, stats)

        for t in np.unique(mt[sel]):
            if int(t) not in handled:
                unhandled[f"{key}:MT{int(t)}"] += int(np.sum(sel & (mt == t)))
    for z in set(np.unique(za[in_diamond])) - set(nuclides):
        unhandled[f"nuclide {int(z)}"] += int(np.sum(in_diamond & (za == z)))
    return ions, stats, unhandled


def _breakup_12c(ions, rec_idx, e_r, ex, direction, pos, rng):
    m_star = kin.MASS["12C"] + ex
    p = np.sqrt(e_r**2 + 2.0 * e_r * kin.MASS["12C"])[:, None] * direction
    e_tot = np.sqrt(np.sum(p * p, axis=1) + m_star**2)
    (p1, _), (p2, e2), ok = kin.decay(e_tot, p, kin.MASS["a"], kin.MASS["8Be"], rng)
    r = pos[rec_idx]
    ions.add("a", rec_idx[ok], _ke(p1[ok], kin.MASS["a"]), r[ok], kin.unit(p1[ok]))
    _alpha_pair(ions, rec_idx[ok], p2[ok], e2[ok], r[ok], rng)


def _alpha_pair(ions, rec_idx, p, e, position, rng):
    """8Be (total energy e, momentum p) decays into two alpha particles."""
    (pa, _), (pb, _), ok = kin.decay(e, p, kin.MASS["a"], kin.MASS["a"], rng)
    for q in (pa, pb):
        ions.add("a", rec_idx[ok], _ke(q[ok], kin.MASS["a"]), position[ok], kin.unit(q[ok]))


def _absorption(ions, idx, za_id, data, e_in, d_in, pos, physics, rng, stats):
    key = NUCLIDE_MASS_KEY[za_id]
    xs = data.absorption_xs(e_in[idx])
    mts = list(xs)
    cum = np.cumsum(np.stack([xs[m] for m in mts], axis=1), axis=1)
    total = cum[:, -1]
    choice = np.argmax(rng.uniform(size=len(idx))[:, None] * total[:, None] < cum, axis=1)
    mn = kin.MASS["n"]
    for c, m in enumerate(mts):
        sub = np.nonzero((choice == c) & (total > 0))[0]
        if len(sub) == 0:
            continue
        stats[f"{key}:absorption_MT{m}"] += len(sub)
        if m == 102:
            continue
        ion_a, ion_b = CHANNELS[za_id][m]
        rec_idx = idx[sub]
        t = e_in[rec_idx]
        p_sys = np.sqrt(t * t + 2.0 * t * mn)[:, None] * d_in[rec_idx]
        e_sys = mn + t + kin.MASS[key]
        r = pos[rec_idx]
        if key == "12C" and m == 107:
            _alpha_channel(ions, rec_idx, e_sys, p_sys, r, physics, rng, stats)
            continue
        (p1, _), (p2, _), ok = kin.decay(e_sys, p_sys, kin.MASS[ion_a], kin.MASS[ion_b], rng)
        stats[f"{key}:absorption_closed"] += int(np.sum(~ok))
        ions.add(ion_a, rec_idx[ok], _ke(p1[ok], kin.MASS[ion_a]), r[ok], kin.unit(p1[ok]))
        ions.add(ion_b, rec_idx[ok], _ke(p2[ok], kin.MASS[ion_b]), r[ok], kin.unit(p2[ok]))


def _alpha_channel(ions, rec_idx, e_sys, p_sys, r, physics, rng, stats):
    """12C(n,alpha)9Be* with sampled 9Be level; unbound levels decay to n + 8Be -> n + 2 alpha."""
    levels = np.array(sorted(physics.alpha_levels))
    w = np.array([physics.alpha_levels[x] for x in levels])
    m_sys = np.sqrt(np.maximum(e_sys**2 - np.sum(p_sys**2, axis=1), 0.0))
    allowed = m_sys[:, None] > (kin.MASS["a"] + kin.MASS["9Be"] + levels)[None, :]
    probs = allowed * w[None, :]
    total = probs.sum(axis=1)
    cum = np.cumsum(probs, axis=1)
    pick = np.argmax(rng.uniform(size=len(total))[:, None] * total[:, None] < cum, axis=1)
    ex = levels[pick]
    stats["12C:absorption_closed"] += int(np.sum(total <= 0))
    for level in np.unique(ex[total > 0]):
        s = np.nonzero((total > 0) & (ex == level))[0]
        (p1, _), (p2, e2), ok = kin.decay(e_sys[s], p_sys[s], kin.MASS["a"], kin.MASS["9Be"] + level, rng)
        ri = rec_idx[s]
        ions.add("a", ri[ok], _ke(p1[ok], kin.MASS["a"]), r[s][ok], kin.unit(p1[ok]))
        if level > kin.E_N_9BE:
            _, (pb, eb), ok2 = kin.decay(e2, p2, kin.MASS["n"], kin.MASS["8Be"], rng)
            good = ok & ok2
            _alpha_pair(ions, ri[good], pb[good], eb[good], r[s][good], rng)
        else:
            ions.add("9Be", ri[ok], _ke(p2[ok], kin.MASS["9Be"]), r[s][ok], kin.unit(p2[ok]))
        stats[f"12C:n_alpha_level_{level:.3f}"] += len(s)

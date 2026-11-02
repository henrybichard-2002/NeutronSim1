"""Ion energy deposition in the charge-collecting volume of the diamond."""

from __future__ import annotations

import numpy as np

from . import kinematics as kin
from .config import DetectorConfig
from .events import IonSet
from .stopping import get_table


def local_frame(theta_deg: float):
    """Detector axes (ex, ey, normal) in lab coordinates for a rotation about y."""
    th = np.radians(theta_deg)
    ex = np.array([np.cos(th), 0.0, -np.sin(th)])
    ey = np.array([0.0, 1.0, 0.0])
    n = np.array([np.sin(th), 0.0, np.cos(th)])
    return ex, ey, n


def active_box_cm(det: DetectorConfig):
    """Lower and upper corners (cm) of the charge-collecting box in detector-local coordinates."""
    hx = det.width_mm[0] / 20.0 - det.dead_layer_lateral_um * 1e-4
    hy = det.width_mm[1] / 20.0 - det.dead_layer_lateral_um * 1e-4
    ht = det.thickness_um * 1e-4 / 2.0
    lo = np.array([-hx, -hy, -ht + det.dead_layer_front_um * 1e-4])
    hi = np.array([hx, hy, ht - det.dead_layer_back_um * 1e-4])
    return lo, hi


def segment_in_box(r: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Entry and exit path lengths (t >= 0) of rays r + t d with an axis-aligned box."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - r) / d
        t2 = (hi - r) / d
    parallel = np.abs(d) < 1e-15
    inside = (r >= lo) & (r <= hi)
    tmin = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    tmax = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    t_in = np.maximum(tmin.max(axis=1), 0.0)
    t_out = tmax.min(axis=1)
    return t_in, t_out


def ion_deposits(ions: IonSet, det: DetectorConfig, nuclear_stopping: bool):
    """Collected energy (MeV) of each ion along its straight track; returns (record, deposit)."""
    ex, ey, nrm = local_frame(det.incidence_angle_deg)
    rot = np.stack([ex, ey, nrm])
    lo, hi = active_box_cm(det)
    records, deposits = [], []
    for name, (record, energy, position, direction) in ions.by_species():
        a, z = kin.ION_AZ[name]
        table = get_table(a, z, det.density_g_cm3, nuclear_stopping)
        t_in, t_out = segment_in_box(position @ rot.T, direction @ rot.T, lo, hi)
        r0 = table.range_of(energy)
        t_out = np.minimum(t_out, r0)
        crossing = t_out > t_in
        e_enter = table.energy_at_range(np.maximum(r0 - t_in, 0.0))
        e_exit = table.energy_at_range(np.maximum(r0 - t_out, 0.0))
        dep = np.where(crossing, table.collected_of(e_enter) - table.collected_of(e_exit), 0.0)
        records.append(record)
        deposits.append(dep)
    if not records:
        return np.empty(0, int), np.empty(0)
    return np.concatenate(records), np.concatenate(deposits)


def pulses(record: np.ndarray, deposit: np.ndarray, rec, window_ns: float) -> np.ndarray:
    """Sum deposits of each particle history that fall within ``window_ns`` of its first deposit."""
    keep = deposit > 0
    if not np.any(keep):
        return np.empty(0)
    record, deposit = record[keep], deposit[keep]
    parent = rec["parent_id"][record].astype(np.int64)
    progeny = rec["progeny_id"][record].astype(np.int64)
    time_ns = rec["time"][record] * 1e9
    order = np.lexsort((time_ns, progeny, parent))
    parent, progeny, time_ns, deposit = parent[order], progeny[order], time_ns[order], deposit[order]
    first = np.ones(len(order), bool)
    first[1:] = (parent[1:] != parent[:-1]) | (progeny[1:] != progeny[:-1])
    group = np.cumsum(first) - 1
    start_time = time_ns[np.nonzero(first)[0]][group]
    new_pulse = first | (time_ns - start_time > window_ns)
    return np.bincount(np.cumsum(new_pulse) - 1, weights=deposit)

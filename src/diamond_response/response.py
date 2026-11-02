"""Run OpenMC chunks, turn collisions into pulse-height histograms and assemble the response matrix."""

from __future__ import annotations

import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SimulationConfig, deposited_edges_mev, incident_edges_mev
from .events import generate_ions
from .model import DIAMOND_MATERIAL_ID, beam_area_cm2, build_model, collision_limit, cross_section_path
from .nucdata import load_carbon
from .transport import ion_deposits, pulses


def openmc_executable() -> str:
    candidate = Path(sys.executable).parent / "openmc"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("openmc")
    if found is None:
        raise FileNotFoundError("openmc executable not found next to the Python interpreter or on PATH")
    return found


@dataclass
class ColumnAccumulator:
    """Per-incident-energy-bin running sums of per-history pulse-height histograms."""

    n_bins: int
    n_chunks: int = 0
    n_histories: float = 0.0
    sum_hist: np.ndarray | None = None
    sum_sq: np.ndarray | None = None
    sum_roi: float = 0.0
    sum_roi_sq: float = 0.0
    stats: Counter | None = None
    unhandled: Counter | None = None

    def __post_init__(self):
        if self.sum_hist is None:
            self.sum_hist = np.zeros(self.n_bins)
            self.sum_sq = np.zeros(self.n_bins)
        self.stats = self.stats or Counter()
        self.unhandled = self.unhandled or Counter()

    def add(self, counts, roi_count, histories, stats, unhandled):
        per_history = counts / histories
        self.sum_hist += per_history
        self.sum_sq += per_history**2
        roi = roi_count / histories
        self.sum_roi += roi
        self.sum_roi_sq += roi * roi
        self.n_chunks += 1
        self.n_histories += histories
        self.stats.update(stats)
        self.unhandled.update(unhandled)

    @staticmethod
    def _sem(total, total_sq, n):
        if n < 2:
            return np.full_like(np.asarray(total, float), np.inf)
        mean = total / n
        var = np.maximum(total_sq / n - mean**2, 0.0) * n / (n - 1)
        return np.sqrt(var / n)

    @property
    def mean(self):
        return self.sum_hist / max(self.n_chunks, 1)

    @property
    def error(self):
        return self._sem(self.sum_hist, self.sum_sq, self.n_chunks)

    @property
    def roi_rel_error(self) -> float:
        if self.n_chunks < 2 or self.sum_roi <= 0:
            return np.inf
        return float(self._sem(self.sum_roi, self.sum_roi_sq, self.n_chunks) / (self.sum_roi / self.n_chunks))

    def save(self, path: Path) -> None:
        np.savez(
            path, n_bins=self.n_bins, n_chunks=self.n_chunks, n_histories=self.n_histories,
            sum_hist=self.sum_hist, sum_sq=self.sum_sq, sum_roi=self.sum_roi, sum_roi_sq=self.sum_roi_sq,
            stats=repr(dict(self.stats)), unhandled=repr(dict(self.unhandled)),
        )

    @classmethod
    def load(cls, path: Path) -> "ColumnAccumulator":
        import ast

        d = np.load(path, allow_pickle=False)
        return cls(
            int(d["n_bins"]), int(d["n_chunks"]), float(d["n_histories"]), d["sum_hist"], d["sum_sq"],
            float(d["sum_roi"]), float(d["sum_roi_sq"]),
            Counter(ast.literal_eval(str(d["stats"]))), Counter(ast.literal_eval(str(d["unhandled"]))),
        )


class Simulator:
    def __init__(self, config: SimulationConfig, threads: int | None = None, workdir: Path | None = None):
        self.config = config
        self.threads = threads
        self.workdir = workdir
        self.inc_edges = incident_edges_mev(config)
        self.dep_edges = deposited_edges_mev(config)
        self.xs_path = cross_section_path(config)
        self.nuclides = load_carbon(self.xs_path, config.physics.temperature_k)
        self.exe = openmc_executable()

    def run_chunk(self, column: int, chunk: int):
        """One OpenMC run for one incident-energy bin; returns (histogram, roi_count, stats, unhandled)."""
        import h5py

        cfg = self.config
        histories = int(cfg.statistics.histories_per_chunk)
        lo, hi = self.inc_edges[column], self.inc_edges[column + 1]
        seed = 1 + cfg.seed * 1_000_003 + column * 997 + chunk
        model = build_model(cfg, lo, hi, histories, seed)
        model.materials.cross_sections = str(self.xs_path)
        tmp = Path(tempfile.mkdtemp(prefix=f"diamond_{column}_{chunk}_", dir=self.workdir))
        try:
            model.run(cwd=tmp, openmc_exec=self.exe, threads=self.threads, output=False)
            with h5py.File(tmp / "collision_track.h5", "r") as handle:
                rec = handle["collision_track_bank"][:]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if len(rec) >= collision_limit(cfg, histories):
            raise RuntimeError("collision bank is full and was truncated; reduce statistics.histories_per_chunk")
        rng = np.random.default_rng(seed)
        if len(rec) == 0:
            return np.zeros(len(self.dep_edges) - 1), 0.0, Counter(), Counter(), histories
        ions, stats, unhandled = generate_ions(rec, self.nuclides, cfg.physics, DIAMOND_MATERIAL_ID, rng)
        record, deposit = ion_deposits(ions, cfg.detector, cfg.physics.nuclear_stopping)
        heights = pulses(record, deposit, rec, cfg.physics.pulse_window_ns)
        counts, _ = np.histogram(heights, bins=self.dep_edges)
        roi = float(np.sum(heights > cfg.statistics.roi_min_mev))
        return counts.astype(float), roi, stats, unhandled, histories

    def column_path(self, outdir: Path, column: int) -> Path:
        return column_file(outdir, column)

    def run_column(self, outdir: Path, column: int, log=print, target_rel_error: float | None = None) -> ColumnAccumulator:
        cfg = self.config.statistics
        target = cfg.target_rel_error if target_rel_error is None else target_rel_error
        path = self.column_path(outdir, column)
        path.parent.mkdir(parents=True, exist_ok=True)
        acc = ColumnAccumulator.load(path) if path.exists() else ColumnAccumulator(len(self.dep_edges) - 1)
        while acc.n_chunks < cfg.max_chunks and (acc.n_chunks < cfg.min_chunks or acc.roi_rel_error > target):
            counts, roi, stats, unhandled, histories = self.run_chunk(column, acc.n_chunks)
            acc.add(counts, roi, histories, stats, unhandled)
            acc.save(path)
            log(f"  bin {column} [{self.inc_edges[column]:.3f}-{self.inc_edges[column + 1]:.3f} MeV] "
                f"chunk {acc.n_chunks}: roi rel. error {acc.roi_rel_error:.4f}")
        return acc

    def run_all(self, outdir: Path, columns=None, log=print) -> "ResponseMatrix":
        outdir.mkdir(parents=True, exist_ok=True)
        todo = range(len(self.inc_edges) - 1) if columns is None else columns
        for column in todo:
            self.run_column(outdir, column, log)
        return assemble(self.config, outdir)


def npz_dir(outdir: Path) -> Path:
    """Binary (.npz) data lives in ``<run>/npz`` so the run directory keeps only CSVs and plots."""
    return Path(outdir) / "npz"


def column_file(outdir: Path, column: int) -> Path:
    name = f"column_{column:04d}.npz"
    legacy = Path(outdir) / name
    return legacy if legacy.exists() else npz_dir(outdir) / name


@dataclass
class ResponseMatrix:
    """``matrix[i, j]``: pulses in deposited-energy bin i per unit fluence of neutrons in incident bin j (cm^2)."""

    inc_edges: np.ndarray
    dep_edges: np.ndarray
    matrix: np.ndarray
    error: np.ndarray
    histories: np.ndarray
    beam_area_cm2: float
    stats: dict

    @property
    def inc_centers(self):
        return 0.5 * (self.inc_edges[1:] + self.inc_edges[:-1])

    @property
    def dep_centers(self):
        return 0.5 * (self.dep_edges[1:] + self.dep_edges[:-1])

    def efficiency(self, threshold_mev: float = 0.0):
        """Intrinsic pulse rate above threshold per unit fluence (cm^2) for each incident bin."""
        return self.matrix[self.dep_centers > threshold_mev].sum(axis=0)

    def save(self, path: Path) -> None:
        np.savez(path, inc_edges=self.inc_edges, dep_edges=self.dep_edges, matrix=self.matrix,
                 error=self.error, histories=self.histories, beam_area_cm2=self.beam_area_cm2,
                 stats=repr(self.stats))

    @classmethod
    def load(cls, path: Path) -> "ResponseMatrix":
        import ast

        d = np.load(path, allow_pickle=False)
        return cls(d["inc_edges"], d["dep_edges"], d["matrix"], d["error"], d["histories"],
                   float(d["beam_area_cm2"]), ast.literal_eval(str(d["stats"])))

    def with_resolution(self, noise_kev: float, relative: float) -> "ResponseMatrix":
        g = resolution_kernel(self.dep_edges, noise_kev, relative)
        return ResponseMatrix(self.inc_edges, self.dep_edges, g @ self.matrix,
                              np.sqrt((g**2) @ self.error**2), self.histories, self.beam_area_cm2, self.stats)

    def to_csv(self, path: Path) -> None:
        header = "deposited_MeV," + ",".join(f"{c:.4f}" for c in self.inc_centers)
        np.savetxt(path, np.column_stack([self.dep_centers, self.matrix]), delimiter=",", header=header, comments="")


def resolution_kernel(edges_mev: np.ndarray, noise_kev: float, relative: float) -> np.ndarray:
    """Matrix ``G[k, i]``: probability that a pulse from true bin i is measured in bin k."""
    from scipy.special import ndtr

    centers = 0.5 * (edges_mev[1:] + edges_mev[:-1])
    fwhm = np.sqrt((noise_kev * 1e-3) ** 2 + (relative * centers) ** 2)
    sigma = np.maximum(fwhm / 2.354820045, 1e-9)
    upper = ndtr((edges_mev[1:, None] - centers[None, :]) / sigma[None, :])
    lower = ndtr((edges_mev[:-1, None] - centers[None, :]) / sigma[None, :])
    return upper - lower


def assemble(config: SimulationConfig, outdir: Path) -> ResponseMatrix:
    inc = incident_edges_mev(config)
    dep = deposited_edges_mev(config)
    area = beam_area_cm2(config)
    matrix = np.zeros((len(dep) - 1, len(inc) - 1))
    error = np.zeros_like(matrix)
    histories = np.zeros(len(inc) - 1)
    stats: dict = {}
    for j in range(len(inc) - 1):
        path = column_file(outdir, j)
        if not path.exists():
            continue
        acc = ColumnAccumulator.load(path)
        if acc.n_chunks == 0:
            continue
        matrix[:, j] = acc.mean * area
        error[:, j] = np.where(np.isfinite(acc.error), acc.error, 0.0) * area
        histories[j] = acc.n_histories
        stats[f"{inc[j]:.3f}-{inc[j + 1]:.3f}"] = {
            "stats": dict(acc.stats), "unhandled": dict(acc.unhandled), "roi_rel_error": acc.roi_rel_error,
        }
    return ResponseMatrix(inc, dep, matrix, error, histories, area, stats)

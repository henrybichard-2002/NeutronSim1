"""Run an OpenMC model while printing a short three-line report per batch."""

from __future__ import annotations

import threading
import time
from pathlib import Path

if not __package__:
    raise SystemExit(
        "batch_report is a library module; run simulations with "
        "'python -m iec_neutrons.cli run CONFIG' (batch reports are on by default)."
    )

from .config import SimulationConfig  # noqa: E402

_POLL_SECONDS = 0.25
_SETTLE_SECONDS = 0.3


def _duration(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:04.1f}s"
    if minutes:
        return f"{minutes}m{secs:04.1f}s"
    return f"{secs:.1f}s"


class _Snapshot:
    """Cumulative (active-batch) quantities read from one statepoint."""

    def __init__(self, path: Path, openmc, np) -> None:
        with openmc.StatePoint(str(path), autolink=False) as sp:
            names = [
                name.decode() if isinstance(name, bytes) else str(name)
                for name in sp.global_tallies["name"]
            ]
            self.leakage = float(
                sp.global_tallies["sum"][names.index("leakage")]
            )
            self.reactions = {"total": 0.0, "scatter": 0.0, "absorption": 0.0,
                              "multiply": 0.0, "capture": 0.0}
            self.flux = None
            self.realizations = 0
            for tally in sp.tallies.values():
                name = tally.name or ""
                if name.startswith("neutron_interactions_"):
                    self.realizations = max(self.realizations, tally.num_realizations)
                    totals = np.asarray(tally.sum).sum(axis=(0, 1))
                    score = dict(zip(tally.scores, totals))
                    self.reactions["total"] += score["total"]
                    self.reactions["scatter"] += score["scatter"]
                    self.reactions["absorption"] += score["absorption"]
                    self.reactions["multiply"] += score["(n,2n)"] + 2 * score["(n,3n)"]
                    self.reactions["capture"] += score["(n,gamma)"]
                elif name == "flux_spectrum_source":
                    energy_filter = next(
                        f for f in tally.filters if type(f).__name__ == "EnergyFilter"
                    )
                    bins = np.asarray(energy_filter.bins, dtype=float)
                    self.flux = (
                        np.asarray(tally.sum).reshape(-1).copy(),
                        bins.mean(axis=1) / 1.0e6,
                    )


def _report(index, total, particles, runtime, elapsed, now, before, np) -> str:
    header = (
        f"[batch {index}/{total}] {_duration(runtime)} "
        f"({particles / max(runtime, 1e-9):,.0f} histories/s, "
        f"total {_duration(elapsed)}) | {particles:,} source neutrons"
    )
    if now.realizations == before.realizations:
        note = "inactive batch: tallies are not scored"
        return f"{header}\n  {note}\n  -"
    delta = {key: now.reactions[key] - before.reactions[key] for key in now.reactions}
    leakage = now.leakage - before.leakage
    line2 = (
        f"  leakage {leakage:.2%} | interactions per source neutron: "
        f"{delta['total']:.3g} (scatter {delta['scatter']:.3g}, "
        f"absorption {delta['absorption']:.3g})"
    )
    energy = ""
    if now.flux is not None and before.flux is not None:
        diff = now.flux[0] - before.flux[0]
        if diff.sum() > 0:
            energy = f" | mean source-region flux energy {np.dot(diff, now.flux[1]) / diff.sum():.3g} MeV"
    line3 = (
        f"  (n,2n)+(n,3n) multiplication {delta['multiply']:.3g}, "
        f"(n,gamma) captures {delta['capture']:.3g} per source neutron{energy}"
    )
    return f"{header}\n{line2}\n{line3}"


def run_with_batch_reports(model, destination: Path, config: SimulationConfig) -> Path:
    """Run ``model`` in ``destination`` and print a report after each batch."""
    import numpy as np
    import openmc

    batches = config.settings.batches
    particles = config.settings.particles
    for stale in destination.glob("statepoint.*.h5"):
        stale.unlink()
    model.settings.statepoint = {"batches": list(range(1, batches + 1))}

    outcome: dict = {}

    def run() -> None:
        try:
            outcome["statepoint"] = model.run(cwd=str(destination))
        except BaseException as exc:  # re-raised in the calling thread
            outcome["error"] = exc

    start = time.time()
    worker = threading.Thread(target=run, daemon=True)
    worker.start()

    previous = None
    previous_time = start
    next_batch = 1
    while next_batch <= batches:
        finished = not worker.is_alive()
        path = destination / f"statepoint.{next_batch:0{len(str(batches))}d}.h5"
        ready = path.is_file() and (
            finished or time.time() - path.stat().st_mtime > _SETTLE_SECONDS
        )
        if not ready:
            if finished:
                break
            time.sleep(_POLL_SECONDS)
            continue
        try:
            snapshot = _Snapshot(path, openmc, np)
        except (OSError, KeyError):
            if finished:
                break
            time.sleep(_POLL_SECONDS)
            continue
        finished_at = path.stat().st_mtime
        if previous is None:
            previous = _Snapshot.__new__(_Snapshot)
            previous.leakage = 0.0
            previous.reactions = dict.fromkeys(snapshot.reactions, 0.0)
            previous.flux = (
                None
                if snapshot.flux is None
                else (np.zeros_like(snapshot.flux[0]), snapshot.flux[1])
            )
            previous.realizations = 0
        print(
            _report(
                next_batch, batches, particles, finished_at - previous_time,
                finished_at - start, snapshot, previous, np,
            ),
            flush=True,
        )
        previous, previous_time = snapshot, finished_at
        if next_batch < batches:
            path.unlink()
        next_batch += 1

    worker.join()
    if "error" in outcome:
        raise outcome["error"]
    return Path(outcome["statepoint"])


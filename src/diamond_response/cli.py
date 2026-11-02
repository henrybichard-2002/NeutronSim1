"""Command-line interface for the diamond detector response simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import ConfigError, incident_edges_mev, load_config
from .response import ResponseMatrix, Simulator, assemble
from .spectra import dt_neutron_energy, fold, gaussian_fluence_fractions


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="diamond-response", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="simulate the response matrix (resumable)")
    r.add_argument("config", type=Path)
    r.add_argument("--out", type=Path, default=None, help="output directory (default runs/<config name>)")
    r.add_argument("--threads", type=int, default=None)
    r.add_argument("--columns", type=int, nargs="*", default=None, help="only these incident-energy columns")
    x = sub.add_parser("reactor", help="diamond response in the IEC reactor field (CSV with uncertainties)")
    x.add_argument("config", type=Path)
    x.add_argument("--out", type=Path, default=None)
    x.add_argument("--threads", type=int, default=None)
    x.add_argument("--rerun-field", action="store_true", help="redo the reactor transport even if saved")
    u = sub.add_parser("unfold", help="unfold the reactor pulse-height spectrum and compare with the true spectrum")
    u.add_argument("config", type=Path)
    u.add_argument("--out", type=Path, required=True, help="existing reactor run directory")
    u.add_argument("--threads", type=int, default=None)
    c = sub.add_parser("compare", help="compare measured pulse-height spectra with simulation, unfold both")
    c.add_argument("config", type=Path, help="reactor config at the measurement position")
    c.add_argument("--matrix", type=Path, required=True, help="response_matrix.npz used for the folding/unfolding")
    c.add_argument("--data", type=str, nargs="+", required=True, help="folders with all_datasets.npz + metadata.json")
    c.add_argument("--out", type=Path, required=True, help="run directory (holds/receives npz/reactor_field.npz)")
    c.add_argument("--threads", type=int, default=None)
    c.add_argument("--noise-kev", type=float, default=None, help="override resolution.noise_kev")
    c.add_argument("--relative-resolution", type=float, default=None, help="override resolution.relative")
    c.add_argument("--iterations", type=int, default=None, help="override unfold.max_iterations (fewer = smoother)")
    c.add_argument("--min-pulse-mev", type=float, default=None, help="override unfold.min_pulse_mev")
    c.add_argument("--replicas", type=int, default=None, help="override unfold.replicas")
    a = sub.add_parser("assemble", help="rebuild npz/response_matrix.npz, csv/plots from finished columns")
    a.add_argument("config", type=Path)
    a.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("fold", help="pulse-height spectrum for a Gaussian D-T source")
    f.add_argument("matrix", type=Path)
    f.add_argument("--mean", type=float, default=None, help="neutron mean energy [MeV]")
    f.add_argument("--deuteron-mev", type=float, default=None, help="deuteron energy for --angle-deg")
    f.add_argument("--angle-deg", type=float, default=0.0)
    f.add_argument("--fwhm-mev", type=float, default=0.2)
    f.add_argument("--output", type=Path, default=None)
    return p


def _finish(config, rm: ResponseMatrix, out: Path) -> None:
    from .plot import plot_matrix

    (out / "npz").mkdir(parents=True, exist_ok=True)
    rm.save(out / "npz" / "response_matrix.npz")
    rm.to_csv(out / "response_matrix.csv")
    res = config.resolution
    smeared = rm.with_resolution(res.noise_kev, res.relative)
    smeared.to_csv(out / "response_matrix_resolution.csv")
    try:
        from .plot import plot_all

        for path in plot_matrix(rm, out, smeared) + plot_all(rm, out, smeared):
            print("wrote", path)
    except ImportError:
        print("matplotlib not installed; skipping plots")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            config = load_config(args.config)
            out = args.out or Path("runs") / args.config.stem
            sim = Simulator(config, threads=args.threads)
            rm = sim.run_all(out, columns=args.columns)
            _finish(config, rm, out)
        elif args.command == "reactor":
            from . import reactor

            config = load_config(args.config)
            if config.reactor is None:
                raise ConfigError("config has no 'reactor:' section")
            out = args.out or Path("runs") / args.config.stem
            for path in reactor.run_all(config, out, args.threads, args.rerun_field):
                print("wrote", path)
        elif args.command == "unfold":
            from . import reactor
            from .unfold import run_unfolding

            config = load_config(args.config)
            if config.reactor is None:
                raise ConfigError("config has no 'reactor:' section")
            field_path = reactor.npz_dir(args.out) / reactor.NEUTRON_FIELD_FILE
            if not field_path.exists():
                raise FileNotFoundError(f"no saved reactor field at {field_path}; run 'reactor' first")
            field = reactor.ReactorField.load(field_path)
            matrices = reactor.run_class_matrices(config, field, args.out, args.threads, print)
            for path in run_unfolding(config, field, matrices, args.out):
                print("wrote", path)
        elif args.command == "compare":
            from . import reactor
            from .compare import load_measured, resolve_path, run_comparison

            config = load_config(args.config)
            if config.reactor is None:
                raise ConfigError("config has no 'reactor:' section")
            field_path = reactor.npz_dir(args.out) / reactor.NEUTRON_FIELD_FILE
            if field_path.exists() and reactor.ReactorField.load(field_path).matches(config):
                field = reactor.ReactorField.load(field_path)
            else:
                field = reactor.run_field(config, args.out, args.threads)
                field.save(field_path)
            for key, value in (("max_iterations", args.iterations), ("min_pulse_mev", args.min_pulse_mev),
                               ("replicas", args.replicas)):
                if value is not None:
                    setattr(config.unfold, key, value)
            matrix = ResponseMatrix.load(resolve_path(args.matrix))
            measured = [load_measured(d, matrix.dep_edges) for d in args.data]
            for path in run_comparison(config, field, matrix, measured, args.out, args.noise_kev,
                                       args.relative_resolution):
                print("wrote", path)
        elif args.command == "assemble":
            config = load_config(args.config)
            _finish(config, assemble(config, args.out), args.out)
        else:
            rm = ResponseMatrix.load(args.matrix)
            mean = args.mean
            if mean is None:
                if args.deuteron_mev is None:
                    raise ConfigError("give --mean or --deuteron-mev")
                mean = dt_neutron_energy(args.deuteron_mev, args.angle_deg)
            frac = gaussian_fluence_fractions(rm.inc_edges, mean, args.fwhm_mev)
            spec, err = fold(rm.matrix, rm.error, frac)
            print(f"mean {mean:.4f} MeV, fraction inside incident grid {frac.sum():.4f}")
            print(f"pulses per unit fluence: total {spec.sum():.4e} cm^2, above 0.5 MeV {spec[rm.dep_centers > 0.5].sum():.4e} cm^2")
            if args.output:
                np.savetxt(args.output, np.column_stack([rm.dep_centers, spec, err]), delimiter=",",
                           header="deposited_MeV,pulses_per_fluence_cm2,error", comments="")
    except (ConfigError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

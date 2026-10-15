"""Command-line interface for building and running IEC source models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import ConfigError, load_config
from .model import export_model, run_model
from .visualize import plot_results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or run a configurable IEC D-T neutron model.")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "run"):
        subparser = commands.add_parser(command, help=f"{command.capitalize()} the OpenMC model")
        subparser.add_argument("config", type=Path, help="YAML configuration file")
        subparser.add_argument(
            "--output-dir",
            type=Path,
            help="directory for OpenMC XML files and run output (default: ./runs/<config-name>)",
        )
    plot_parser = commands.add_parser("plot", help="Plot flux and energy results from an OpenMC statepoint")
    plot_parser.add_argument("config", type=Path, help="YAML configuration file used for the run")
    plot_parser.add_argument("statepoint", type=Path, help="OpenMC statepoint HDF5 file")
    plot_parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for plots (default: <statepoint-directory>/plots)",
    )
    return parser


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    try:
        config = load_config(arguments.config)
    except (ConfigError, OSError) as exc:
        parser.error(str(exc))

    output_dir = arguments.output_dir or Path("runs") / arguments.config.stem
    try:
        if arguments.command == "build":
            destination = export_model(config, output_dir)
            print(f"OpenMC model inputs written to {destination}")
        elif arguments.command == "run":
            statepoint = run_model(config, output_dir)
            print(f"OpenMC run complete; statepoint written to {statepoint}")
        else:
            plot_dir = arguments.output_dir or arguments.statepoint.parent / "plots"
            plot_paths = plot_results(config, arguments.statepoint, plot_dir)
            print(f"Plots written to {plot_dir.resolve()}:")
            for plot_path in plot_paths:
                print(f"  {plot_path.name}")
    except (RuntimeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

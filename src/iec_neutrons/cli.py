"""Command-line interface for building and running IEC source models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import ConfigError, load_config
from .compiled_source import run_custom_source_model
from .diagnostics import export_he3_spectra_csv, print_run_summary
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
            if config.custom_source is not None:
                statepoint, source_spectrum = run_custom_source_model(
                    config, output_dir
                )
                print(
                    "Custom-source birth-energy spectrum written to "
                    f"{source_spectrum}"
                )
            else:
                statepoint = run_model(config, output_dir)
            print(f"OpenMC run complete; statepoint written to {statepoint}")
            print_run_summary(statepoint, config)
            csv_path = export_he3_spectra_csv(
                statepoint, config, Path(statepoint).parent / "he3_detector_spectra.csv"
            )
            if csv_path is not None:
                print(f"He-3 detector spectra written to {csv_path}")
        else:
            plot_dir = arguments.output_dir or arguments.statepoint.parent / "plots"
            csv_path = export_he3_spectra_csv(
                arguments.statepoint, config, plot_dir / "he3_detector_spectra.csv"
            )
            plot_paths = plot_results(config, arguments.statepoint, plot_dir)
            print(f"Plots written to {plot_dir.resolve()}:")
            for plot_path in plot_paths:
                print(f"  {plot_path.name}")
            if csv_path is not None:
                print(f"He-3 detector spectra written to {csv_path}")
    except (RuntimeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

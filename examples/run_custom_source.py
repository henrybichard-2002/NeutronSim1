#!/usr/bin/env python3
"""Run the configured OpenMC model with the compiled IEC fusion source."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import uuid

try:
    import openmc
except ImportError as exc:
    raise SystemExit(
        f"Could not import OpenMC with {sys.executable}: {exc}. Run this script "
        "with the Python from the neutrons-openmc environment."
    ) from exc

from iec_neutrons.batch_report import run_with_batch_reports
from iec_neutrons.compiled_source import (
    collect_source_birth_spectrum,
    compile_custom_source,
    custom_source_parameters,
)
from iec_neutrons.config import load_config
from iec_neutrons.diagnostics import export_he3_spectra_csv, print_run_summary
from iec_neutrons.model import build_model
from iec_neutrons.visualize import plot_results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an OpenMC model using the compiled IEC D-T/D-D source."
    )
    parser.add_argument("config", type=Path, help="OpenMC model YAML configuration")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "directory for the shared library, XML inputs, statepoint, and plots "
            "(default: runs/<config-name>-custom-source)"
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="plot the results after a successful simulation (requires Matplotlib)",
    )
    parser.add_argument(
        "--parameters",
        help=(
            "semicolon-separated C++ source parameters (overrides any "
            "custom_source section in the YAML)"
        ),
    )
    parser.add_argument(
        "--source-cpp",
        type=Path,
        help="custom_source.cpp path (defaults to the installed package source)",
    )
    parser.add_argument(
        "--compiler",
        help="C++ compiler executable (defaults to c++ on PATH)",
    )
    args = parser.parse_args()
    if args.plot and importlib.util.find_spec("matplotlib") is None:
        parser.error(
            "Matplotlib is required for --plot. Install this project into "
            "the same OpenMC environment with: python -m pip install -e "
            '"[plot]"'
        )

    output_dir = (
        args.output_dir or Path("runs") / f"{args.config.stem}-custom-source"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    library = compile_custom_source(
        source_file=args.source_cpp,
        output_library=output_dir / "libiec_custom_source.so",
        compiler=args.compiler,
    )

    config = load_config(args.config)
    model = build_model(config)
    source_spectrum_base = (
        output_dir / f"source_birth_spectrum.{uuid.uuid4().hex}"
        if config.custom_source is not None
        else None
    )
    parameters = custom_source_parameters(
        config,
        source_spectrum_base,
        override_parameters=args.parameters,
    )
    model.settings.source = openmc.CompiledSource(
        library=library,
        parameters=parameters,
    )
    statepoint = run_with_batch_reports(model, output_dir, config)
    if source_spectrum_base is not None:
        source_spectrum = collect_source_birth_spectrum(
            source_spectrum_base, output_dir / "source_birth_spectrum.csv"
        )
        print(
            "Custom-source birth-energy spectrum written to "
            f"{source_spectrum}"
        )
    print(f"OpenMC run complete; statepoint written to {statepoint}")
    print_run_summary(statepoint, config)
    csv_path = export_he3_spectra_csv(
        statepoint, config, output_dir / "he3_detector_spectra.csv"
    )
    if csv_path is not None:
        print(f"He-3 detector spectra written to {csv_path}")
    if args.plot:
        plot_paths = plot_results(
            config, statepoint, output_dir / "plots"
        )
        print(f"Plots written to {output_dir / 'plots'}:")
        for plot_path in plot_paths:
            print(f"  {plot_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the configured OpenMC model with the compiled IEC fusion source."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

try:
    import openmc
except ImportError as exc:
    raise SystemExit(
        f"Could not import OpenMC with {sys.executable}: {exc}. Run this script "
        "with the Python from the neutrons-openmc environment."
    ) from exc

from iec_neutrons.compiled_source import compile_custom_source
from iec_neutrons.config import load_config
from iec_neutrons.diagnostics import export_he3_spectra_csv, print_run_summary
from iec_neutrons.model import build_model
from iec_neutrons.visualize import plot_results


DEFAULT_PARAMETERS = (
    "r_anode_cm=3.85;"
    "r_cathode_cm=0.73;"
    "half_height_cm=17.5;"
    "voltage_kev=90;"
    "neutral_temperature_k=300;"
    "tritium_fraction=0.5;"
    "zone_beam_target=1;"
    "zone_beam_wall=0;"
    "zone_beam_beam=0;"
    "source_strength=1;"
    "neutral_profile=0:1,1:1"
)


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
    parameters = args.parameters or DEFAULT_PARAMETERS
    if args.parameters is None and config.custom_source is not None:
        custom = config.custom_source
        profile = ",".join(
            f"{radius_fraction:g}:{density:g}"
            for radius_fraction, density in custom.neutral_profile
        )
        parameters = ";".join(
            (
                f"r_anode_cm={config.geometry.cavity_radius_cm:g}",
                f"r_cathode_cm={custom.cathode_radius_cm:g}",
                f"half_height_cm={config.source.height_cm / 2:g}",
                f"voltage_kev={custom.voltage_kev:g}",
                f"neutral_temperature_k={custom.neutral_temperature_k:g}",
                f"tritium_fraction={custom.tritium_fraction:g}",
                f"zone_beam_target={custom.zone_beam_target:g}",
                f"zone_beam_wall={custom.zone_beam_wall:g}",
                f"zone_beam_beam={custom.zone_beam_beam:g}",
                f"source_strength={custom.source_strength:g}",
                f"neutral_profile={profile}",
            )
        )
    model.settings.source = openmc.CompiledSource(
        library=library,
        parameters=parameters,
    )
    statepoint = model.run(cwd=str(output_dir))
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

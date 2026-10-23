"""Compile the IEC fusion source plugin against the active OpenMC installation."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import csv
import uuid

from .config import SimulationConfig


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


def custom_source_parameters(
    config: SimulationConfig,
    source_spectrum_base: str | Path | None = None,
    override_parameters: str | None = None,
) -> str:
    """Serialize the YAML custom-source settings for OpenMC's compiled plugin."""
    custom = config.custom_source
    if override_parameters is not None:
        parameters = override_parameters
    elif custom is None:
        parameters = DEFAULT_PARAMETERS
    else:
        profile = ",".join(
            f"{radius_fraction:g}:{density:g}"
            for radius_fraction, density in custom.neutral_profile
        )
        reactor_centers = "|".join(
            f"{x_cm:g},{y_cm:g}" for x_cm, y_cm in custom.reactor_centers_cm
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
                f"reactor_centers_cm={reactor_centers}",
            )
        )
    if source_spectrum_base is not None:
        parameters = ";".join(
            (
                parameters,
                f"source_spectrum_file={Path(source_spectrum_base).resolve()}",
                "source_spectrum_bins_mev="
                + ",".join(f"{edge:.17g}" for edge in config.tally.energy_bins_mev),
            )
        )
    return parameters


def collect_source_birth_spectrum(
    source_spectrum_base: str | Path, destination: str | Path
) -> Path:
    """Combine per-process custom-source histograms into a normalized CSV."""
    base = Path(source_spectrum_base)
    partial_files = sorted(base.parent.glob(f"{base.name}.*.csv"))
    if not partial_files:
        raise RuntimeError(
            "The compiled source did not write its birth-energy spectrum; "
            "rerun with the current custom-source plugin."
        )
    edges: list[tuple[float, float]] | None = None
    counts: list[int] | None = None
    underflow = 0
    overflow = 0
    for path in partial_files:
        with path.open(newline="", encoding="utf-8") as source_file:
            rows = list(csv.DictReader(source_file))
        file_edges = [
            (float(row["energy_lower_mev"]), float(row["energy_upper_mev"]))
            for row in rows
        ]
        file_counts = [int(row["neutron_count"]) for row in rows]
        if edges is None:
            edges = file_edges
            counts = [0] * len(file_counts)
        if file_edges != edges or len(file_counts) != len(counts):
            raise ValueError(
                f"Custom source spectrum bins differ in {path.name}"
            )
        assert counts is not None
        counts = [total + value for total, value in zip(counts, file_counts, strict=True)]
        underflow += int(rows[0]["underflow_count"])
        overflow += int(rows[0]["overflow_count"])
    assert edges is not None and counts is not None
    total = sum(counts)
    if total <= 0:
        raise RuntimeError("The compiled source recorded no emitted neutrons")
    if underflow or overflow:
        raise ValueError(
            "Custom source birth energies fall outside tally.energy_bins_mev "
            f"({underflow} underflow, {overflow} overflow); widen the configured bins."
        )
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "energy_lower_mev",
                "energy_upper_mev",
                "neutron_count",
                "probability_per_source_neutron",
            )
        )
        for (lower, upper), count in zip(edges, counts, strict=True):
            writer.writerow((lower, upper, count, count / total))
    for path in partial_files:
        path.unlink()
    return target


def run_custom_source_model(
    config: SimulationConfig, output_dir: str | Path
) -> tuple[Path, Path]:
    """Run a model with its YAML custom source and persist source birth energies."""
    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC is required to run a model with its compiled custom source"
        ) from exc
    from .batch_report import run_with_batch_reports
    from .model import build_model

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    library = compile_custom_source(output_library=destination / "libiec_custom_source.so")
    source_spectrum_base = destination / f"source_birth_spectrum.{uuid.uuid4().hex}"
    model = build_model(config)
    model.settings.source = openmc.CompiledSource(
        library=library,
        parameters=custom_source_parameters(config, source_spectrum_base),
    )
    statepoint = run_with_batch_reports(model, destination, config)
    spectrum = collect_source_birth_spectrum(
        source_spectrum_base, destination / "source_birth_spectrum.csv"
    )
    return statepoint, spectrum


def compile_custom_source(
    source_file: str | Path | None = None,
    output_library: str | Path | None = None,
    compiler: str | None = None,
) -> Path:
    """Build and return the OpenMC compiled-source shared library.

    The OpenMC headers and shared library are taken from the active Python
    environment. Set ``OPENMC_PREFIX`` if they are installed elsewhere.
    """
    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC must be importable in the active Python environment to compile "
            "the custom source."
        ) from exc
    if not hasattr(openmc, "CompiledSource"):
        raise RuntimeError(
            "The active OpenMC Python package does not provide CompiledSource."
        )

    prefix = Path(os.environ.get("OPENMC_PREFIX", sys.prefix)).resolve()
    include_dir = prefix / "include"
    library_dir = prefix / "lib"
    openmc_library = library_dir / "libopenmc.so"
    if not (include_dir / "openmc" / "source.h").is_file():
        raise FileNotFoundError(
            f"OpenMC headers were not found under {include_dir}; "
            "set OPENMC_PREFIX to the OpenMC installation prefix."
        )
    if not openmc_library.is_file():
        raise FileNotFoundError(
            f"OpenMC shared library was not found at {openmc_library}; "
            "set OPENMC_PREFIX to the OpenMC installation prefix."
        )

    compiler_path = compiler
    if compiler_path is None:
        compiler_names = (
            "c++",
            "g++",
            "clang++",
            "x86_64-conda-linux-gnu-c++",
        )
        compiler_path = next(
            (
                candidate
                for name in compiler_names
                if (candidate := shutil.which(name)) is not None
            ),
            None,
        )
        if compiler_path is None:
            compiler_path = next(
                (
                    str(candidate)
                    for name in compiler_names
                    if (candidate := prefix / "bin" / name).is_file()
                ),
                None,
            )
    if compiler_path is None:
        raise FileNotFoundError(
            "A C++ compiler was not found on PATH; install a compiler such as g++."
        )

    source_path = Path(source_file) if source_file else Path(__file__).with_name(
        "custom_source.cpp"
    )
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Custom source C++ file does not exist: {source_path}")

    destination = Path(output_library or Path.cwd() / "libiec_custom_source.so")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler_path,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        f"-I{include_dir}",
        str(source_path),
        f"-L{library_dir}",
        f"-Wl,-rpath,{library_dir}",
        "-lopenmc",
        "-o",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(
            f"Failed to compile the IEC source plugin with {compiler_path}:\n{details}"
        ) from exc
    return destination

"""Compile the IEC fusion source plugin against the active OpenMC installation."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


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

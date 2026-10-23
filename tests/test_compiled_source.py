import csv
from pathlib import Path

from iec_neutrons.compiled_source import (
    collect_source_birth_spectrum,
    custom_source_parameters,
)
from iec_neutrons.config import load_config
from iec_neutrons import cli


PROFILE_CONFIG = (
    Path(__file__).parents[1] / "config" / "gen_reactor_config_profile.yaml"
)


def test_custom_source_parameters_include_birth_energy_bins(tmp_path: Path) -> None:
    config = load_config(PROFILE_CONFIG)
    source_base = tmp_path / "source_birth_spectrum.run-id"

    parameters = custom_source_parameters(config, source_base)

    assert "tritium_fraction=0.5" in parameters
    assert f"source_spectrum_file={source_base.resolve()}" in parameters
    assert "source_spectrum_bins_mev=0,1.0000000000000001e-09" in parameters


def test_source_birth_spectra_merge_per_process_files(tmp_path: Path) -> None:
    source_base = tmp_path / "source_birth_spectrum.run-id"
    rows = [
        ("0", "1", 3, 0, 0),
        ("1", "2", 1, 0, 0),
    ]
    for process_id, counts in (("101", (rows[0][2], rows[1][2])), ("202", (2, 2))):
        path = Path(f"{source_base}.{process_id}.csv")
        with path.open("w", newline="", encoding="utf-8") as source_file:
            writer = csv.writer(source_file)
            writer.writerow(
                (
                    "energy_lower_mev",
                    "energy_upper_mev",
                    "neutron_count",
                    "underflow_count",
                    "overflow_count",
                )
            )
            for row, count in zip(rows, counts, strict=True):
                writer.writerow((*row[:2], count, *row[3:]))

    destination = collect_source_birth_spectrum(
        source_base, tmp_path / "source_birth_spectrum.csv"
    )

    with destination.open(newline="", encoding="utf-8") as output_file:
        output_rows = list(csv.DictReader(output_file))
    assert [int(row["neutron_count"]) for row in output_rows] == [5, 3]
    assert [float(row["probability_per_source_neutron"]) for row in output_rows] == [
        0.625,
        0.375,
    ]
    assert not list(tmp_path.glob("source_birth_spectrum.run-id.*.csv"))


def test_cli_automatically_uses_yaml_custom_source(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    monkeypatch.setattr(
        cli,
        "run_custom_source_model",
        lambda config, output_dir: (
            calls.append((config.custom_source is not None, output_dir))
            or (tmp_path / "statepoint.2.h5", tmp_path / "source_birth_spectrum.csv")
        ),
    )
    monkeypatch.setattr(cli, "print_run_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "export_he3_spectra_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "iec-neutrons",
            "run",
            str(PROFILE_CONFIG),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    assert cli.main() == 0
    assert calls == [(True, tmp_path / "output")]

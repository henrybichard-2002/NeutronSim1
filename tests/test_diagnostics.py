import csv
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from iec_neutrons.config import load_config
from iec_neutrons.diagnostics import (
    _spectrum_summary,
    export_he3_spectra_csv,
)


CONFIG_DIR = Path(__file__).parents[1] / "config"


def test_spectrum_summary_uses_bin_centres_and_reports_upper_edge() -> None:
    summary = _spectrum_summary([0.0, 1.0, 3.0], (0.0, 1.0, 2.0, 3.0))

    assert summary is not None
    assert summary[0] == pytest.approx(2.25)
    assert summary[1] == 3.0


def test_spectrum_summary_handles_empty_spectrum() -> None:
    assert _spectrum_summary([0.0, 0.0], (0.0, 1.0, 2.0)) is None


def test_export_he3_spectra_csv_writes_all_detector_bins(tmp_path, monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    config = load_config(CONFIG_DIR / "gen_reactor.yaml")
    components = [
        component
        for component in config.geometry.components
        if component.material == "helium3_gas"
    ]

    class FakeEnergyFilter:
        bins = np.asarray([[0.0, 1.0e6], [1.0e6, 2.0e6]])

    class FakeTally:
        filters = [FakeEnergyFilter()]

        def __init__(self, mean, std_dev):
            self.values = {"mean": mean, "std_dev": std_dev}

        def get_values(self, *, scores, value):
            return np.asarray(self.values[value])

    tallies = {}
    for component in components:
        tallies[f"he3_neutron_spectrum_{component.name}"] = FakeTally(
            [10.0, 20.0], [1.0, 2.0]
        )
        tallies[f"he3_n_p_spectrum_{component.name}"] = FakeTally(
            [3.0, 4.0], [0.3, 0.4]
        )

    class FakeStatePoint:
        def __init__(self, _path):
            self.tallies = {
                name: SimpleNamespace(name=name) for name in tallies
            }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_tally(self, *, name):
            return tallies[name]

    monkeypatch.setitem(
        sys.modules,
        "openmc",
        SimpleNamespace(StatePoint=FakeStatePoint, EnergyFilter=FakeEnergyFilter),
    )
    output = export_he3_spectra_csv(
        tmp_path / "statepoint.h5",
        config,
        tmp_path / "he3_detector_spectra.csv",
    )

    assert output is not None and output.is_file()
    with output.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2 * len(components)
    assert {row["detector"] for row in rows} == {
        component.name for component in components
    }
    gas = components[0]
    gas_volume_cm3 = math.pi * gas.dimensions_cm[0] ** 2 * gas.dimensions_cm[1]
    first_bin = next(row for row in rows if row["detector"] == gas.name)
    assert float(first_bin["energy_min_mev"]) == 0.0
    assert float(first_bin["energy_max_mev"]) == 1.0
    assert float(first_bin["neutron_flux_mean_cm-2_source-1_MeV-1"]) == pytest.approx(
        10.0 / gas_volume_cm3
    )
    assert float(first_bin["neutron_flux_std_dev_cm-2_source-1_MeV-1"]) == pytest.approx(
        1.0 / gas_volume_cm3
    )
    assert float(
        first_bin["he3_np_capture_mean_reactions_source-1_MeV-1"]
    ) == pytest.approx(3.0)
    assert float(
        first_bin["he3_np_capture_std_dev_reactions_source-1_MeV-1"]
    ) == pytest.approx(0.3)

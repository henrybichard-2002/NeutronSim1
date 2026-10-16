import pytest

from iec_neutrons.diagnostics import _spectrum_summary


def test_spectrum_summary_uses_bin_centres_and_reports_upper_edge() -> None:
    summary = _spectrum_summary([0.0, 1.0, 3.0], (0.0, 1.0, 2.0, 3.0))

    assert summary is not None
    assert summary[0] == pytest.approx(2.25)
    assert summary[1] == 3.0


def test_spectrum_summary_handles_empty_spectrum() -> None:
    assert _spectrum_summary([0.0, 0.0], (0.0, 1.0, 2.0)) is None

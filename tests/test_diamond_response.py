from pathlib import Path

import numpy as np
import pytest

from diamond_response import kinematics as k
from diamond_response.config import ConfigError, deposited_edges_mev, incident_edges_mev, load_config
from diamond_response.response import resolution_kernel
from diamond_response.spectra import dt_neutron_energy, gaussian_fluence_fractions

CONFIG = Path(__file__).parents[1] / "config" / "diamond_dt.yaml"


def test_default_config_grids():
    cfg = load_config(CONFIG)
    inc = incident_edges_mev(cfg)
    assert np.all(np.diff(inc) > 0)
    assert inc[0] == 1.0 and inc[-1] == 17.0
    assert deposited_edges_mev(cfg)[-1] == pytest.approx(16.0)


def test_bad_config_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(CONFIG.read_text().replace("thickness_um: 500.0", "thickness_um: -1"))
    with pytest.raises(ConfigError):
        load_config(bad)


def test_two_body_decay_conserves_energy_momentum():
    rng = np.random.default_rng(1)
    n = 1000
    m1, m2 = k.MASS["a"], k.MASS["9Be"]
    e_sys = np.full(n, m1 + m2 + 8.0)
    p_sys = np.zeros((n, 3))
    (p1, e1), (p2, e2), ok = k.decay(e_sys, p_sys, m1, m2, rng)
    assert ok.all()
    assert np.allclose(e1 + e2, e_sys)
    assert np.allclose(p1 + p2, 0.0, atol=1e-9)


def test_dt_neutron_energy():
    assert dt_neutron_energy(0.1, 90.0) == pytest.approx(14.05, abs=0.1)
    assert dt_neutron_energy(0.1, 0.0) > dt_neutron_energy(0.1, 180.0)
    assert dt_neutron_energy(0.0001, 0.0) == pytest.approx(14.03, abs=0.05)


def test_gaussian_fractions_sum_to_one():
    edges = np.linspace(12, 16, 41)
    assert gaussian_fluence_fractions(edges, 14.0, 0.2).sum() == pytest.approx(1.0, abs=1e-6)


def test_resolution_kernel_normalised():
    edges = np.arange(0, 10.001, 0.01)
    g = resolution_kernel(edges, 40.0, 0.008)
    assert g[:, 500].sum() == pytest.approx(1.0, abs=1e-3)


def test_alpha_range_matches_astar():
    pytest.importorskip("pycatima")
    from diamond_response.stopping import get_table

    table = get_table(4, 2, 3.515, True)
    assert 20e-4 < table.range_of(8.0) < 30e-4  # cm; ASTAR gives ~25 um


@pytest.mark.slow
def test_openmc_column_has_n_alpha_peak(tmp_path):
    pytest.importorskip("openmc")
    pytest.importorskip("pycatima")
    from diamond_response.response import Simulator

    cfg = load_config(CONFIG)
    cfg.statistics.histories_per_chunk = 2e6
    cfg.statistics.min_chunks = 2
    cfg.statistics.max_chunks = 2
    cfg.incident.sampling = "center"
    sim = Simulator(cfg)
    edges = incident_edges_mev(cfg)
    j = int(np.searchsorted(edges, 14.0))
    sim.run_column(tmp_path, j, log=lambda *a: None)
    from diamond_response.response import assemble

    rm = assemble(cfg, tmp_path)
    col = rm.matrix[:, j]
    peak = rm.dep_centers[np.argmax(np.where(rm.dep_centers > 7.5, col, 0))]
    assert 8.1 < peak < 8.4

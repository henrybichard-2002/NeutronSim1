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
    assert inc[0] == 1.0 and inc[-1] >= 16.0
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
    assert 8.1 < peak < 8.9  # depends on the bin centre of the configured grid


def test_reactor_config_and_fold(tmp_path):
    from diamond_response import reactor
    from diamond_response.config import load_config
    from diamond_response.response import ResponseMatrix

    cfg = load_config("config/diamond_in_reactor.yaml")
    assert cfg.reactor.distance_from_surface_cm == 5.0 and cfg.reactor.z_cm == 0.0
    assert incident_edges_mev(cfg)[0] == 1.0 and len(incident_edges_mev(cfg)) > 100

    inc = np.array([1.0, 2.0, 3.0])
    dep = np.array([0.0, 1.0, 2.0])
    rm = ResponseMatrix(inc, dep, np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([[0.1, 0.0], [0.0, 0.2]]),
                        np.ones(2), 1.0, {})
    field = reactor.ReactorField(
        inc_edges=inc, fluence_rate=np.array([10.0, 5.0]), fluence_rate_err=np.array([1.0, 0.5]),
        coarse_edges=np.array([1.0, 3.0]), class_edges_deg=np.array([0.0, 45.0, 90.0]),
        class_fraction=np.array([[0.25], [0.75]]), class_mean_angle_deg=np.array([20.0, 60.0]),
        norm_rel_error=0.0, source_rate=1.0, detector_x_cm=12.0, detector_z_cm=0.0)
    _, rate, err = reactor.fold_field(field, {0: rm, 1: rm})
    assert np.allclose(rate, [10.0, 10.0])
    assert np.isclose(err[0], np.sqrt((0.25**2 + 0.75**2) * (1.0 + 1.0)))


def test_unfolding_recovers_known_spectrum(tmp_path):
    from types import SimpleNamespace

    from diamond_response.config import SimulationConfig, UnfoldConfig
    from diamond_response.response import ResponseMatrix
    from diamond_response.unfold import run_unfolding

    inc_edges = np.linspace(1.0, 16.0, 31)
    dep_edges = np.linspace(0.0, 16.0, 81)
    ic, dc = 0.5 * (inc_edges[1:] + inc_edges[:-1]), 0.5 * (dep_edges[1:] + dep_edges[:-1])
    # triangular response: a neutron deposits up to ~its energy, with a small peak at 0.4*E
    m = np.where(dc[:, None] < 0.5 * ic[None, :], 1.0, 0.0) + 3.0 * np.exp(-(((dc[:, None] - 0.4 * ic[None, :]) / 0.3) ** 2))
    m = 1e-4 * m / m.sum(axis=0).max()
    rm = ResponseMatrix(inc_edges, dep_edges, m, 0.01 * m, 1, 1.0, {})
    truth = 1e6 * (np.exp(-0.5 * ((ic - 4.0) / 1.0) ** 2) + 0.7 * np.exp(-0.5 * ((ic - 12.0) / 1.0) ** 2))
    field = SimpleNamespace(fluence_rate=truth, fluence_rate_err=0.01 * truth,
                            class_fractions_fine=lambda: {0: np.ones_like(truth)})
    cfg = SimulationConfig()
    cfg.unfold = UnfoldConfig(live_time_s=1e5, replicas=5, with_resolution=False, min_pulse_mev=0.0)
    run_unfolding(cfg, field, {0: rm}, tmp_path, log=lambda *_: None)
    data = np.genfromtxt(tmp_path / "unfolded_spectrum.csv", delimiter=",", names=True)
    assert data["unfolded_effective_n_cm2_s"].sum() == pytest.approx(truth.sum(), rel=0.1)
    assert (tmp_path / "plots" / "unfolded_vs_true.png").exists()


def test_windows_path_and_measured_rebin(tmp_path):
    from diamond_response.compare import load_measured, resolve_path

    assert str(resolve_path(r"C:\Users\x\data")) == "/mnt/c/Users/x/data"
    energy = np.linspace(0.0, 10.0, 1001)
    counts = np.ones(1001, dtype=np.int32)
    np.savez(tmp_path / "all_datasets.npz", **{"channel a_spectrum_energy [MeV]": energy,
                                              "channel a_spectrum_counts": counts})
    (tmp_path / "metadata.json").write_text('{"duration [s]": 10.0}')
    edges = np.linspace(0.0, 16.0, 161)
    m = load_measured(tmp_path, edges)
    assert m.counts.sum() == 1000  # overflow bin dropped
    assert m.live_time_s == 10.0

"""Diamond detector response in the neutron field of an ``iec_neutrons`` reactor model.

Stage 1 transports the reactor's custom D-T source and tallies, at the detector position,
the neutron fluence spectrum (averaged over an axisymmetric ring to gain statistics) and the
distribution of incidence angles relative to the detector normal (the radial direction).
Stage 2 simulates the response matrix at a representative incidence angle for each angle class.
Stage 3 folds fluence and matrices into an absolute pulse-height spectrum in counts per second.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from .config import SimulationConfig, deposited_edges_mev, incident_edges_mev
from .response import ResponseMatrix, Simulator, assemble, npz_dir, resolution_kernel

NEUTRON_FIELD_FILE = "reactor_field.npz"


@dataclasses.dataclass
class ReactorField:
    inc_edges: np.ndarray            # MeV, fine grid (the response-matrix incident grid)
    fluence_rate: np.ndarray         # n/cm^2/s per fine bin at the detector
    fluence_rate_err: np.ndarray
    coarse_edges: np.ndarray         # MeV, grid on which the angle distribution was measured
    class_edges_deg: np.ndarray
    class_fraction: np.ndarray       # [class, coarse bin], sums to 1 over classes
    class_mean_angle_deg: np.ndarray
    norm_rel_error: float            # statistical error of the flux calibration (common to all bins)
    source_rate: float               # source neutrons per second matching the configured surface flux
    detector_x_cm: float
    detector_z_cm: float
    fluence_below_grid: float = 0.0  # n/cm^2/s from 0 up to the lowest incident energy (not simulated)
    fluence_above_grid: float = 0.0  # n/cm^2/s above the highest incident energy (not simulated)
    angle_to_ring_ratio: float = 1.0  # angle-box flux / ring flux over the same energies (consistency check)

    def save(self, path: Path) -> None:
        np.savez(path, **{f.name: getattr(self, f.name) for f in dataclasses.fields(self)})

    @classmethod
    def load(cls, path: Path) -> "ReactorField":
        data = np.load(path)
        return cls(**{f.name: (float(data[f.name]) if data[f.name].ndim == 0 else data[f.name])
                      for f in dataclasses.fields(cls) if f.name in data.files})

    def matches(self, config: SimulationConfig) -> bool:
        """True if this saved field used the current energy grid and angle classes."""
        edges = incident_edges_mev(config)
        classes = np.asarray(config.reactor.angle_edges_deg, float)
        coarse = np.asarray(config.reactor.angle_energy_edges_mev, float)
        return (self.inc_edges.shape == edges.shape and np.allclose(self.inc_edges, edges)
                and self.class_edges_deg.shape == classes.shape and np.allclose(self.class_edges_deg, classes)
                and self.coarse_edges.shape == coarse.shape and np.allclose(self.coarse_edges, coarse))

    def class_fractions_fine(self) -> np.ndarray:
        """Angle-class fractions on the fine incident grid, ``[class, fine bin]``."""
        centers = 0.5 * (self.inc_edges[1:] + self.inc_edges[:-1])
        index = np.clip(np.searchsorted(self.coarse_edges, centers, side="right") - 1, 0, len(self.coarse_edges) - 2)
        return self.class_fraction[:, index]


def resolve_reactor_config(config: SimulationConfig) -> Path:
    name = Path(config.reactor.config)
    candidates = [name] if name.is_absolute() else [(config.base_dir or Path.cwd()) / name, Path.cwd() / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"reactor config not found: {config.reactor.config}")


def _cross_sections(config: SimulationConfig) -> Path:
    rc = config.reactor
    if rc.cross_sections:
        path = Path(rc.cross_sections)
        path = path if path.is_absolute() else (config.base_dir or Path.cwd()) / path
    else:
        reactor_yaml = resolve_reactor_config(config)
        path = reactor_yaml.parent / "cross_sections" / f"{reactor_yaml.stem}.xml"
    if not path.is_file():
        raise FileNotFoundError(f"cross-section library not found: {path} (set reactor.cross_sections)")
    return path.resolve()


def _load_iec(config: SimulationConfig):
    from iec_neutrons.config import load_config as load_iec

    rc = config.reactor
    iec = load_iec(resolve_reactor_config(config))
    if iec.custom_source is None or iec.normalization is None:
        raise ValueError("the reactor config needs a custom_source and a normalization section")
    settings = dataclasses.replace(iec.settings, particles=int(rc.particles), batches=int(rc.batches), inactive=0)
    return dataclasses.replace(iec, settings=settings, detector=dataclasses.replace(iec.detector, enabled=False))


def detector_position_cm(config: SimulationConfig, iec) -> tuple[float, float, float]:
    surface = next(c for c in iec.geometry.components if c.name == iec.normalization.surface_component)
    if abs(surface.center_cm[0]) > 1e-9 or abs(surface.center_cm[1]) > 1e-9:
        raise ValueError("the normalisation surface must be a cylinder on the z axis (axisymmetric field assumed)")
    radius = surface.dimensions_cm[0] + config.reactor.distance_from_surface_cm
    return radius, 0.0, config.reactor.z_cm


def build_field_model(config: SimulationConfig, iec):
    import openmc
    from iec_neutrons.model import build_model

    rc = config.reactor
    model = build_model(iec)
    x, y, z = detector_position_cm(config, iec)
    fine_ev = incident_edges_mev(config) * 1e6
    coarse = np.asarray(rc.angle_energy_edges_mev, float)
    neutron = openmc.ParticleFilter("neutron")

    def ring(r_in, r_out, z_lo, z_hi, name):
        mesh = openmc.CylindricalMesh(r_grid=[r_in, r_out], phi_grid=[0.0, 2.0 * math.pi], z_grid=[z_lo, z_hi], name=name)
        return mesh, math.pi * (r_out**2 - r_in**2) * (z_hi - z_lo)

    spectrum_mesh, spectrum_volume = ring(
        x - rc.ring_thickness_cm / 2, x + rc.ring_thickness_cm / 2, z - rc.ring_height_cm / 2, z + rc.ring_height_cm / 2, "diamond_ring")
    surface = next(c for c in iec.geometry.components if c.name == iec.normalization.surface_component)
    r_surface = surface.dimensions_cm[0]
    norm_mesh, norm_volume = ring(r_surface, r_surface + iec.normalization.sample_width_cm, -0.5, 0.5, "surface_ring")

    half = rc.angle_box_cm / 2
    angle_mesh = openmc.RegularMesh(name="diamond_angle_box")
    angle_mesh.dimension = (1, 1, 1)
    angle_mesh.lower_left = (x - half, y - half, z - half)
    angle_mesh.upper_right = (x + half, y + half, z + half)

    spectrum = openmc.Tally(name="diamond_fluence_spectrum")
    spectrum.filters = [openmc.MeshFilter(spectrum_mesh), neutron, openmc.EnergyFilter(fine_ev)]
    spectrum.scores = ["flux"]
    norm = openmc.Tally(name="surface_flux_ring")
    norm.filters = [openmc.MeshFilter(norm_mesh), neutron]
    norm.scores = ["flux"]
    angle = openmc.Tally(name="diamond_angle_distribution")
    angle.filters = [openmc.MeshFilter(angle_mesh), neutron, openmc.EnergyFilter(coarse * 1e6),
                     openmc.PolarFilter(ANGLE_BINS), openmc.AzimuthalFilter(ANGLE_BINS)]
    angle.scores = ["flux"]
    wide = openmc.Tally(name="diamond_fluence_wide")
    wide.filters = [openmc.MeshFilter(spectrum_mesh), neutron,
                    openmc.EnergyFilter([0.0, fine_ev[0], fine_ev[-1], 3.0e7])]
    wide.scores = ["flux"]
    model.tallies = openmc.Tallies([spectrum, norm, angle, wide])
    return model, {"spectrum_volume": spectrum_volume, "norm_volume": norm_volume, "box_volume": rc.angle_box_cm**3}


ANGLE_BINS = 36


def run_field(config: SimulationConfig, outdir: Path, threads: int | None = None) -> ReactorField:
    import openmc
    from iec_neutrons.compiled_source import compile_custom_source, custom_source_parameters

    data = npz_dir(outdir)
    data.mkdir(parents=True, exist_ok=True)
    iec = _load_iec(config)
    model, volumes = build_field_model(config, iec)
    model.materials.cross_sections = str(_cross_sections(config))
    library = compile_custom_source(output_library=data / "libiec_custom_source.so")
    model.settings.source = openmc.CompiledSource(library=library, parameters=custom_source_parameters(iec))
    run_dir = data / "reactor_run"
    run_dir.mkdir(exist_ok=True)
    for stale in run_dir.glob("statepoint.*.h5"):
        stale.unlink()
    model.settings.statepoint = {"batches": [iec.settings.batches]}
    statepoint = model.run(cwd=str(run_dir), threads=threads)
    return extract_field(config, iec, Path(statepoint), volumes)


def extract_field(config: SimulationConfig, iec, statepoint: Path, volumes: dict) -> ReactorField:
    import openmc

    rc = config.reactor
    x, _, z = detector_position_cm(config, iec)
    with openmc.StatePoint(statepoint) as sp:
        spectrum = sp.get_tally(name="diamond_fluence_spectrum")
        norm = sp.get_tally(name="surface_flux_ring")
        angle = sp.get_tally(name="diamond_angle_distribution")
        wide = sp.get_tally(name="diamond_fluence_wide").get_values(value="mean").reshape(-1)
        flux = spectrum.get_values(value="mean").reshape(-1)
        flux_err = spectrum.get_values(value="std_dev").reshape(-1)
        norm_mean = float(norm.get_values(value="mean").reshape(-1)[0])
        norm_err = float(norm.get_values(value="std_dev").reshape(-1)[0])
        coarse = np.asarray(rc.angle_energy_edges_mev, float)
        ang = angle.get_values(value="mean").reshape(len(coarse) - 1, ANGLE_BINS, ANGLE_BINS)
    if norm_mean <= 0:
        raise RuntimeError("no flux scored in the normalisation ring; increase reactor.particles")
    source_rate = iec.normalization.surface_flux_n_cm2_s / (norm_mean / volumes["norm_volume"])
    scale = source_rate / volumes["spectrum_volume"]
    fine = incident_edges_mev(config)
    inside = np.where(coarse[1:] <= fine[-1] + 1e-9)[0]
    box_in = ang[inside].sum() / volumes["box_volume"] if len(inside) else 0.0
    ring_cmp = sum(flux[(0.5 * (fine[1:] + fine[:-1]) > lo) & (0.5 * (fine[1:] + fine[:-1]) <= hi)].sum()
                   for lo, hi in zip(coarse[inside], coarse[inside + 1])) / volumes["spectrum_volume"]
    ratio = float(box_in / ring_cmp) if ring_cmp > 0 else 1.0

    theta = 0.5 * (np.linspace(0, math.pi, ANGLE_BINS + 1)[1:] + np.linspace(0, math.pi, ANGLE_BINS + 1)[:-1])
    phi = 0.5 * (np.linspace(-math.pi, math.pi, ANGLE_BINS + 1)[1:] + np.linspace(-math.pi, math.pi, ANGLE_BINS + 1)[:-1])
    incidence = np.degrees(np.arccos(np.clip(np.abs(np.sin(theta)[:, None] * np.cos(phi)[None, :]), 0.0, 1.0)))
    edges = np.asarray(rc.angle_edges_deg, float)
    which = np.clip(np.searchsorted(edges, incidence, side="right") - 1, 0, len(edges) - 2)
    n_class = len(edges) - 1
    per_class = np.zeros((n_class, len(coarse) - 1))
    angle_sum = np.zeros(n_class)
    weight_sum = np.zeros(n_class)
    for k in range(n_class):
        mask = which == k
        per_class[k] = ang[:, mask].sum(axis=1)
        angle_sum[k] = (ang[:, mask] * incidence[mask]).sum()
        weight_sum[k] = ang[:, mask].sum()
    total = per_class.sum(axis=0)
    overall = per_class.sum(axis=1)
    overall = overall / overall.sum() if overall.sum() > 0 else np.full(n_class, 1.0 / n_class)
    fraction = np.where(total > 0, per_class / np.where(total > 0, total, 1.0), overall[:, None])
    mid = 0.5 * (edges[1:] + edges[:-1])
    mean_angle = np.where(weight_sum > 0, angle_sum / np.where(weight_sum > 0, weight_sum, 1.0), mid)
    return ReactorField(
        inc_edges=incident_edges_mev(config), fluence_rate=flux * scale, fluence_rate_err=flux_err * scale,
        coarse_edges=coarse, class_edges_deg=edges, class_fraction=fraction, class_mean_angle_deg=mean_angle,
        norm_rel_error=norm_err / norm_mean, source_rate=source_rate, detector_x_cm=x, detector_z_cm=z,
        fluence_below_grid=float(wide[0] * scale), fluence_above_grid=float(wide[2] * scale),
        angle_to_ring_ratio=ratio)


def run_class_matrices(config: SimulationConfig, field: ReactorField, outdir: Path, threads=None, log=print):
    """Response matrix for every angle class that receives flux; returns {class index: ResponseMatrix}."""
    matrices = {}
    fractions = field.class_fractions_fine()
    flux_weight = fractions * field.fluence_rate[None, :]
    for k, angle in enumerate(field.class_mean_angle_deg):
        if flux_weight[k].sum() <= 0:
            continue
        detector = dataclasses.replace(config.detector, incidence_angle_deg=float(angle))
        stats = dataclasses.replace(config.statistics, target_rel_error=config.reactor.matrix_target_rel_error)
        cfg_k = dataclasses.replace(config, detector=detector, statistics=stats)
        folder = npz_dir(outdir) / f"matrix_class{k}_{field.class_edges_deg[k]:g}-{field.class_edges_deg[k + 1]:g}deg"
        log(f"Angle class {k}: {field.class_edges_deg[k]:g}-{field.class_edges_deg[k + 1]:g} deg, "
            f"simulated at {angle:.1f} deg ({100 * flux_weight[k].sum() / flux_weight.sum():.1f}% of fluence)")
        sim = Simulator(cfg_k, threads=threads)
        for j in range(len(field.fluence_rate)):
            # columns carrying little fluence need less precision
            weight = flux_weight[k, j] / max(flux_weight[k].max(), 1e-300)
            if weight <= 0:
                continue
            target = min(0.3, stats.target_rel_error / math.sqrt(max(weight, 1e-4)))
            sim.run_column(folder, j, log, target_rel_error=target)
        matrices[k] = assemble(cfg_k, folder)
    return matrices


def fold_field(field: ReactorField, matrices: dict[int, ResponseMatrix]):
    """Counts per second per deposited-energy bin and its statistical uncertainty."""
    fractions = field.class_fractions_fine()
    any_matrix = next(iter(matrices.values()))
    rate = np.zeros(any_matrix.matrix.shape[0])
    var = np.zeros_like(rate)
    for k, rm in matrices.items():
        phi = field.fluence_rate * fractions[k]
        phi_err = field.fluence_rate_err * fractions[k]
        rate += rm.matrix @ phi
        var += (rm.matrix**2) @ phi_err**2 + (rm.error**2) @ phi**2
    return any_matrix.dep_edges, rate, np.sqrt(var)


def write_outputs(config: SimulationConfig, field: ReactorField, matrices: dict[int, ResponseMatrix], outdir: Path) -> list[Path]:
    dep_edges, rate, err = fold_field(field, matrices)
    width = np.diff(dep_edges)
    centers = 0.5 * (dep_edges[1:] + dep_edges[:-1])
    paths = []

    def table(path, columns, header):
        np.savetxt(path, np.column_stack(columns), delimiter=",", header=header, comments="", fmt="%.8g")
        paths.append(path)

    header = ("deposited_low_MeV,deposited_high_MeV,deposited_center_MeV,counts_per_s,"
              "uncertainty_counts_per_s,relative_uncertainty,counts_per_s_per_MeV,uncertainty_counts_per_s_per_MeV")
    rel = np.divide(err, rate, out=np.zeros_like(err), where=rate > 0)
    table(outdir / "detector_response.csv",
          [dep_edges[:-1], dep_edges[1:], centers, rate, err, rel, rate / width, err / width], header)

    g = resolution_kernel(dep_edges, config.resolution.noise_kev, config.resolution.relative)
    rate_r, err_r = g @ rate, np.sqrt((g**2) @ err**2)
    rel_r = np.divide(err_r, rate_r, out=np.zeros_like(err_r), where=rate_r > 0)
    table(outdir / "detector_response_with_resolution.csv",
          [dep_edges[:-1], dep_edges[1:], centers, rate_r, err_r, rel_r, rate_r / width, err_r / width], header)

    fractions = field.class_fractions_fine()
    inc_centers = 0.5 * (field.inc_edges[1:] + field.inc_edges[:-1])
    flux_rel = np.divide(field.fluence_rate_err, field.fluence_rate, out=np.zeros_like(field.fluence_rate),
                         where=field.fluence_rate > 0)
    angle_cols = [fractions[k] for k in range(fractions.shape[0])]
    angle_names = ",".join(f"fraction_{field.class_edges_deg[k]:g}-{field.class_edges_deg[k + 1]:g}deg"
                           for k in range(fractions.shape[0]))
    table(outdir / "incident_neutron_spectrum.csv",
          [field.inc_edges[:-1], field.inc_edges[1:], inc_centers, field.fluence_rate, field.fluence_rate_err, flux_rel,
           *angle_cols],
          "energy_low_MeV,energy_high_MeV,energy_center_MeV,fluence_rate_n_cm2_s,uncertainty_n_cm2_s,"
          f"relative_uncertainty,{angle_names}")

    roi = centers > config.statistics.roi_min_mev
    summary = {
        "detector_x_cm": field.detector_x_cm, "detector_z_cm": field.detector_z_cm,
        "source_rate_n_per_s": field.source_rate,
        "fluence_rate_simulated_range_n_cm2_s": float(field.fluence_rate.sum()),
        "simulated_energy_range_MeV": [float(field.inc_edges[0]), float(field.inc_edges[-1])],
        "fluence_rate_below_range_n_cm2_s": field.fluence_below_grid,
        "fluence_rate_above_range_n_cm2_s": field.fluence_above_grid,
        "fraction_of_fluence_not_simulated": (field.fluence_below_grid + field.fluence_above_grid)
        / max(field.fluence_below_grid + field.fluence_above_grid + float(field.fluence_rate.sum()), 1e-300),
        "angle_box_to_ring_flux_ratio": field.angle_to_ring_ratio,
        "counts_per_s_total": float(rate.sum()),
        "counts_per_s_above_roi_min": float(rate[roi].sum()),
        "uncertainty_counts_per_s_above_roi_min": float(np.sqrt((err[roi] ** 2).sum())),
        "roi_min_mev": config.statistics.roi_min_mev,
        "angle_classes_deg": field.class_edges_deg.tolist(),
        "angle_class_simulated_at_deg": field.class_mean_angle_deg.tolist(),
        "flux_calibration_rel_error_common_to_all_bins": field.norm_rel_error,
        "note": "Uncertainties are statistical (reactor tally + response matrix), bins treated as independent.",
    }
    path = outdir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    paths.append(path)
    return paths


def run_all(config: SimulationConfig, outdir: Path, threads=None, rerun_field=False, log=print) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    field_path = npz_dir(outdir) / NEUTRON_FIELD_FILE
    legacy = outdir / NEUTRON_FIELD_FILE
    if legacy.exists() and not field_path.exists():
        field_path.parent.mkdir(parents=True, exist_ok=True)
        legacy.rename(field_path)
    saved = ReactorField.load(field_path) if field_path.exists() else None
    if saved is not None and not rerun_field and saved.matches(config):
        field = saved
        log(f"Using saved reactor field {field_path}")
    else:
        if saved is not None and not rerun_field:
            log("Saved reactor field does not match the configured energy grid / angle classes; re-running it")
        field = run_field(config, outdir, threads)
        field.save(field_path)
    log(f"Detector at x = {field.detector_x_cm:.2f} cm, z = {field.detector_z_cm:.2f} cm; "
        f"angle classes {field.class_edges_deg.tolist()} deg, mean angles {np.round(field.class_mean_angle_deg, 1).tolist()}")
    probability = source_birth_probability(config, outdir, log)
    matrices = run_class_matrices(config, field, outdir, threads, log)
    paths = write_outputs(config, field, matrices, outdir)
    paths += write_plots(config, field, matrices, probability, outdir)
    from .unfold import run_unfolding

    paths += run_unfolding(config, field, matrices, outdir, log=log)
    return paths


def effective_matrix(field: ReactorField, matrices: dict[int, ResponseMatrix]) -> ResponseMatrix:
    """Angle-averaged response matrix of the reactor field (per unit fluence)."""
    fractions = field.class_fractions_fine()
    first = next(iter(matrices.values()))
    matrix = np.zeros_like(first.matrix)
    var = np.zeros_like(first.matrix)
    for k, rm in matrices.items():
        matrix += rm.matrix * fractions[k][None, :]
        var += (rm.error * fractions[k][None, :]) ** 2
    return ResponseMatrix(first.inc_edges, first.dep_edges, matrix, np.sqrt(var), first.histories, first.beam_area_cm2, {})


def _birth_edges() -> np.ndarray:
    return np.round(np.arange(0.0, 20.0 + 1e-9, 0.01), 6)


def source_birth_probability(config: SimulationConfig, outdir: Path, log=print) -> np.ndarray:
    """Emission spectrum of the custom source (probability per incident bin), via a short source-only transport run."""
    import csv

    import openmc
    from iec_neutrons.compiled_source import collect_source_birth_spectrum, compile_custom_source, custom_source_parameters
    from iec_neutrons.model import build_model

    data = npz_dir(outdir)
    saved = data / "source_birth_spectrum.csv"
    if not saved.exists():
        log("Sampling the custom-source birth spectrum")
        iec = _load_iec(config)
        iec = dataclasses.replace(
            iec, tally=dataclasses.replace(iec.tally, energy_bins_mev=tuple(_birth_edges())),
            settings=dataclasses.replace(iec.settings, particles=200_000, batches=1, inactive=0))
        model = build_model(iec)
        model.materials.cross_sections = str(_cross_sections(config))
        base = data / "birth_tmp"
        library = compile_custom_source(output_library=data / "libiec_custom_source.so")
        model.settings.source = openmc.CompiledSource(library=library, parameters=custom_source_parameters(iec, base))
        model.tallies = openmc.Tallies()
        model.settings.particles = 200_000
        run_dir = data / "birth_run"
        run_dir.mkdir(exist_ok=True)
        model.run(cwd=str(run_dir), output=False)
        collect_source_birth_spectrum(base, saved)
    with saved.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lower = np.array([float(r["energy_lower_mev"]) for r in rows])
    prob = np.array([float(r["probability_per_source_neutron"]) for r in rows])
    centers = lower + 0.005
    inc = incident_edges_mev(config)
    binned, _ = np.histogram(centers, bins=inc, weights=prob)
    fine = outdir / "source_birth_spectrum.csv"
    np.savetxt(fine, np.column_stack([inc[:-1], inc[1:], binned]), delimiter=",", comments="",
               header="energy_low_MeV,energy_high_MeV,probability_per_source_neutron", fmt="%.8g")
    return binned


def write_plots(config, field, matrices, probability, outdir: Path) -> list[Path]:
    from .plot import plot_reactor

    try:
        rm = effective_matrix(field, matrices)
        dep_edges, rate, err = fold_field(field, matrices)
        g = resolution_kernel(dep_edges, config.resolution.noise_kev, config.resolution.relative)
        return plot_reactor(rm, dep_edges, rate, err, g @ rate, np.sqrt((g**2) @ err**2),
                            float(field.fluence_rate.sum()), outdir / "plots")
    except ImportError:
        return []

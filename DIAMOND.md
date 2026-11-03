# sCVD diamond detector response matrix (D-T neutrons)

`diamond_response` simulates the pulse-height response matrix of a 4 x 4 mm x 500 um sCVD diamond
for neutrons of 1-17 MeV, with the grid refined around 14 MeV.

## Method
1. **OpenMC** (JEFF-4.0 data, 12C/13C plus N, O, Al, Si impurities/support) transports neutrons and writes
   `collision_track.h5`: every neutron collision in the diamond with pre-collision energy, post-collision
   direction, energy loss `dE` and reaction MT.
2. **Python** rebuilds the charged products of each collision (`events.py`): elastic and inelastic 12C/13C recoils
   (including 12C* -> 3 alpha via 8Be), and (n,p), (n,d), (n,alpha) with 9Be level sampling and 9Be* -> n + 2 alpha.
3. Ions are straight tracks through the active box (`transport.py`). Collected energy is the *electronic*
   energy loss from catima (pycatima) stopping powers, with ZBL nuclear stopping removed. Ions that leave
   the diamond deposit only part of their energy (edge effects are exact within this model).
4. Energy deposits of one source neutron within `pulse_window_ns` form a single pulse (multi-interaction pile-up).
5. Each incident bin is simulated with energies sampled uniformly in the bin (or at the centre).
   `matrix[i, j]` = pulses in deposited bin i per unit fluence (cm^2) in incident bin j. Chunks of 2e7
   histories (one OpenMC batch each, since OpenMC particle ids repeat between batches) run until the pulse count above
   `roi_min_mev` reaches `target_rel_error` (min/max chunks configurable). Columns are resumable.

## Usage
```
deactivate 2>/dev/null   # leave any other venv (e.g. .venv) first, or pip/python resolve to it
conda activate /home/Romulus1/miniforge3/envs/neutrons-openmc
which python pip         # both must be under .../envs/neutrons-openmc/bin
cd ~/NeutronSim
pip install -e . --no-deps   # installs the diamond-response command; --no-deps keeps pip from fetching OpenMC

diamond-response run config/diamond_dt.yaml --out runs/diamond_dt
diamond-response fold runs/diamond_dt/npz/response_matrix.npz --deuteron-mev 0.1 --angle-deg 90 --fwhm-mev 0.2
```
Optional `run` arguments: `--threads N` and `--columns 20 21` (do not type brackets; the shell treats them as globs).
Without installing: `PYTHONPATH=src python -m diamond_response.cli run ...`.
Runs resume from the saved `column_XXXX.npz` files in the output directory.
Outputs: `npz/response_matrix.npz` (all binary `.npz` data, incl. per-column files, lives in `npz/`), `response_matrix.csv`, `response_matrix_resolution.csv` (Gaussian noise + relative smearing),
plots. Requires `openmc`, `pycatima`, `numpy`, `scipy`, `h5py`, `PyYAML` (and `matplotlib` for plots).
The full default run takes ~7.5 min on 32 cores.

## Validation (default run, 14-14.1 MeV bin)
- (n,alpha0) peak at 8.3 MeV (E_n - 5.70 MeV minus lattice loss); elastic edge <= 4 MeV.
- Efficiency 1.07e-3 cm^2 above 0.5 MeV, consistent with sigma_t * thickness * area.
- 8 MeV alpha range in diamond ~ 25-30 um (ASTAR); tested in `tests/test_diamond_response.py`.

## Assumptions / limitations
- **12C(n,alpha)9Be level branching is an assumption** (`physics.alpha_levels`); JEFF-4.0 only has the summed
  cross section. This directly shapes the 5-9 MeV region; replace with evaluated/measured branching.
- Gamma rays and gamma-induced electrons, ion multiple scattering, 12B/11B excited states and 13C* particle
  emission are ignored. Escaping neutrons are not re-tracked; ions born in the support are not transported in.
- No charge-collection inefficiency or quenching; resolution parameters (40 keV noise, 0.8 %) are placeholders.
- Pulse window is measured from the first deposit of each history.

## Diamond response inside the IEC reactor

`config/diamond_in_reactor.yaml` simulates the pulse-height spectrum the diamond actually records in the
`config/gen_reactor_config_profile.yaml` reactor (1-17 MeV, 0.1 MeV incident grid).

```bash
diamond-response reactor config/diamond_in_reactor.yaml --out runs/diamond_in_reactor   # [--threads N] [--rerun-field]
```

Geometry: the diamond centre is `distance_from_surface_cm` (5 cm) outside the normalisation surface
(`outer_cowling`, r = 7 cm), i.e. at x = 12 cm, y = 0, z = 0 (middle of the cathode region); the face normal points
at the reactor axis. The run is resumable (the field is saved in `reactor_field.npz`, matrix columns per class folder).

Stages: (1) transport the reactor's custom D-T source and tally, on an axisymmetric ring through the detector
position, the neutron fluence spectrum (n/cm²/s, normalised to the configured surface flux), plus the distribution of
incidence angles in a small box; (2) simulate one response matrix per incidence-angle class
(`angle_edges_deg`, simulated at the flux-weighted mean angle of each class); (3) fold:
counts/s = Σ_class R_class · (φ · f_class).

Outputs in the `--out` directory (all CSV with headers):

| file | content |
|---|---|
| `detector_response.csv` | counts/s per deposited bin (10 keV) with uncertainty, relative uncertainty, per-MeV values |
| `detector_response_with_resolution.csv` | same, convolved with the `resolution:` settings |
| `incident_neutron_spectrum.csv` | fluence rate at the detector per 0.1 MeV bin, uncertainty, angle-class fractions |
| `source_birth_spectrum.csv` | emission spectrum of the custom D-T source on the incident grid |
| `plots/` | `reactor_response.png` (counts/s/MeV of the diamond at its reactor position, 1σ band, linear y + log inset), `reactor_efficiency_vs_energy.png` (thresholds all, 0.5, 1, 5, 7.5 MeV, error bars), `reactor_efficiency_vs_threshold.png` (counts/s per unit 1-17 MeV fluence rate vs threshold, labelled at the same thresholds) |
| `npz/` | binary data: `reactor_field.npz`, per-class matrix columns, OpenMC run files (resumable) |
| `summary.json` | totals, counts/s above `roi_min_mev`, source rate, flux-calibration error |

Uncertainties are statistical only (reactor tally and response matrix, bins independent). The flux-calibration
error (common scale factor, in `summary.json`) is not included.

Assumptions: the field is axisymmetric about z (ring averaging); angle classes are coarse (the angle fractions use
the coarse `angle_energy_edges_mev` grid and carry no uncertainty, the diamond is tilted about y); the matrix
assumptions listed above (guessed (n,α) branching, no gammas, placeholder resolution) also apply. Increase
`reactor.particles × batches` to reduce the spectrum noise; `matrix_target_rel_error` controls the matrix noise.

Plot notes: all plots have major/minor gridlines. Efficiency vs energy uses the angle-averaged matrix of the reactor field and is blank above ~15.5 MeV, where the reactor delivers no neutrons. The generic (non-reactor) `run` command still writes the response-lines/heatmap plots.

### Completeness checks (reactor mode)

`summary.json` reports `fluence_rate_below_range` / `above_range` / `fraction_of_fluence_not_simulated`
(neutrons outside the simulated incident-energy grid; a wide tally on the same ring) and
`angle_box_to_ring_flux_ratio` (flux of the 3 cm angle box over the same energies divided by the ring flux; ~1 means the
angular distribution and the spectrum describe the same neutrons). A saved `npz/reactor_field.npz` is reused only if its
energy grid and angle classes match the config; otherwise the reactor stage is re-run. `reactor.angle_edges_deg` must be a
YAML list (the angle tally has 5 degree bins; each class costs one response matrix).

### Unfolding the pulse-height spectrum

`diamond-response reactor ...` finishes by unfolding the simulated detector spectrum (with resolution) back to
incident neutron energy and comparing it with the true reactor fluence. To redo only this step on an existing run:

```bash
diamond-response unfold config/diamond_in_reactor.yaml --out runs/diamond_in_reactor
```

Method (`unfold.py`): MLEM (non-negative Poisson maximum likelihood) on `counts = Poisson(live_time * M @ phi)`,
stopped at the first iteration with Pearson chi2/ndf <= `unfold.stop_chi2_ndf` (early stopping is the regulariser).
Uncertainties are the spread over `unfold.replicas` repetitions with Poisson-resampled counts and the matrix
perturbed by its own MC error. They exclude unfolding bias.
Two matrices are used: the angle-averaged one (incidence distribution known) and the nearest-to-normal class matrix
(angles unknown). Outputs:

- `unfolded_spectrum.csv`: true fluence rate, unfolded rate, uncertainties and unfolded/true per incident bin
- `unfolding_summary.json`: iterations, chi2/ndf, totals and per-band (1-5, 5-10, 10-13, 13-14.4, 14.4-16 MeV) comparison
- `plots/unfolded_vs_true.png` (with ratio panel), `plots/unfold_refold_check.png`

Settings are in the `unfold:` section (`live_time_s` sets the statistics; `min_pulse_mev` discards low pulses).
Caveats: the problem is ill-posed, so neighbouring bins are anti-correlated and fine structure cannot be resolved;
the same matrix generates and unfolds the data (an "inverse crime"), so results are optimistic compared with real data
where matrix errors (e.g. the assumed (n,alpha) branching) add bias; neutrons below 1 MeV are not in the matrix.

## Comparing with measured spectra

```bash
# 1) field at the measurement position (4.7 cm) is simulated on first use (slow), then reused
diamond-response compare config/diamond_compare_4p7.yaml \
  --matrix runs/diamond_dt/npz/response_matrix.npz \
  --data "C:\Users\Romul\Analysis\DATA\15092026_90kV40mA_ChanAspectrum_1440" \
  --out runs/diamond_compare_4p7 \
  --mv-per-mev 95            # asked interactively if omitted; stored axis assumes 100 mV/MeV
  # optional: --iterations N --min-pulse-mev X --noise-kev K
```

`--data` takes folders containing `all_datasets.npz` + `metadata.json` (Windows paths are converted under WSL; several
folders can be given). The matrix must have the same incident grid as the field (0.025 MeV, 1-16 MeV).
Use `runs/diamond_dt/npz/response_matrix.npz`; the copy directly in `runs/diamond_dt/` can be older.

Outputs in the run directory: `compare_deposited_spectra.csv`, `compare_unfolded_spectra.csv`, `compare_summary.json`
(count rates, iterations, chi2/ndf), `plots/compare_deposited.png` (measured vs simulated, ratio panel, log inset) and
`plots/compare_unfolded.png` (unfolded measured, unfolded simulated and the true simulated incident spectrum with ratio
panel). All y-axes are relative flux: each curve has unit area over its comparison range (deposited: `unfold.min_pulse_mev`
to the top of the measured range; incident: 1-16 MeV).

Notes: simulated deposited = field x resolution-smeared normal-incidence matrix (angular distribution ignored; the
default resolution noise 100 keV / 0.8 % is a placeholder, override with `--noise-kev`). The simulated spectrum is scaled to
the measured counts before unfolding so both have the same statistics. The measured spectrum includes everything the
simulation lacks (gammas, <1 MeV neutrons, electronics noise), so chi2/ndf of the measured unfolding will be >> 1 and
the unfolded curve is noisy at 0.025 MeV binning; use `--iterations` and `--min-pulse-mev` to trade resolution for stability.
`response_matrix.png` colour scale spans the 1st-99th percentile of bins (bins below 1e-12 of the maximum, i.e. Gaussian-tail
underflow, are excluded first).

## Flux from measured data (`scripts/flux_analysis.py`)

Freestanding script (numpy, scipy, matplotlib only). It unfolds the measured pulse-height spectrum, derives the
spectrum-weighted efficiency of a pulse-height window, and converts pulse rates to flux at the detector and at the reactor
surface.

```bash
python scripts/flux_analysis.py \
  --data "C:\Users\Romul\Analysis\DATA\15092026_90kV40mA_ChanAspectrum_1440" \
  --matrix runs/diamond_dt/npz/response_matrix.npz \
  --field runs/diamond_compare_4p7/npz/reactor_field.npz \
  --out runs/flux_4p7            # --mv-per-mev X (prompted if omitted), --series roi|total, --threshold-mev T, --bin-s 10
```

- Flux = pulse rate in the window / effective efficiency, `eff = sum_j w_j eps_j(window)` with `w` the unfolded spectral shape.
  The time series uses the acquisition's per-second counters: `roi` (hardware ROI, default) or `total` (above trigger).
  `--threshold-mev` sets the PH threshold of the spectrum-based flux (default: lower ROI edge, upper limit: ADC range).
- Efficiency is computed from the resolution-smeared matrix unless `--efficiency table.csv` is given (first column incident MeV,
  header = thresholds in MeV). The table used is written to `efficiency_matrix.csv`.
- Surface flux = detector flux x k, k = configured surface flux / simulated detector flux (`--field`, or `--surface-scale k`).
  One energy-independent factor: the surface spectral shape is not separately simulated. `reactor_field.npz` files made before
  this feature lack `surface_flux_n_cm2_s`; use `--surface-scale` or re-save the field with that value.
- The unfolding variance comes from `--replicas` fits with Poisson-resampled counts and matrix values varied within their statistical error.
  Incident bins below `min_pulse/0.284` MeV (carbon-recoil kinematic limit) are excluded.
- Outputs: `flux_vs_time.csv`, `unfolded_spectrum.csv`, `efficiency_matrix.csv`, `flux_summary.json`,
  `plots/flux_vs_time.png`, `plots/flux_histogram.png`, `plots/unfolded_spectrum_surface.png`.

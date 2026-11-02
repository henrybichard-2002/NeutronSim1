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
| `plots/` | `response_lines*.png` (2.45, 13.8, 14.1, 14.5 MeV and the unmoderated source, linear y), `efficiency_vs_energy.png`, `efficiency_vs_threshold.png`, `efficiency_heatmap.png` (efficiency vs incident energy vs pulse height) |
| `npz/` | binary data: `reactor_field.npz`, per-class matrix columns, OpenMC run files (resumable) |
| `summary.json` | totals, counts/s above `roi_min_mev`, source rate, flux-calibration error |

Uncertainties are statistical only (reactor tally and response matrix, bins independent). The flux-calibration
error (common scale factor, in `summary.json`) is not included.

Assumptions: the field is axisymmetric about z (ring averaging); angle classes are coarse (the angle fractions use
the coarse `angle_energy_edges_mev` grid and carry no uncertainty, the diamond is tilted about y); the matrix
assumptions listed above (guessed (n,α) branching, no gammas, placeholder resolution) also apply. Increase
`reactor.particles × batches` to reduce the spectrum noise; `matrix_target_rel_error` controls the matrix noise.

Plot notes: `efficiency_vs_energy.png` and `efficiency_vs_threshold.png` carry vertical lines (the 2.45/13.8/14.1/14.5 MeV energies and the 0/0.5/1/5/7.5 MeV thresholds respectively) with the y value labelled where each line meets a curve; `response_custom_source.png` (linear y, log inset) replaces the single-column plot; all plots have major and minor gridlines; line responses are linear in y with a log-y inset; `efficiency_vs_energy.png` shows 1σ error bars for pulses above 0 (all), 0.5, 1, 5 and 7.5 MeV; `efficiency_heatmap.png` has two panels (differential, log colour over 4 decades; and integral efficiency above a pulse-height threshold). line responses interpolate linearly between neighbouring 0.1 MeV columns; "unmoderated source" is the custom-source birth spectrum (angle-averaged emission, sampled in a short source-only run) folded with the angle-averaged matrix. Matrix columns carrying little fluence use a relaxed `target_rel_error` (scaled by 1/√relative fluence, max 30%) and `max_chunks` caps the effort.

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
export PYTHONPATH=src   # or pip install -e .
diamond-response run config/diamond_dt.yaml --out runs/diamond_dt [--threads N] [--columns 20 21]
diamond-response fold runs/diamond_dt/response_matrix.npz --deuteron-mev 0.1 --angle-deg 90 --fwhm-mev 0.2
```
Outputs: `response_matrix.npz/.csv`, `response_matrix_resolution.csv` (Gaussian noise + relative smearing),
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

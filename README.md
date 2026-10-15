# IEC D-T Neutron Source and Moderator Model

This project builds an [OpenMC](https://openmc.org/) fixed-source Monte Carlo
model of an isotropic, cylindrical IEC deuterium-tritium neutron source and
coaxial moderation/shielding layers. Geometry, material composition, run
settings, and energy-bin boundaries are configured in YAML.

The default source is a uniform-volume cylinder emitting monoenergetic 14.1 MeV
neutrons. Its spectrum is an idealized D-T source spectrum; it does not model
the IEC plasma, ion trajectories, or source-specific energy broadening.

## Install

Use Python 3.10 or newer. Install OpenMC and its nuclear data library using the
[official installation guide](https://docs.openmc.org/en/stable/quickinstall.html).
OpenMC must be available on `PATH`, and its nuclear-data library must be
configured (for example, with `OPENMC_CROSS_SECTIONS` pointing to
`cross_sections.xml`).

Then install this project and its test dependency:

```sh
python -m pip install -e ".[test]"
```

## Configure and run

The three preliminary paths are configured in
[`config/reactorpath1.yaml`](config/reactorpath1.yaml),
[`config/reactorpath2.yaml`](config/reactorpath2.yaml), and
[`config/reactorpath3.yaml`](config/reactorpath3.yaml). Each uses the same
uniform-volume, monoenergetic cylindrical source: 1 cm diameter, 35 cm tall,
14.1 MeV. Layers are listed from the source outward as radial cylindrical
shells; `thickness_mm` is the thickness of each shell. In these three configs
every layer is 60 cm tall.

Materials specify either element atom ratios or nuclide atom ratios, along with
a density in g/cm3. Optional thermal-scattering table names are passed through
to OpenMC (`c_H_in_H2O` for water). The preliminary configurations model dry
air using natural N/O/Ar/C elemental proportions, aluminium at 2.70 g/cm3, and
water at 1.0 g/cm3. The selected nuclear data library must include the natural
isotopes for H, C, N, O, Ar and Al, plus the `c_H_in_H2O` thermal-scattering
table.

Build OpenMC XML inputs without starting a simulation:

```sh
iec-neutrons build config/reactorpath1.yaml
```

Run any one path (each preliminary configuration is set to 100,000 particles
per batch and 10 batches):

```sh
iec-neutrons run config/reactorpath1.yaml
iec-neutrons run config/reactorpath2.yaml
iec-neutrons run config/reactorpath3.yaml
```

Both commands accept `--output-dir`; by default files go to
`runs/<config-name>/`. The energy tallies use 0.1 MeV bins from 1 to 16 MeV.
The spatial map tallies use a separate, coarser set of energy groups to keep
plot panels manageable. XY and XZ maps are central-plane slab averages; the
slab thickness is set by `visualization.slice_thickness_cm` (0.4 cm by default).
`visualization.mesh_bins` controls spatial resolution. The source-flux spectrum
is tallied in the configured cylindrical source volume; the remaining source
cavity is tallied separately. Flux results are normalized per source neutron,
not to a physical source emission rate.

OpenMC tallies are normalized per source particle; multiply them by the
physical neutron emission rate when a source-rate-normalized result is needed.
The run creates a statepoint HDF5 file, which can be analyzed with OpenMC's
Python API.

Each config includes a 0.4 × 0.4 cm detector face and 500 µm thickness. Change
`detector.x_cm`, `detector.y_cm`, and `detector.z_cm` to position it, then
rerun the simulation. The detector energy plot is the volume-averaged neutron
flux spectrum in that box (not a detector-response/count-rate simulation).
Detector dimensions can be changed with `width_x_cm`, `width_y_cm`, and
`thickness_z_cm`.

Plot the results from the statepoint:

```sh
iec-neutrons plot config/reactorpath1.yaml runs/reactorpath1/statepoint.10.h5
```

Plots are written to `runs/reactorpath1/plots/` by default. The output contains
labelled top-down and side-on geometry figures, spatial total-flux and
energy-group maps, relative-uncertainty maps, x/y/z line profiles at the
configured `visualization.marginal_*_cm` positions, region spectra, and the
0.1 MeV detector spectrum. The spatial meshes and geometry plots extend across
the full configured world radius in each plotted direction, including the
vacuum region; spatial heatmaps include dotted flux contours. Line profiles use
the nearest mesh-bin center to the requested coordinate and span the world
radius. Run the matching simulation before plotting; plotting checks that
statepoint mesh, detector, and energy bins agree with the supplied config.
Matplotlib is required and can be installed with
`python -m pip install -e ".[plot]"`.

## Tests

The configuration parser and geometry validation can be tested without OpenMC:

```sh
python -m pytest
```

## GitHub

The project includes a GitHub Actions workflow for the Python tests. To connect
this local project to a GitHub repository, add the repository remote and push:

```sh
git remote add origin <repository-url>
git push -u origin HEAD
```

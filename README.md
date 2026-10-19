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
configured through `OPENMC_CROSS_SECTIONS`.

Install this project and the plotting dependency into the same environment as
OpenMC. This matters when both a project `.venv` and the `neutrons-openmc`
Conda environment are active: a `python` or `iec-neutrons` command may resolve
to the virtual environment, where OpenMC or Matplotlib may not be installed.
From the project root, use the OpenMC environment explicitly. For the
environment used in this setup, set these variables once per terminal:

```sh
export OPENMC_ENV=/home/Romulus1/miniforge3/envs/neutrons-openmc
export PYTHONPATH="$PWD/src"
"$OPENMC_ENV/bin/python" -m pip install -e ".[plot,test]"
```

If your OpenMC environment is in a different location, replace the path with
that environment's prefix. The explicit `"$OPENMC_ENV/bin/python"` command
avoids ambiguity when another environment (such as `.venv`) is active.
`PYTHONPATH` points that interpreter at this checkout's `src/` package.

## Cross-section indexes

Each configuration has a corresponding trimmed cross-section index under
[`config/cross_sections/`](config/cross_sections/). These list only the
neutron nuclides and thermal-scattering tables used by that configuration:

| Configuration | Cross-section index |
| --- | --- |
| `reactorpath1.yaml` | `config/cross_sections/reactorpath1.xml` |
| `reactorpath2.yaml` | `config/cross_sections/reactorpath2.xml` |
| `reactorpath3.yaml` | `config/cross_sections/reactorpath3.xml` |
| `example.yaml` | `config/cross_sections/example.xml` |
| `he3_detector.yaml` | `config/cross_sections/he3_detector.xml` |
| `gen_reactor.yaml` | `config/cross_sections/gen_reactor.xml` |

Before a run, set `OPENMC_CROSS_SECTIONS` to the index for that config. For
example:

```sh
OPENMC_CROSS_SECTIONS="$PWD/config/cross_sections/reactorpath1.xml" \
  "$OPENMC_ENV/bin/python" -m iec_neutrons.cli run config/reactorpath1.yaml
OPENMC_CROSS_SECTIONS="$PWD/config/cross_sections/gen_reactor.xml" \
  "$OPENMC_ENV/bin/python" examples/run_custom_source.py config/gen_reactor.yaml
```

The index files use the local data directory
`/home/Romulus1/NeutronsInMatter`. If your nuclear data is elsewhere, update
the `<directory>` element in each index. The `he3_detector` configuration
enables photon transport; its index needs compatible photon interaction
libraries as well as the listed neutron and thermal data. Photon libraries
were not present in the local data index used to create these files, so add
photon entries from your photon library before running that configuration.

## Configure and run

The three preliminary paths are configured in
[`config/reactorpath1.yaml`](config/reactorpath1.yaml),
[`config/reactorpath2.yaml`](config/reactorpath2.yaml), and
[`config/reactorpath3.yaml`](config/reactorpath3.yaml). Each uses the same
uniform-volume, monoenergetic cylindrical source: 1 cm diameter, 35 cm tall,
14.1 MeV. Layers are listed from the source outward as radial cylindrical
shells; `thickness_mm` is the thickness of each shell. In these three configs
every layer is 60 cm tall.

Materials specify either element atom ratios or nuclide atom ratios, along
with a density in g/cm3. Optional thermal-scattering table names are passed
through to OpenMC (for example, `c_H_in_H2O` for water). The configuration-
specific indexes above include the corresponding natural-isotope neutron
tables and thermal-scattering tables, including He-3 for the two detector
configurations.

The standalone `he3_detector.yaml` example also enables photon transport and
therefore needs compatible photon interaction data.

Build OpenMC XML inputs without starting a simulation:

```sh
"$OPENMC_ENV/bin/python" -m iec_neutrons.cli build config/reactorpath1.yaml
```

Run any one path (each preliminary configuration is set to 100,000 particles
per batch and 10 batches):

```sh
OPENMC_CROSS_SECTIONS="$PWD/config/cross_sections/reactorpath1.xml" \
  "$OPENMC_ENV/bin/python" -m iec_neutrons.cli run config/reactorpath1.yaml
OPENMC_CROSS_SECTIONS="$PWD/config/cross_sections/reactorpath2.xml" \
  "$OPENMC_ENV/bin/python" -m iec_neutrons.cli run config/reactorpath2.yaml
OPENMC_CROSS_SECTIONS="$PWD/config/cross_sections/reactorpath3.xml" \
  "$OPENMC_ENV/bin/python" -m iec_neutrons.cli run config/reactorpath3.yaml
```

Both commands accept `--output-dir`; by default files go to
`runs/<config-name>/`. The energy tallies use 0.1 MeV bins from 1 to 16 MeV.
The spatial map tallies use a separate, coarser set of energy groups to keep
plot panels manageable. XY and XZ maps are central-plane slab averages; the
slab thickness is set by `visualization.slice_thickness_cm` (0.4 cm by default).
`visualization.mesh_bins` controls spatial resolution. The source-flux spectrum
is tallied in the configured cylindrical source volume; the remaining source
cavity is tallied separately. XY/XZ flux maps and the detector spectrum divide
OpenMC's track-length tallies by the exact mesh-cell volume to report scalar
flux in cm-2 per source neutron. Region spectra are integrated track-length
tallies in cm per source neutron, so their raw values should not be compared
across regions of different volumes.

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

Plot the results from the statepoint using the same OpenMC Python environment:

```sh
"$OPENMC_ENV/bin/python" -m iec_neutrons.cli plot config/reactorpath1.yaml \
  runs/reactorpath1/statepoint.10.h5
```

Plots are written to `runs/reactorpath1/plots/` by default. The output contains
labelled top-down and side-on geometry figures, spatial total-flux and
energy-group maps, relative-uncertainty maps, x/y/z line profiles at the
configured `visualization.marginal_*_cm` positions, region spectra, and the
0.1 MeV detector spectrum. The spatial meshes and geometry plots extend across
the full configured world radius in each plotted direction, including the
vacuum region; only total-flux heatmaps include dotted, numerically labelled
contour lines, and spatial heatmaps show a magenta detector marker. Geometry
figures also show configured component materials and the detector location.
Line profiles use the nearest mesh-bin center to
the requested coordinate and span the world radius. Plotting checks that the
statepoint mesh fits within the configured world, energy bins match, and any
detector tally matches the supplied config.
Matplotlib is required; install it into the OpenMC environment with the
`.[plot]` extra shown above. Run the plot command with that environment's
Python (rather than a `iec-neutrons` executable that might belong to `.venv`).

The source-volume flux spectrum and detector spectrum require statepoints
generated with the current model tallies. Older statepoints remain plottable
when they contain the map and region tallies: the old source-cavity curve is
labelled as including the source, the detector position is still marked from
the config, and the unavailable detector spectrum is skipped with a warning.
Rerun the simulation to obtain the separated source spectrum and detector
spectrum:

```sh
"$OPENMC_ENV/bin/python" -m iec_neutrons.cli run config/example.yaml
"$OPENMC_ENV/bin/python" -m iec_neutrons.cli \
  plot config/example.yaml runs/example/statepoint.10.h5
```

Plot appearance is controlled in the `PLOT_STYLE` block near the top of
`src/iec_neutrons/visualize.py`. It centralizes spatial, energy-map,
uncertainty, geometry, and spectrum color schemes; spatial/spectrum axis
scales; map color scales; and `xlim`, `ylim`, `zlim`, colorbar limits, and
spectrum limits. The relative-uncertainty maps use a separate `gnuplot2`
colormap by default. Leave limits as `None` to retain the full world-radius
view and automatic color limits.
`zlim` controls the vertical extent on XZ plots; color limits are specified as
`(minimum, maximum)`. Automatic heatmap lower color limits use the configured
`heatmap_min_percentile` (2nd percentile by default), preventing isolated
low-value outliers from compressing the displayed color range.

## External assemblies and interaction tallies

`geometry.components` configures finite components beyond or within the radial
layers. Supported shapes are axis-aligned boxes (`size_cm: [x, y, z]`) and
cylinders (`radius_cm`, `length_cm`, and `axis: x|y|z`), each located by
`center_cm: [x, y, z]`. Components carve their volume out of the reactor
materials by default. For nested assemblies, set `carve_from_base: false` on
inner components that are fully contained inside an outer component which
already carves the reactor material. This keeps redundant internal surfaces
out of every reactor cell while the component cells still replace one another
in list order, with later components taking priority over earlier ones.
Components and the detector scoring volume must fit inside the spherical world
boundary.

The provisional [`config/he3_detector.yaml`](config/he3_detector.yaml) models
reactor path 2 with an external assembly centered 30 cm from the source along
+x. Replaceable assumptions are an 8 × 8 × 40 cm HDPE block, a 30 cm active
tube length, 1.22 cm inner radius, a 0.5 mm aluminium wall, and 4 atm He-3 at
293 K represented by an ideal-gas density of 0.00050 g/cm3. The tube axis is z.
These dimensions and material properties are examples, not measured detector
specifications.

Material-region tallies include neutron total, scattering, absorption, fission,
fission-neutron production, (n,2n), (n,3n), and (n,gamma) scores.
`neutron_energy_transfer_*` tallies bin incoming and outgoing neutron energies
to show scattering energy loss or gain; `secondary_particle_production`
records neutron/photon secondary yields. The He-3 gas also has an energy-binned
(n,p) response tally. Reaction channels and yields are limited by the selected
evaluated nuclear data and do not imply every conceivable reaction is present.

The He-3 example enables `physics.photon_transport` and requests photon energy
deposition tallies. Photon transport requires compatible photon interaction
data in the cross-section index. The example `cross_sections.xml` includes the 293.6 K He-3 neutron table.
Alternate indices must include an equivalent entry (and photon interaction
tables before enabling photon transport). Existing reactor-path
configurations leave photon transport disabled.

`visualization.voxel_bins` sets the geometry voxel resolution in each direction
(64 × 64 × 64 by default). Plotting produces `geometry_voxels.png`, a
material-coloured 3D voxel view of the source region, radial layers, and
external components; vacuum remains transparent. Its bounds focus on the
configured physical geometry rather than the full vacuum world sphere.
Voxelization is a visualization approximation: features thinner than a voxel
may be omitted or assigned to an adjacent material. Rendering cost grows
cubically with this setting, so increase it gradually. XY and XZ transport
maps remain separate central-plane views.

The `gen_reactor` voxel image removes the +x/−y quarter of the geometry and
views the model from that side, exposing the cylindrical layers, reactor end
caps, and the separate detector assembly. The detector is centered at
x = −20 cm. Its outside dimensions are 25.4 mm diameter × 310 mm tall, with a
1 mm aluminium wall and end caps enclosing a 23.4 mm diameter × 308 mm He-3
gas volume. A 3 mm lead layer and 12 cm HDPE layer surround the detector; both
extend axially by their respective thicknesses at each end. The He-3 density is
the ideal-gas value
for 4 bar absolute at 50 °C. The configured 1300 V bias and nominal
50 cps/(n/cm²/s) calibration are detector metadata only; OpenMC reports the
energy-binned He-3 (n,p) response tally and does not model charge collection or
apply that calibration. The two 38 mm radius × 14 mm thick aluminium end caps
are centered immediately above and below the 70 cm cavity.

## Compiled IEC fusion source

The default source remains the simple 14.1 MeV cylindrical source. To use the
separate C++ fusion source instead, run (inside the same environment that has
OpenMC installed):

```sh
"$OPENMC_ENV/bin/python" examples/run_custom_source.py \
  config/reactorpath1.yaml --plot
```

The `gen_reactor` configuration is a ready-to-run custom-source example:

```sh
"$OPENMC_ENV/bin/python" examples/run_custom_source.py \
  config/gen_reactor.yaml --plot
```

It models an anode/inner-cavity radius of 38 mm (76 mm cavity diameter),
14 mm aluminium, 15 mm water, 3 mm aluminium, and a 500 mm radial air layer.
The cavity and first three layers are 70 cm tall; the air cylinder is 100 cm
tall. Its 35 cm source includes D-T and D-D reactions, with beam-target,
beam-wall (anode/inner cavity wall), and beam-beam zones all enabled. The wall
sample is placed just inside the anode surface to avoid starting a particle on
a material boundary; it retains the plugin's applied-voltage beam energy and
is an idealized wall-impact population, not a self-consistent ion trajectory.
The three zone weights are equal starting weights, not
measured or calculated physical reaction-rate fractions; set them from
device-specific data for a predictive yield model. The 50% tritium fraction
permits both D-T and D-D channels in the configured source.

The runner prints a post-run summary with track-length-weighted mean neutron
energies and highest occupied energy-bin upper edges in each region, a
world-boundary outward-current estimate, secondary neutron/photon production,
and per-material reaction tallies. These are source-normalized Monte Carlo
estimates, not exact integer particle counts. The energy summaries use 0.1 MeV
bins from 0 to 20 MeV; the reported maximum is therefore a bin-edge estimate,
not an event-by-event maximum. Per-material scattering energy loss is estimated
from the incoming/outgoing energy-transfer tally using bin midpoints. The
escaped-minus-primary comparison indicates net leakage relative to the source
histories; it is not by itself a complete neutron conservation balance because
absorption and secondary production must also be considered. The report's
interaction scores, energy transfer, and secondary production help inspect
those processes.

The runner compiles [`custom_source.cpp`](src/iec_neutrons/custom_source.cpp)
against the active environment's OpenMC headers and `libopenmc.so`, substitutes
an `openmc.CompiledSource` for the model's default source, and starts the
fixed-source run. `--plot` creates the plots after the run using Matplotlib in
that same Python environment. By default, all files for this custom-source run
are kept together in `runs/<config-name>-custom-source/` (for example,
`runs/reactorpath2-custom-source/`), separate from the standard run at
`runs/reactorpath2/`. The statepoint and generated plots are in that folder.
Set `OPENMC_PREFIX` if the OpenMC headers and shared library are not under the
active Python environment. `--source-cpp`, `--compiler`, `--output-dir`, and
`--parameters` can override their defaults.

To plot an already completed custom-source run without rerunning it, pass the
actual statepoint path printed by OpenMC. The run made before this directory
change is in `runs/custom_source/`:

```sh
"$OPENMC_ENV/bin/python" -m iec_neutrons.cli plot \
  config/reactorpath2.yaml runs/custom_source/statepoint.10.h5
```

For a fresh run and plots in one command, use the `--plot` example above.

The C++ source uses the OpenMC 0.16 compiled-source ABI (`openmc_create_source`)
and implements the Bosch-Hale total cross sections for T(d,n)4He,
D(d,n)3He, and D(d,p)T. D-D events are selected using the sum of both branch
cross sections, and proton-branch events are rejected because this source emits
only neutrons. The `custom_source` YAML block in `gen_reactor.yaml` sets the
plugin's reaction-zone weights, potential, neutral temperature, isotope
fraction, and radial neutral-density profile. An explicit `--parameters`
argument overrides those values. Two-body product energies and directions are calculated from
relativistic four-momentum conservation; the outgoing neutron is isotropic in
the center-of-momentum frame. Bosch-Hale total cross sections do not provide
differential angular distributions, so this isotropic angular assumption is
explicit and should not be mistaken for an evaluated angular model.

The C++ defaults are illustrative, not a measured IEC plasma model: a 4.05 cm
anode radius, 0.73 cm cathode radius, 35 cm source height, 90 keV accelerating
voltage, 300 K neutral gas, and 50% tritium in the target. Beam-target events
are selected by rejection sampling with the annular-cylinder volume Jacobian,
the configured neutral radial-density profile, Bosch-Hale cross section, and
relativistic Moller flux factor. The profile is linearly interpolated in
`radius_fraction:density` points, normalized radius `r/r_anode`; it must span
0 to 1. The gas is axially uniform and has a truncated Maxwell velocity
distribution (12 thermal standard deviations, whose omitted tail probability
is negligible). The user-supplied `zone_beam_target`, `zone_beam_wall`, and
`zone_beam_beam` parameters are relative probabilities for selecting the
respective emitted-neutron source populations. The beam-wall and beam-beam
populations are disabled by default because their rates require device-specific
wall-impact probabilities and beam-density/current distributions that cannot be
inferred from geometry alone.

Parameters are semicolon-separated `key=value` pairs. For example:

```sh
python examples/run_custom_source.py config/reactorpath1.yaml \
  --parameters 'r_anode_cm=4.05;r_cathode_cm=0.73;half_height_cm=17.5;voltage_kev=90;neutral_temperature_k=300;tritium_fraction=0.8;zone_beam_target=1;zone_beam_wall=0;zone_beam_beam=0;source_strength=1;neutral_profile=0:1,0.5:0.8,1:0.4'
```

`tritium_fraction` is the local target tritium fraction (and the beam isotope
fraction for beam-beam pairs); `neutral_profile` is a nonnegative relative
density profile. `source_strength` is the OpenMC source strength. Because
OpenMC source tallies are normalized per source particle, this does not by
itself convert tallies to a physical rate; scale results by the measured neutron
emission rate as appropriate.

The radial expression used for the local beam energy is
`V_applied * ln(r/r_anode) / ln(r_cathode/r_anode)`, matching the requested
concentric-cylinder profile and giving 0 at the outer anode and full voltage
magnitude at the cathode. It is the charge-free cylindrical Laplace solution,
not a Poisson solution including plasma space charge; a signed negative
cathode potential is implicit for positive deuterons moving inward. Space
charge, nonuniform beam distributions, grid-impact probabilities, and
species-resolved measured densities must be supplied or modeled separately
before the resulting source can be called a predictive IEC yield model.

Masses in the C++ implementation are bare nuclear masses, with each reaction
Q-value derived from the same mass set (the D-T, D-D neutron, and D-D proton
Q-values are about 17.58925, 3.26886, and 4.03266 MeV, respectively). The mass
values are based on evaluated atomic masses with electron masses and binding
energies removed consistently (AME2020; Wang et al., *Chinese Physics C* 45
(2021), 030003, DOI: [10.1088/1674-1137/abddaf](https://doi.org/10.1088/1674-1137/abddaf)).
The Bosch-Hale fit is from Bosch and Hale, *Nuclear Fusion* 32 (1992), 611-631,
DOI: [10.1088/0029-5515/32/4/I07](https://doi.org/10.1088/0029-5515/32/4/I07).

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

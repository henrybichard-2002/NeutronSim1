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
export PATH="$OPENMC_ENV/bin:$PATH"
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
| `genDD_reactor.yaml` | `config/cross_sections/genDD_reactor.xml` |
| `genDT_borated_jacket.yaml` | `config/cross_sections/genDT_borated_jacket.xml` |
| `genDT_dual_borated_jacket.yaml` | `config/cross_sections/genDT_dual_borated_jacket.xml` |

Each index has the same basename as its YAML configuration. Set
`OPENMC_CROSS_SECTIONS` to the matching index before each run, for example:

```sh
export OPENMC_CROSS_SECTIONS="$PWD/config/cross_sections/reactorpath1.xml"
```

The index files use the local data directory
`/home/Romulus1/NeutronsInMatter`. If your nuclear data is elsewhere, update
the `<directory>` element in each index. The `he3_detector` configuration
enables photon transport; its index needs compatible photon interaction
libraries as well as the listed neutron and thermal data. Photon libraries
were not present in the local data index used to create these files, so add
photon entries from your photon library before running that configuration.

## Configuration basics

The three preliminary paths are configured in
[`config/reactorpath1.yaml`](config/reactorpath1.yaml),
[`config/reactorpath2.yaml`](config/reactorpath2.yaml), and
[`config/reactorpath3.yaml`](config/reactorpath3.yaml). Layers are listed from
the source outward as radial cylindrical shells; `thickness_mm` is the
thickness of each shell. Each YAML file defines its own source, layer heights,
energy bins, and run settings.

Materials specify either element atom ratios or nuclide atom ratios, along
with a density in g/cm3. Optional thermal-scattering table names are passed
through to OpenMC (for example, `c_H_in_H2O` for water). The configuration-
specific indexes above include the corresponding natural-isotope neutron
tables and thermal-scattering tables, including He-3 for the two detector
configurations.

Build OpenMC XML inputs without starting a simulation:

```sh
"$OPENMC_ENV/bin/python" -m iec_neutrons.cli build config/reactorpath1.yaml
```

## Run and plot a simulation

The workflow is the same for every configuration: choose its YAML file, use
the cross-section index with the same basename, run it, then plot the
statepoint path printed by the run. Keep `OPENMC_ENV` and `PYTHONPATH` set as
shown in [Install](#install).

For example, to run the standard cylindrical source:

```sh
CONFIG=config/reactorpath1.yaml
CONFIG_STEM="$(basename "$CONFIG" .yaml)"
export OPENMC_CROSS_SECTIONS="$PWD/config/cross_sections/${CONFIG_STEM}.xml"
"$OPENMC_ENV/bin/python" -m iec_neutrons.cli run "$CONFIG"
```

For the pure-deuterium D-D compiled source, use the custom-source runner. The
`genDD_reactor.yaml` configuration includes its source parameters:

```sh
CONFIG=config/genDD_reactor.yaml
CONFIG_STEM="$(basename "$CONFIG" .yaml)"
export OPENMC_CROSS_SECTIONS="$PWD/config/cross_sections/${CONFIG_STEM}.xml"
"$OPENMC_ENV/bin/python" examples/run_custom_source.py "$CONFIG"
```

The standard runner always uses the YAML `source` block. The custom-source
runner uses the YAML `custom_source` block when present; otherwise it uses its
documented defaults, which can be overridden with `--parameters`.
The standard runner is `iec_neutrons.cli run`; the compiled-source runner is
`examples/run_custom_source.py`. Both print the statepoint path and support
`--output-dir`. Standard runs default to `runs/<config-name>/`; compiled-source
runs default to `runs/<config-name>-custom-source/`.

Plot either kind of run with the same command, using the YAML configuration
and the exact statepoint path printed by the run:

```sh
CONFIG=config/genDD_reactor.yaml
STATEPOINT=runs/genDD_reactor-custom-source/statepoint.100.h5
"$OPENMC_ENV/bin/python" -m iec_neutrons.cli plot \
  "$CONFIG" "$STATEPOINT"
```

Set `CONFIG` to the same YAML used for the run. The `STATEPOINT` value is an
example: replace it with the exact path printed by OpenMC. For a standard run,
use that run's statepoint path. Batch numbers depend on the configuration.
Plots go to `<statepoint-directory>/plots/` by default. The plot command also
writes `he3_detector_spectra.csv` there when the configuration contains He-3
detectors. To change the plot directory, pass `--output-dir`.

The run itself writes a statepoint HDF5 file and, for He-3 configurations,
`he3_detector_spectra.csv` in the run output directory. Use the same
configuration used for the run when plotting. The statepoint must contain the
tallies expected by the current plotter; if it does not, rerun with the
current model before plotting.

Both run commands accept `--output-dir`. Energy bins and spatial resolution
are set by each YAML file. XY and XZ maps are central-plane slab averages; the
slab thickness is `visualization.slice_thickness_cm` (0.4 cm by default), and
`visualization.mesh_bins` controls spatial resolution. The source-flux spectrum
is tallied in the configured cylindrical source volume; the remaining source
cavity is tallied separately. XY/XZ flux maps divide track-length tallies by
the exact mesh-cell volume to report scalar flux in cm-2 per source neutron.
Region spectra are integrated track-length tallies in cm per source neutron,
so raw values should not be compared across regions of different volumes.

OpenMC tallies are normalized per source particle; multiply them by the
physical neutron emission rate when a source-rate-normalized result is needed.
The run creates a statepoint HDF5 file, which can be analyzed with OpenMC's
Python API.

Each config may also define a rectangular detector scoring volume (0.4 × 0.4 cm
by 500 µm by default). Change `detector.x_cm`, `detector.y_cm`, and
`detector.z_cm` to position it, or set `width_x_cm`, `width_y_cm`, and
`thickness_z_cm` to change its dimensions. This tally reports volume-averaged
neutron flux, not detector response or count rate.

Plots are written to `<statepoint-directory>/plots/` by default. They include
top-down and side-on geometry figures, spatial total-flux and energy-group
maps, relative-uncertainty maps, line profiles at the configured
`visualization.marginal_*_cm` positions, region spectra, and the configured
detector spectra. He-3 configurations additionally produce a combined
flux/capture spectrum on a logarithmic y scale and a position-resolved capture
heatmap. Spatial maps span the configured world radius,
including vacuum; total-flux heatmaps include dotted, numerically labelled
contours. Geometry figures show configured materials and components. Line
profiles use the nearest mesh-bin center to the requested coordinate.
Matplotlib must be installed in the same OpenMC environment; the `.[plot]`
extra above installs it.

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
Set `material: void` for a true vacuum/void component. Set
`score_spectrum: true` on a component to add a fine-energy neutron-flux tally
for that volume. `geometry.background_material` optionally fills otherwise
unoccupied space inside the spherical world boundary; it is omitted by default,
so the external region remains vacuum. `detector.enabled: false` suppresses
the generic rectangular detector tally and marker.
Components and the detector scoring volume must fit inside the spherical world
boundary.

## Slotted borated-polyethylene jackets

[`config/genDT_borated_jacket.yaml`](config/genDT_borated_jacket.yaml) and
[`config/genDT_dual_borated_jacket.yaml`](config/genDT_dual_borated_jacket.yaml)
use the D-T custom source and the same 70 cm tall reactor vessel as
`genDT_reactor`: 14 mm aluminium, 15 mm water, and 3 mm aluminium around the
3.8 cm-radius cavity, including aluminium end caps. Each jacket is 90 cm tall
(10 cm beyond the vessel at each end), with 15 cm of HDPE, then 5 cm of
5 wt% natural-boron polyethylene, followed by 3 mm of lead. Both polymer
materials use a density of 0.94 g/cm3; the BPE recipe is 5 wt% natural boron
and 95 wt% polyethylene, represented by normalized atom fractions in YAML.
The 14 cm-wide opening extends radially from the jacket's outer edge to the
reactor axis and spans the jacket's full height. Its cut is restricted to the
jacket materials: the cavity, vessel, and other reactor volume remain intact.
Consequently, a neutron traveling on the +x axis passes through no jacket
material, while still traversing the reactor's own vessel layers. The single
jacket opens toward +x. The dual configuration places reactor centers 100 cm
apart and points the openings toward each other; each opening reaches its
reactor's axis.
the compiled source samples the two identical reactor centers with equal
probability. The four spectrum scoring cylinders remain fixed relative to
reactor 1: at 50 cm radius in the gap direction, opposite the gap, and at
either side of the gap-aligned position, each offset by 5 cm in y. Their active
dimensions match the previous He-3 gas volume (2.34 cm diameter by 30.8 cm
high), but these are virtual air-filled scoring regions, not He-3 detectors.
Plotting writes `survey_region_neutron_spectra.png`: the main plot shows 1–16
MeV on a linear energy axis, with a sub-MeV inset using a logarithmic energy
axis to expose the lower-energy spectrum.

Run either configuration with its matching cross-section index and the
compiled custom-source runner:

```sh
CONFIG=config/genDT_borated_jacket.yaml
CONFIG_STEM="$(basename "$CONFIG" .yaml)"
export OPENMC_CROSS_SECTIONS="$PWD/config/cross_sections/${CONFIG_STEM}.xml"
"$OPENMC_ENV/bin/python" examples/run_custom_source.py "$CONFIG"
```

Change `CONFIG` to `config/genDT_dual_borated_jacket.yaml` for the pair.
Plot either run with the normal plotting command and the statepoint path
printed by the run:

```sh
"$OPENMC_ENV/bin/python" -m iec_neutrons.cli plot \
  "$CONFIG" runs/"${CONFIG_STEM}-custom-source"/statepoint.100.h5
```

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
records neutron/photon secondary yields. The He-3 gas has energy-binned
neutron-flux and (n,p) response tallies. Plotting a statepoint from a He-3
configuration writes `he3_detector_spectrum_<component>_log.png`, which
overlays the volume-averaged neutron spectrum and capture response on separate
y-axes, with a log-x inset from 1 eV to 0.1 MeV. It also writes
`he3_capture_response_<component>_height_energy.png`: energy is logarithmic,
height is linear and measured relative to the center of the active gas, and
color is logarithmic capture-response density per source neutron, MeV, and cm.
The heatmap uses 40 equal axial bins by default; set
`visualization.he3_axial_bins` to change the height resolution. Fine logarithmic
energy bins span 1e-5 eV to 100 keV (24 bins per decade), with coarser bins
above 100 keV. This resolves thermal capture as well as the fast source
spectrum. Zero-response bins are left blank on the logarithmic color scale.
The 0–1e-5 eV tally bin is omitted because it cannot be drawn on a logarithmic
energy axis.
Rerun the simulation to add the axial-response tally to an older statepoint;
plotting older files still produces the log spectrum and reports the missing
heatmap tally. The capture response is
not a pulse-height spectrum: the model does not simulate charge collection or
infer neutron energy from the tube's electrical signal.
Reaction channels and yields are limited by the selected evaluated nuclear
data and do not imply every conceivable reaction is present.

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

The `genDD_reactor` voxel image removes the +x/−y quarter of the geometry and
views the model from that side, exposing the cylindrical layers, reactor end
caps, and detector assemblies. Four vertical detectors are centered 25 cm from
the reactor axis at +x, +y, −x, and −y, in that order. Each has an outside
diameter of 25.4 mm and an active height of 310 mm, with a 1 mm aluminium wall
and end caps enclosing a 23.4 mm diameter × 308 mm He-3 gas volume. Each is
surrounded by 3 mm of lead and by an HDPE cylinder whose thickness is 2, 3, 4,
or 5 cm respectively; the lead and HDPE cylinders have the same 31.6 cm
height as the detector assembly, so the shielding thicknesses are radial. The
He-3 density is the ideal-gas value for 4 bar absolute at 50 °C. The configured
1300 V bias and nominal
50 cps/(n/cm²/s) calibration are detector metadata only; OpenMC reports the
energy-binned He-3 (n,p) response tally and does not model charge collection or
apply that calibration. The two 38 mm radius × 14 mm thick aluminium end caps
are centered immediately above and below the 70 cm cavity.

## Compiled IEC fusion source

The standard `iec_neutrons.cli run` command uses the YAML `source` block,
normally the simple cylindrical monoenergetic source. To use the compiled C++
source, use the custom-source runner shown in the run workflow above.
`genDD_reactor.yaml` is the ready-to-run pure-deuterium D-D configuration. Both
run paths produce a statepoint; plot them with the same command and the
statepoint path printed at the end of either run.

It models an anode/inner-cavity radius of 38 mm (76 mm cavity diameter),
14 mm aluminium, 15 mm water, 3 mm aluminium, and a 500 mm radial air layer.
The cavity and first three layers are 70 cm tall; the air cylinder is 100 cm
tall. Its 35 cm source includes D-D reactions only, with beam-target,
beam-wall (anode/inner cavity wall), and beam-beam zones all enabled. The wall
sample is placed just inside the anode surface to avoid starting a particle on
a material boundary; it retains the plugin's applied-voltage beam energy and
is an idealized wall-impact population, not a self-consistent ion trajectory.
The three zone weights are equal starting weights, not
measured or calculated physical reaction-rate fractions; set them from
device-specific data for a predictive yield model. The configured tritium
fraction is zero, so all source zones use deuterium and produce D-D events.

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

Runs create `he3_detector_spectra.csv` in the run output directory when the
configuration includes He-3 gas. It contains one row per energy bin and
detector, with bin bounds, volume-averaged neutron flux, He-3 (n,p) capture
response, and their tally standard deviations. Values are source-normalized
per MeV; neutron flux is additionally divided by the active He-3 volume. When
plotting an existing statepoint with `iec-neutrons plot`, the CSV is written
beside the plots.

The custom-source runner compiles
[`custom_source.cpp`](src/iec_neutrons/custom_source.cpp) against the active
environment's OpenMC headers and `libopenmc.so`, assigns an
`openmc.CompiledSource`, and starts the fixed-source run. It writes output to
`runs/<config-name>-custom-source/` by default, separate from the standard
`runs/<config-name>/` directory. Set `OPENMC_PREFIX` if the OpenMC headers and
shared library are not under the active Python environment. `--source-cpp`,
`--compiler`, `--output-dir`, and `--parameters` are available to override
defaults. Its optional `--plot` flag is a convenience; the two-step run then
plot workflow above is the same for standard and compiled-source statepoints.

The C++ source uses the OpenMC 0.16 compiled-source ABI (`openmc_create_source`)
and implements the Bosch-Hale total cross sections for T(d,n)4He,
D(d,n)3He, and D(d,p)T. D-D events are selected using the sum of both branch
cross sections, and proton-branch events are rejected because this source emits
only neutrons. The `custom_source` YAML block in `genDD_reactor.yaml` sets the
plugin's reaction-zone weights, potential, neutral temperature, isotope
fraction, and radial neutral-density profile. An explicit `--parameters`
argument overrides those values. Two-body product energies and directions are calculated from
relativistic four-momentum conservation; the outgoing neutron is isotropic in
the center-of-momentum frame. Bosch-Hale total cross sections do not provide
differential angular distributions, so this isotropic angular assumption is
explicit and should not be mistaken for an evaluated angular model.

The rejection sampler uses separate maximum cross sections for D-T and D-D
events, so the higher D-T cross section does not unnecessarily suppress
sampling efficiency in a pure-deuterium run.

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
"$OPENMC_ENV/bin/python" examples/run_custom_source.py config/reactorpath1.yaml \
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
"$OPENMC_ENV/bin/python" -m pytest
```

## GitHub

The project includes a GitHub Actions workflow for the Python tests. To connect
this local project to a GitHub repository, add the repository remote and push:

```sh
git remote add origin <repository-url>
git push -u origin HEAD
```

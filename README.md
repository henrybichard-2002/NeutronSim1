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
table. `config/he3_detector.yaml` also needs a helium-3 neutron table and
compatible photon interaction data.

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
vacuum region; only total-flux heatmaps include dotted, numerically labelled
contour lines, and spatial heatmaps show a magenta detector marker. Geometry
figures also show configured component materials and the detector location.
Line profiles use the nearest mesh-bin center to
the requested coordinate and span the world radius. Plotting checks that the
statepoint mesh fits within the configured world, energy bins match, and any
detector tally matches the supplied config.
Matplotlib is required and can be installed with
`python -m pip install -e ".[plot]"`.

The source-volume flux spectrum and detector spectrum require statepoints
generated with the current model tallies. Older statepoints remain plottable
when they contain the map and region tallies: the old source-cavity curve is
labelled as including the source, the detector position is still marked from
the config, and the unavailable detector spectrum is skipped with a warning.
Rerun the simulation to obtain the separated source spectrum and detector
spectrum:

```sh
iec-neutrons run config/example.yaml
iec-neutrons plot config/example.yaml runs/example/statepoint.10.h5
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
`center_cm: [x, y, z]`. The listed order defines overlap priority: later
components carve their volume out of earlier components, while the radial
layers take precedence over all components. Components and the detector scoring
volume must fit inside the spherical world boundary.

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
data in the cross-section index. The locally inspected
`runs/example/cross_sections.xml` currently has neither photon interaction
tables nor a He-3 neutron table, so the example can be built but cannot be run
until those libraries are added and indexed. Existing reactor-path configurations
leave photon transport disabled.

`visualization.voxel_bins` sets the geometry voxel resolution in each direction
(64 × 64 × 64 by default). Plotting produces `geometry_voxels.png`, a
material-coloured 3D voxel view of the source region, radial layers, and
external components; vacuum remains transparent. Its bounds focus on the
configured physical geometry rather than the full vacuum world sphere.
Voxelization is a visualization approximation: features thinner than a voxel
may be omitted or assigned to an adjacent material. Rendering cost grows
cubically with this setting, so increase it gradually. XY and XZ transport
maps remain separate central-plane views.

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

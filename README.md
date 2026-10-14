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

Start with [`config/example.yaml`](config/example.yaml). The source is sampled
uniformly by volume inside its radius and height. The cavity is vacuum. Layers
are listed from inner to outer; each cylindrical layer must have a larger
radius and at least the height of the preceding cylinder. Where an outer layer
extends beyond the preceding cylinder's height, it forms an end cap. The
configured spherical vacuum boundary must enclose the whole system.

Materials specify either element atom ratios or nuclide atom ratios, along with
a density in g/cm3. Optional thermal-scattering table names are passed through
to OpenMC (for example, `c_H_in_CH2` for polyethylene). Element and nuclide
names and thermal-scattering tables must be present in the selected nuclear
data library.

Build OpenMC XML inputs without starting a simulation:

```sh
iec-neutrons build config/example.yaml
```

Run the fixed-source simulation:

```sh
iec-neutrons run config/example.yaml
```

Both commands accept `--output-dir`; by default files go to
`runs/example/`. The model scores an energy-binned flux spectrum in the source
cavity and each moderator layer. OpenMC tallies are normalized per source
particle; multiply them by the physical neutron emission rate when a
source-rate-normalized result is needed. The run creates a statepoint HDF5 file,
which can be analyzed with OpenMC's Python API.

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

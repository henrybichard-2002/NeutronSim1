from pathlib import Path

import pytest

from iec_neutrons.config import load_config
from iec_neutrons.model import build_model


CONFIG_DIR = Path(__file__).parents[1] / "config"


@pytest.mark.parametrize(
    "filename", ["reactorpath1.yaml", "reactorpath2.yaml", "reactorpath3.yaml"]
)
def test_source_flux_tally_and_spatial_mesh_cover_world(filename: str) -> None:
    openmc = pytest.importorskip("openmc")
    config = load_config(CONFIG_DIR / filename)
    model = build_model(config)
    tallies = {tally.name: tally for tally in model.tallies}

    source_cell = next(
        cell
        for cell in model.geometry.get_all_cells().values()
        if cell.name == "source volume"
    )
    source_tally = tallies["flux_spectrum_source"]
    source_cell_filter = next(
        item for item in source_tally.filters if isinstance(item, openmc.CellFilter)
    )
    assert source_cell.id in source_cell_filter.bins

    for view in ("xy", "xz"):
        flux_tally = tallies[f"flux_map_{view}"]
        mesh_filter = next(
            item for item in flux_tally.filters if isinstance(item, openmc.MeshFilter)
        )
        radius = config.geometry.world_radius_cm
        if view == "xy":
            assert tuple(mesh_filter.mesh.lower_left) == pytest.approx(
                (-radius, -radius, -config.visualization.slice_thickness_cm / 2)
            )
            assert tuple(mesh_filter.mesh.upper_right) == pytest.approx(
                (radius, radius, config.visualization.slice_thickness_cm / 2)
            )
        else:
            assert tuple(mesh_filter.mesh.lower_left) == pytest.approx(
                (-radius, -config.visualization.slice_thickness_cm / 2, -radius)
            )
            assert tuple(mesh_filter.mesh.upper_right) == pytest.approx(
                (radius, config.visualization.slice_thickness_cm / 2, radius)
            )

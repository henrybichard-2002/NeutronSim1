from pathlib import Path

import pytest

from iec_neutrons.config import load_config
from iec_neutrons.visualize import (
    PLOT_STYLE,
    _geometry_voxel_data,
    _plot_geometry,
    _plot_geometry_voxels,
    _plot_he3_detector_spectrum,
    _plot_spatial_map,
)


CONFIG = Path(__file__).parents[1] / "config" / "reactorpath1.yaml"


def test_spatial_heatmap_uses_world_extent_and_dotted_contours(
    tmp_path: Path, monkeypatch
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    from matplotlib.colors import Normalize
    from matplotlib.patches import Circle, Rectangle

    config = load_config(CONFIG)
    world_radius = config.geometry.world_radius_cm
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    monkeypatch.setattr(plt, "close", lambda figure: None)
    data = np.geomspace(1.0, 100.0, 100).reshape((10, 10))
    _plot_spatial_map(
        data,
        (-world_radius, world_radius, -world_radius, world_radius),
        "test map",
        "flux",
        tmp_path / "map.png",
        config,
        "xy",
        LogNorm,
        Circle,
        Rectangle,
        plt,
        PLOT_STYLE,
        Normalize,
        show_contours=True,
    )

    figure, axis = figures[0]
    assert axis.get_xlim() == pytest.approx((-world_radius, world_radius))
    assert axis.get_ylim() == pytest.approx((-world_radius, world_radius))
    assert axis.collections
    assert any("e" in label.get_text() for label in axis.texts)
    assert any("Detector" in label.get_text() for label in axis.texts)
    figure.clear()


def test_spatial_heatmap_lower_scale_ignores_low_outliers(tmp_path: Path, monkeypatch) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Circle, Rectangle

    config = load_config(CONFIG)
    world_radius = config.geometry.world_radius_cm
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    monkeypatch.setattr(plt, "close", lambda figure: None)
    data = np.geomspace(1.0, 100.0, 100).reshape((10, 10))
    data[0, 0] = 1e-12
    _plot_spatial_map(
        data,
        (-world_radius, world_radius, -world_radius, world_radius),
        "outlier test",
        "flux",
        tmp_path / "outlier.png",
        config,
        "xy",
        LogNorm,
        Circle,
        Rectangle,
        plt,
        PLOT_STYLE,
        Normalize,
        show_contours=False,
    )

    image = figures[0][1].images[0]
    expected_minimum = np.percentile(data[data > 0], PLOT_STYLE.heatmap_min_percentile)
    assert image.norm.vmin == pytest.approx(expected_minimum)
    assert image.norm.vmin > data.min()
    figures[0][0].clear()


def test_uncertainty_heatmap_uses_its_own_colormap(tmp_path: Path, monkeypatch) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib.patches import Circle, Rectangle

    config = load_config(CONFIG)
    radius = config.geometry.world_radius_cm
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    monkeypatch.setattr(plt, "close", lambda figure: None)
    _plot_spatial_map(
        np.arange(1, 101, dtype=float).reshape((10, 10)),
        (-radius, radius, -radius, radius),
        "uncertainty",
        "relative uncertainty",
        tmp_path / "uncertainty.png",
        config,
        "xy",
        None,
        Circle,
        Rectangle,
        plt,
        PLOT_STYLE,
        Normalize,
        show_contours=False,
    )

    assert figures[0][1].images[0].get_cmap().name == "gnuplot2"
    figures[0][0].clear()


@pytest.mark.parametrize("view", ["xy", "xz"])
def test_geometry_plot_covers_world_radius(view: str, tmp_path: Path, monkeypatch) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch, Rectangle, Wedge

    config = load_config(CONFIG)
    world_radius = config.geometry.world_radius_cm
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    monkeypatch.setattr(plt, "close", lambda figure: None)
    _plot_geometry(
        config,
        view,
        tmp_path / f"geometry_{view}.png",
        plt,
        Circle,
        Patch,
        Wedge,
        Rectangle,
        PLOT_STYLE,
    )

    figure, axis = figures[0]
    assert axis.get_xlim() == pytest.approx((-world_radius, world_radius))
    assert axis.get_ylim() == pytest.approx((-world_radius, world_radius))
    assert any("Detector" in label.get_text() for label in axis.texts)
    figure.clear()


def test_geometry_plot_labels_external_components(tmp_path: Path, monkeypatch) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch, Rectangle, Wedge

    config = load_config(Path(__file__).parents[1] / "config" / "he3_detector.yaml")
    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    monkeypatch.setattr(plt, "close", lambda figure: None)
    _plot_geometry(
        config,
        "xy",
        tmp_path / "geometry_xy.png",
        plt,
        Circle,
        Patch,
        Wedge,
        Rectangle,
        PLOT_STYLE,
    )

    figure, axis = figures[0]
    labels = {label.get_text() for label in axis.texts}
    assert "hdpe_moderator\npolyethylene" in labels
    assert "he3_gas\nhelium3_gas" in labels
    assert any("he3_gas: helium3_gas" in label.get_text() for label in axis.get_legend().texts)
    figure.clear()


def test_geometry_voxel_plot_shows_material_voxels(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    config = load_config(Path(__file__).parents[1] / "config" / "he3_detector.yaml")
    labels, materials, bounds = _geometry_voxel_data(config, np)
    assert "D-T source volume" in materials
    assert {"air", "aluminium", "water", "polyethylene", "helium3_gas"} <= set(materials)
    assert np.any(labels >= 0)
    category_ids = {name: index for index, name in enumerate(materials)}
    assert np.any(labels == category_ids["helium3_gas"])
    assert np.any(labels == category_ids["polyethylene"])
    path = _plot_geometry_voxels(
        labels,
        materials,
        bounds,
        config.visualization.voxel_bins,
        tmp_path / "geometry_voxels.png",
        plt,
        np,
        Patch,
        PLOT_STYLE,
        Poly3DCollection,
    )

    assert path.is_file()


def test_he3_spectra_share_linear_and_log_plots_with_low_energy_inset(
    tmp_path: Path, monkeypatch
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    monkeypatch.setattr(plt, "close", lambda figure: None)
    edges = np.concatenate(
        ([0.0], np.geomspace(1.0e-6, 0.1, 31), [1.0, 20.0])
    )
    flux = np.linspace(1.0, 2.0, len(edges) - 1)
    capture_response = np.linspace(1.0e-5, 2.0e-5, len(edges) - 1)
    for yscale in ("linear", "log"):
        path = _plot_he3_detector_spectrum(
            flux,
            capture_response,
            edges,
            "detector_he3_gas",
            yscale,
            tmp_path / f"he3_spectrum_{yscale}.png",
            plt,
            PLOT_STYLE,
        )

        assert path.is_file()
        figure = figures[-1][0]
        axis, capture_axis, inset, inset_capture_axis = figure.axes
        assert axis.get_xlabel() == "Incident neutron energy (MeV)"
        assert "Volume-averaged neutron flux" in axis.get_ylabel()
        assert "He-3 (n,p) reactions" in capture_axis.get_ylabel()
        assert yscale in axis.get_title()
        assert axis.get_yscale() == yscale
        assert capture_axis.get_yscale() == yscale
        assert inset.get_xscale() == "log"
        assert inset.get_xlim() == pytest.approx((1.0e-6, 0.1))
        assert inset.get_yscale() == yscale
        assert inset_capture_axis.get_yscale() == yscale
        figure.clear()


def test_gen_reactor_voxel_geometry_has_cutaway_and_components() -> None:
    pytest.importorskip("matplotlib")
    import numpy as np

    config = load_config(Path(__file__).parents[1] / "config" / "gen_reactor.yaml")
    labels, materials, bounds = _geometry_voxel_data(config, np)
    bins = config.visualization.voxel_bins
    x = np.linspace(bounds[0][0], bounds[0][1], bins, endpoint=False)
    x += (bounds[0][1] - bounds[0][0]) / (2 * bins)
    y = np.linspace(bounds[1][0], bounds[1][1], bins, endpoint=False)
    y += (bounds[1][1] - bounds[1][0]) / (2 * bins)
    cutaway = (x[:, None] > 0) & (y[None, :] < 0)

    assert np.all(labels[cutaway, :] == -1)
    assert {"air", "aluminium", "water", "hdpe", "lead", "helium3_gas"} <= set(materials)
    material_ids = {name: index for index, name in enumerate(materials)}
    assert np.any(labels == material_ids["hdpe"])
    assert np.any(labels == material_ids["lead"])
    assert np.any(labels == material_ids["helium3_gas"])

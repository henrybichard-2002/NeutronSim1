from pathlib import Path

import pytest

from iec_neutrons.config import load_config
from iec_neutrons.visualize import (
    PLOT_STYLE,
    _material_colors,
    _plot_geometry,
    _plot_he3_capture_heatmap,
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
    assert axis.get_xlim() == pytest.approx((-15.0, 15.0))
    assert axis.get_ylim() == pytest.approx((-15.0, 15.0))
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
    expected_x, expected_y = (
        ((-15.0, 15.0), (-15.0, 15.0)) if view == "xy" else ((-30.0, 30.0), (-80.0, 80.0))
    )
    assert axis.get_xlim() == pytest.approx(expected_x)
    assert axis.get_ylim() == pytest.approx(expected_y)
    assert axis.get_aspect() == 1.0
    assert any("Detector" in label.get_text() for label in axis.texts)
    figure.clear()


def test_geometry_plot_uses_material_legend_without_component_labels(
    tmp_path: Path, monkeypatch
) -> None:
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
    colors = _material_colors(config, PLOT_STYLE)
    legend = axis.get_legend()
    legend_labels = [label.get_text() for label in legend.get_texts()]
    legend_handles = legend.legend_handles
    labels = set(legend_labels)
    expected_material_labels = {
        f"Material: {material}" for material in colors
    }
    assert expected_material_labels <= labels
    assert {text.get_text() for text in axis.texts} == {"Detector (y=0 cm)"}
    assert not any("he3_gas" in label for label in labels)
    material_handles = {
        label.removeprefix("Material: "): handle
        for handle, label in zip(legend_handles, legend_labels, strict=True)
        if label.startswith("Material: ")
    }
    for material, handle in material_handles.items():
        assert handle.get_facecolor() == pytest.approx(colors[material])
    component_start = len(config.geometry.layers) + 1
    component_patches = axis.patches[
        component_start : component_start + len(config.geometry.components)
    ]
    for component, patch in zip(
        config.geometry.components, component_patches, strict=True
    ):
        assert patch.get_facecolor()[:3] == pytest.approx(
            colors[component.material][:3]
        )
    figure.clear()


def test_he3_spectrum_log_plot_retains_low_energy_inset(
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
    capture_response[1:-2] = 0.0
    path = _plot_he3_detector_spectrum(
        flux,
        capture_response,
        edges,
        "detector_he3_gas",
        "log",
        tmp_path / "he3_spectrum_log.png",
        plt,
        PLOT_STYLE,
    )

    assert path.is_file()
    figure = figures[-1][0]
    axis, capture_axis, inset_capture_axis = figure.axes
    inset = axis.child_axes[0]
    assert axis.get_xlabel() == "Incident neutron energy (MeV)"
    assert "Volume-averaged neutron flux" in axis.get_ylabel()
    assert "He-3 (n,p) reactions" in capture_axis.get_ylabel()
    assert "log" in axis.get_title()
    assert axis.get_yscale() == "log"
    assert capture_axis.get_yscale() == "log"
    assert inset.get_xscale() == "log"
    assert inset.get_xlim() == pytest.approx((1.0e-6, 0.1))
    assert inset.get_yscale() == "log"
    assert inset_capture_axis.get_yscale() == "log"
    figure.clear()


def test_he3_capture_heatmap_uses_log_energy_and_response_scales(
    tmp_path: Path, monkeypatch
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    monkeypatch.setattr(plt, "close", lambda figure: None)
    energy_edges = np.array(
        [0.0, 1.0e-11, 1.0e-9, 1.0e-7, 1.0e-5, 1.0e-3, 0.1, 1.0, 20.0]
    )
    z_edges = np.linspace(-15.4, 15.4, 41)
    response_density = np.geomspace(
        1.0e-12, 1.0e-5, 40 * (len(energy_edges) - 1)
    ).reshape(40, -1)
    response_density[0, 1:] = 0.0

    path = _plot_he3_capture_heatmap(
        energy_edges,
        z_edges,
        response_density,
        "detector_he3_gas",
        tmp_path / "he3_capture_heatmap.png",
        plt,
        LogNorm,
        PLOT_STYLE,
    )

    assert path.is_file()
    figure = figures[-1]
    axis, colorbar_axis = figure.axes
    assert axis.get_xscale() == "log"
    assert axis.get_xlim() == pytest.approx((1.0e-11, 20.0))
    assert axis.get_ylim() == pytest.approx((-15.4, 15.4))
    assert axis.get_ylabel() == "Height in He-3 gas relative to detector center (cm)"
    assert isinstance(axis.collections[0].norm, LogNorm)
    assert "capture response density" in colorbar_axis.get_ylabel().lower()
    figure.clear()


from pathlib import Path

import pytest

from iec_neutrons.config import load_config
from iec_neutrons.visualize import PLOT_STYLE, _plot_geometry, _plot_spatial_map


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
    )

    figure, axis = figures[0]
    assert axis.get_xlim() == pytest.approx((-world_radius, world_radius))
    assert axis.get_ylim() == pytest.approx((-world_radius, world_radius))
    assert axis.collections
    assert any("e" in label.get_text() for label in axis.texts)
    assert any("Detector" in label.get_text() for label in axis.texts)
    figure.clear()


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

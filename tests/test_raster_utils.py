import pytest
import rasterio
from shapely.geometry import Point
from rasterio.windows import Window

from data_preprocessing.raster_utils import (
    clamp_window,
    find_raster_file_for_geometry,
    get_departement_and_year,
)


@pytest.fixture
def basic_src():
    """Raster IGN de test, 25000 x 25000."""
    with rasterio.open("tests/31-2025-0490-6180-LA93-0M20-E080.jp2") as src:
        yield src


def test_clamp_window_center(basic_src):
    assert clamp_window(
        col=1128,
        row=2128,
        size=256,
        src=basic_src,
    ) == Window(1000, 2000, 256, 256)


def test_clamp_window_edges(basic_src):
    cases = [
        ((50, 128, 256), Window(0, 0, 256, 256)),
        ((24900, 30, 256), Window(24744, 0, 256, 256)),
        ((50, 24800, 500), Window(0, 24500, 500, 500)),
        ((24900, 24999, 500), Window(24500, 24500, 500, 500)),
    ]

    for (col, row, size), expected in cases:
        assert clamp_window(col, row, size, basic_src) == expected


@pytest.mark.parametrize("size", [-10, 0, 30000])
def test_clamp_window_invalid_size_raises_error(basic_src, size):
    with pytest.raises(ValueError):
        clamp_window(col=1000, row=1000, size=size, src=basic_src)


def test_get_departement_and_year(tmp_path):
    raster = tmp_path / "31-2025-0490-6180-LA93-0M20-E080.jp2"
    raster.touch()

    assert get_departement_and_year(tmp_path) == (31, 2025)


def test_find_raster_file_for_geometry(tmp_path):
    # representative point: x=492100 m, y=6178100 m
    # x -> 490 km (multiple inférieur de 5)
    # y: int(6178.1)+1 = 6179 -> 6180 km (multiple supérieur de 5)
    (tmp_path / "31-2025-anything.jp2").touch()

    geometry = Point(492_100, 6_178_100)

    expected = tmp_path / "31-2025-0490-6180-LA93-0M20-E080.jp2"
    assert find_raster_file_for_geometry(geometry, tmp_path) == expected


def test_find_raster_file_for_geometry_near_tile_boundary(tmp_path):
    (tmp_path / "31-2025-anything.jp2").touch()

    geometry = Point(494_999, 6_179_001)

    expected = tmp_path / "31-2025-0490-6180-LA93-0M20-E080.jp2"
    assert find_raster_file_for_geometry(geometry, tmp_path) == expected

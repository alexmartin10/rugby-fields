import random
from pathlib import Path
from unittest.mock import call

from shapely.geometry import Point

import data_preprocessing.create_positive_data as positive


class FakeRaster:
    width = 10_000
    height = 10_000

    def index(self, x, y):
        return 5_000, 5_000


class RasterContext:
    def __enter__(self):
        return FakeRaster()

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_crop_images_produces_three_crops_with_consecutive_indices(
    tmp_path,
    monkeypatch,
):
    raster_path = tmp_path / "fake.jp2"
    raster_path.touch()

    monkeypatch.setattr(
        positive,
        "find_raster_file_for_geometry",
        lambda geometry, path_raw_data: raster_path,
    )
    monkeypatch.setattr(
        positive.rasterio,
        "open",
        lambda path: RasterContext(),
    )

    saved = []

    def fake_save(src, window, output_dir, crop_index):
        saved.append((window, crop_index))

    monkeypatch.setattr(positive, "save_crop_image", fake_save)

    next_index = positive.crop_images(
        geometry=Point(100, 100),
        path_raw_data=tmp_path,
        output_dir=tmp_path,
        crop_index=7,
        rng=random.Random(42),
        img_size=2048,
    )

    assert [index for _, index in saved] == [7, 8, 9]
    assert len(saved) == 3
    assert saved[0][0].width == 2048
    assert saved[0][0].height == 2048
    assert saved[1][0].width == 1024
    assert saved[1][0].height == 1024
    assert saved[2][0].width == 2048
    assert saved[2][0].height == 2048
    assert next_index == 10

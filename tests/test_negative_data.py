import geopandas as gpd
from shapely.geometry import box

import data_preprocessing.create_negative_data as negative


class FakeRaster:
    width = 1000
    height = 1000

    def index(self, x, y):
        return 500, 500


class RasterContext:
    def __enter__(self):
        return FakeRaster()

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_negative_field_intersecting_rugby_field_is_rejected(
    tmp_path,
    monkeypatch,
):
    raster_path = tmp_path / "fake.jp2"
    raster_path.touch()

    candidate = gpd.GeoDataFrame(
        geometry=[box(0, 0, 10, 10)],
        crs="EPSG:2154",
    )
    rugby = gpd.GeoDataFrame(
        geometry=[box(0, 0, 10, 10)],
        crs="EPSG:2154",
    )

    monkeypatch.setattr(
        negative,
        "find_raster_file_for_geometry",
        lambda geometry, path: raster_path,
    )
    monkeypatch.setattr(
        negative.rasterio,
        "open",
        lambda path: RasterContext(),
    )
    monkeypatch.setattr(
        negative,
        "window_to_geometry",
        lambda window, src: box(0, 0, 10, 10),
    )

    saved_indices = []
    monkeypatch.setattr(
        negative,
        "save_crop_image",
        lambda src, window, output_dir, index: saved_indices.append(index),
    )

    next_index = negative.extract_negative_fields(
        gdf=candidate,
        path_ign_data=tmp_path,
        output_dir=tmp_path,
        gdf_rugby=rugby,
        n_negative_fields=1,
        seed=42,
        index_start=3,
        img_size=256,
    )

    assert saved_indices == []
    assert next_index == 3


def test_crs_normalization_preserves_spatial_intersection():
    rugby_l93 = gpd.GeoDataFrame(
        geometry=[box(573_000, 6_276_000, 574_000, 6_277_000)],
        crs="EPSG:2154",
    )

    rugby_wgs84 = rugby_l93.to_crs("EPSG:4326")
    normalized = negative.change_gdf_crs(rugby_wgs84, "EPSG:2154")

    crop = box(573_500, 6_276_500, 573_600, 6_276_600)

    assert normalized.intersects(crop).any()

import random
import rasterio
import geopandas

from pathlib import Path
from geopandas import GeoDataFrame
from shapely.geometry.base import BaseGeometry

from raster_utils import save_crop_image, clamp_window, find_raster_file_for_geometry, change_gdf_crs
from raster_utils import get_raster_crs

def crop_images(
        geometry: BaseGeometry,
        path_raw_data: Path,
        output_dir: Path,
        crop_index: int,
        rng: random.Random,
        img_size=2048,
    ) -> int:
    #multiply index by number of crops made with one image, here 3 (centered, zoomed, shifted ), to
    #be at the right index

    img_size_zoomed = int(0.5 * img_size)

    dx = int(rng.uniform(-0.25, 0.25) * img_size)
    dy = int(rng.uniform(-0.25, 0.25) * img_size)

    raster_path = find_raster_file_for_geometry(
        geometry,
        path_raw_data
    )

    if not raster_path.exists():
        raise FileNotFoundError(
            f"No orthophoto found for geometry centroid: {raster_path}"
        )

    with rasterio.open(raster_path) as src:
        point = geometry.representative_point()
        row, col = src.index(point.x, point.y)

        ## centered
        window_centered = clamp_window(col, row, img_size, src)

        save_crop_image(
            src,
            window_centered,
            output_dir,
            crop_index
        )
        crop_index += 1

        ## zoomed in
        window_centered_zoomed = clamp_window(col, row, img_size_zoomed, src)
        
        save_crop_image(
            src,
            window_centered_zoomed,
            output_dir,
            crop_index
        )
        crop_index += 1

        ##shifted randomly
        window_shifted = clamp_window(col - dx, row - dy, img_size, src)

        save_crop_image(
            src,
            window_shifted,
            output_dir,
            crop_index
        )
        crop_index += 1

    return crop_index


def make_positive_data(
        gdf_rugby_fields: GeoDataFrame,
        path_raw_data: Path,
        output_dir: Path,
        index_start: int = 0,
        seed: int = 42
    ):
    rng = random.Random(seed)

    target_crs = get_raster_crs(path_raw_data)
    gdf_rugby_fields = change_gdf_crs(gdf_rugby_fields, target_crs)

    output_dir.mkdir(parents=True)

    n_fields = gdf_rugby_fields.shape[0]
    crop_index = index_start

    for i in range(n_fields):
        field = gdf_rugby_fields.iloc[i]
        geom = field.geometry

        crop_index = crop_images(
            geom,
            path_raw_data,
            output_dir,
            crop_index=crop_index,
            rng=rng
        )

        if (i+1) % 10 == 0:
            print(f"{i+1} / {n_fields} done")
    

def main():
    base = Path().resolve()
    path_raw_data = base / "data/raw/D65/data_ign/BDORTHO_2-0_RVB-0M20_JP2-E080_LAMB93_D065_2025-01-01/ORTHOHR/1_DONNEES_LIVRAISON_2026-02-00070/OHR_RVB_0M20_JP2-E080_LAMB93_D65-2025"
    output_dir = base / "data/raw/D65/images_positives"

    gdf_rugby_fields = geopandas.read_file(base / "data/raw/D65/osm/export_rugby.geojson")
    gdf_rugby_fields = gdf_rugby_fields[["sport", "geometry"]]

    make_positive_data(gdf_rugby_fields, path_raw_data, output_dir)


if __name__ == "__main__":
    main()
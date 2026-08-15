"""
The idea here is to create "negative" images for training i.e images with no rugby fields. Best to put
other type of fields (football, tracks ...) for the model to understand what really is a rugby field.
For that we are going to use the data collected via OSM of sport fields that are not rugby and will also
add images taken randomly in the whole departement if those images do not intersect with known rugby
fields. We will only keep 2 points of view. There is no need for labeling here.
"""

import random
import rasterio
import geopandas

from pathlib import Path
from shapely.geometry import box
from geopandas import GeoDataFrame
from rasterio.io import DatasetReader
from rasterio.windows import Window, bounds

from .raster_utils import save_crop_image, change_gdf_crs, find_raster_file_for_geometry, clamp_window
from .raster_utils import get_raster_crs

def window_to_geometry(
    window: Window,
    src: DatasetReader,
):
    return box(
        *bounds(
            window,
            src.transform,
        )
    )

def crop_random_image(
        path_ign_data: Path,
        n_random_crops: int,
        gdf_rugby_fields: GeoDataFrame,
        output_dir: Path,
        index: int,
        rng: random.Random,
        img_size: int = 2048
):
    orthophotos = list(path_ign_data.rglob("*.jp2"))

    crops = 0

    max_attempts = n_random_crops * 100
    attempts = 0

    while crops < n_random_crops and attempts < max_attempts:
        attempts += 1
        
        path_image = rng.choice(orthophotos)

        with rasterio.open(path_image) as src:
            row = rng.randint(0, src.height - 1)
            col = rng.randint(0, src.width - 1)

            window = clamp_window(col, row, img_size, src)

            crop_geom = window_to_geometry(window, src)

            if not gdf_rugby_fields.intersects(crop_geom).any():
                save_crop_image(
                    src,
                    window,
                    output_dir,
                    index
                )

                index += 1
                crops += 1

    if crops < n_random_crops:
        raise RuntimeError(
            f"Only {crops}/{n_random_crops} valid random crops found "
            f"after {attempts} attempts."
        )


def extract_negative_fields(
        gdf: GeoDataFrame,
        path_ign_data: Path,
        output_dir: Path,
        gdf_rugby: GeoDataFrame,
        n_negative_fields: int,
        seed: int,
        index_start=0,
        img_size=2048
):
    gdf = gdf.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_fields = gdf.shape[0]

    fields_extracted = 0
    candidate_index = 0
    crop_index = index_start

    #condition to stop: no field left in gdf or enough fields extracted
    while candidate_index < n_fields and fields_extracted < n_negative_fields:
        field = gdf.iloc[candidate_index]
        geom = field.geometry

        point = geom.representative_point()
        x, y = point.x, point.y    
        
        raster_path = find_raster_file_for_geometry(geom, path_ign_data)

        if not raster_path.exists():
            raise FileNotFoundError(
                f"No orthophoto found for geometry centroid: {raster_path}"
            )
        
        with rasterio.open(raster_path) as src:

            row, col = src.index(x, y)

            candidate_window = clamp_window(col, row, img_size, src)

            candidate_crop_geometry = window_to_geometry(candidate_window, src)

            if gdf_rugby.intersects(candidate_crop_geometry).any():
                candidate_index += 1
                continue
                
            ## centered
            save_crop_image(
                    src,
                    candidate_window,
                    output_dir,
                    crop_index
            )
            crop_index += 1

            ## zoomed in
            window_zoomed = clamp_window(col, row, int(0.5 * img_size), src)

            save_crop_image(
                    src,
                    window_zoomed,
                    output_dir,
                    crop_index
                )
            crop_index += 1

        fields_extracted += 1
        candidate_index += 1

        if fields_extracted % 10 == 0:
            print(f"{fields_extracted} extracted")
            print(f"{candidate_index - fields_extracted} fields passed")
    

    return crop_index

def make_negative_data(
        gdf_rugby_free: GeoDataFrame,
        gdf_rugby_fields: GeoDataFrame,
        path_ign_data: Path,
        output_dir: Path,
        n_negative_fields: int = 100,
        n_random_crops: int = 50,
        seed: int = 42
):
    target_crs = get_raster_crs(path_ign_data)

    gdf_rugby_free = change_gdf_crs(
        gdf_rugby_free,
        target_crs
    )

    gdf_rugby_fields = change_gdf_crs(
        gdf_rugby_fields,
        target_crs
    )

    rng = random.Random(seed)

    output_dir.mkdir(parents=True)
    
    next_index = extract_negative_fields(
        gdf_rugby_free,
        path_ign_data,
        output_dir,
        gdf_rugby_fields,
        n_negative_fields,
        seed
    )

    crop_random_image(
        path_ign_data=path_ign_data,
        n_random_crops=n_random_crops,
        gdf_rugby_fields=gdf_rugby_fields,
        output_dir=output_dir,
        index=next_index,
        rng=rng
    )

def main():
    base = Path().resolve()
    path_ign_data = base / "data/raw/D65/data_ign/BDORTHO_2-0_RVB-0M20_JP2-E080_LAMB93_D065_2025-01-01/ORTHOHR/1_DONNEES_LIVRAISON_2026-02-00070/OHR_RVB_0M20_JP2-E080_LAMB93_D65-2025"
    output_dir = base / "data/raw/D65/images_negatives_test"

    gdf_rugby_fields = geopandas.read_file(base / "data/raw/D65/osm/export_rugby.geojson")

    gdf = geopandas.read_file(base / "data/raw/D65/osm/export_all_fields.geojson")
    gdf = gdf[["sport", "geometry"]]

    is_other_sport = gdf["sport"].str.contains(
        "football|soccer|athletics",
        case=False,
        na=False
    )

    is_rugby = gdf["sport"].str.contains(
        "rugby",
        case=False,
        na=False
    )

    gdf_rugby_free = gdf[is_other_sport & ~is_rugby]

    make_negative_data(
        gdf_rugby_free,
        gdf_rugby_fields,
        path_ign_data,
        output_dir
    )


if __name__ == "__main__":
    main()

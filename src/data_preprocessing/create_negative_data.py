"""
The idea here is to create "negative" images for training i.e images with no rugby fields. Best to put
other type of fields (football, tracks ...) for the model to understand what really is a rugby field.
For that we are going to use the data collected via OSM of sport fields that are not rugby and will also
add images taken randomly in the whole departement if those images do not intersect with known rugby
fields. We will only keep 2 points of view. There is no need for labeling here.

Goal : 400 images with other sports fields, 100 random images.
"""

import rasterio
import geopandas
import random
from shapely.geometry import box
import numpy as np
import re

from PIL import Image
from pathlib import Path
from rasterio.windows import Window
from rasterio.io import DatasetReader
from geopandas import GeoDataFrame

SEED = 42

def round_to_lower_multiple_of_5(n: int) -> int:
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a positive integer")
    
    while n % 5 != 0:
        n -= 1
    
    return n

def round_to_higher_multiple_of_5(n: int) -> int:
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a positive integer")
    
    while n % 5 != 0:
        n += 1
    
    return n

def get_departement_and_year(path_raw_data: Path):
    """all files follow the same naming convention as given by IGN. We can use the first file
    to guess the departement and year
    
    Returns departement, year"""
    file = next(path_raw_data.rglob("*.jp2"))
    pattern = r"^(\d+)-(\d+)-*"
    m = re.search(pattern, file.name)
    return int(m.group(1)), int(m.group(2))

def make_file_name(departement: int, year, x_bound, y_bound):
    return f"{departement}-{year}-0{x_bound}-{y_bound}-LA93-0M20-E080.jp2"


def generate_yolo_format_crop_from_window(
        src: DatasetReader,
        window: Window,
        output_dir: Path,
        crop_index: int,
):

    cropped_image = src.read((1, 2, 3), window=window) # shape: (bands, h, w)

    img = np.transpose(cropped_image, (1, 2, 0))  # -> (h, w, bands)

    Image.fromarray(img).save(output_dir / f"img_{crop_index}.jpg")


def clamp_window(col: int, row: int, size: int, src: DatasetReader):
    col_off = col - size // 2
    row_off = row - size // 2

    col_off = max(0, col_off)
    row_off = max(0, row_off)

    col_off = min(col_off, src.width - size)
    row_off = min(row_off, src.height - size)

    return Window(col_off, row_off, size, size)

def crop_images(
        geometry,
        path_raw_data: Path,
        output_dir: Path,
        index: int,
        crops_made_from_img,
        departement,
        year_orthophtos,
        img_size=2048,
    ):
    #multiply index by number of crops made with one image, here 2 (centered, zoomed), to
    #be at the right index
    index = index * crops_made_from_img

    img_size_zoomed = int(0.5 * img_size)

    point = geometry.representative_point()
    x, y = point.x, point.y

    x_km = int(x / 1000)
    y_km = int(y / 1000) + 1

    x_km = round_to_lower_multiple_of_5(x_km)
    y_km = round_to_higher_multiple_of_5(y_km)

    file_name = make_file_name(departement, year_orthophtos, x_km, y_km)
    
    raster_path = path_raw_data / file_name

    if not raster_path.exists():
        raise FileNotFoundError(
            f"No orthophoto found for geometry centroid: {raster_path}"
        )

    with rasterio.open(path_raw_data.joinpath(file_name)) as src:
        row, col = src.index(x, y)
        
        ## centered
        window_centered = clamp_window(col, row, img_size, src)

        generate_yolo_format_crop_from_window(
            src,
            window_centered,
            output_dir,
            index
        )

        index += 1

        ## zoomed in
        window_centered_zoomed = clamp_window(col, row, img_size_zoomed, src)
        
        generate_yolo_format_crop_from_window(
            src,
            window_centered_zoomed,
            output_dir,
            index
        )

        index += 1

    src.close()

def crop_random_image(
        path_raw_data: Path,
        n_images_to_crop: int,
        gdf_boxes_rugby: GeoDataFrame,
        output_dir: Path,
        index: int,
        img_size: int = 2048
):
    rng = random.Random(SEED)

    orthophotos = list(path_raw_data.rglob("*.jp2"))

    crops = 0

    max_attempts = n_images_to_crop * 100
    attempts = 0

    while crops < n_images_to_crop and attempts < max_attempts:
        attempts += 1
        
        path_image = rng.choice(orthophotos)

        with rasterio.open(path_image) as src:
            row = rng.randint(0, src.height - 1)
            col = rng.randint(0, src.width - 1)

            window = clamp_window(col, row, img_size, src)

            transform = src.window_transform(window)

            xmin_img, ymax_img = transform * (0, 0)
            xmax_img, ymin_img = transform * (window.width, window.height)

            crop_geom = box(xmin_img, ymin_img, xmax_img, ymax_img)

            if not gdf_boxes_rugby.intersects(crop_geom).any():
                generate_yolo_format_crop_from_window(
                    src,
                    window,
                    output_dir,
                    index
                )

                index += 1
                crops += 1

    if crops < n_images_to_crop:
        raise RuntimeError(
            f"Only {crops}/{n_images_to_crop} valid random crops found "
            f"after {attempts} attempts."
        )

def extract_all_crops_from_gdf(
        gdf: GeoDataFrame,
        path_raw_data: Path,
        output_dir: Path,
        gdf_rugby: GeoDataFrame,
        n_negative_fields,
        crops_made_from_img,
        index_start=0
    ):
    """
    path_to_dir is the path to the directory where crops are stored. Starts from the project base
    directory. 
    """
    departement, year = get_departement_and_year(path_raw_data)

    TARGET_CRS = "EPSG:9794"

    if gdf.crs is None:
        raise ValueError("The input GeoDataFrame has no CRS.")

    gdf = gdf.to_crs(TARGET_CRS)

    gdf = gdf.sample(frac=1, random_state=SEED).reset_index(drop=True)

    n_fields = gdf.shape[0]

    fields_extracted = 0
    i = 0

    #condition to stop: no field left in gdf or enough fields extracted
    while i < n_fields and fields_extracted < n_negative_fields:
        field = gdf.iloc[i]
        geom = field.geometry

        if not gdf_rugby.intersects(geom).any():

            crop_images(
                geom,
                path_raw_data,
                output_dir,
                index=fields_extracted + index_start,
                crops_made_from_img=crops_made_from_img,
                departement=departement,
                year_orthophtos=year
            )
            fields_extracted += 1

            if fields_extracted % 10 == 0:
                print(f"{fields_extracted} extracted")
                print(f"{i - fields_extracted} fields passed")
        
        i += 1

    return fields_extracted + index_start

def make_negative_data(
        gdf_rugby_free: GeoDataFrame,
        gdf_positive_crops_boxes: GeoDataFrame,
        path_raw_data: Path,
        output_dir: Path,
        overwrite_output_dir: bool = False,
        crops_made_from_img: int = 2,
        n_negative_fields: int = 100
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if any(output_dir.glob()) and not overwrite_output_dir:
        raise ValueError("The output directory already contains files. If you want to overwrite it, " \
        "pass the overwrite_output_dir to True.")
    
    next_index = extract_all_crops_from_gdf(
        gdf_rugby_free,
        path_raw_data,
        output_dir,
        gdf_positive_crops_boxes,
        crops_made_from_img=crops_made_from_img,
        n_negative_fields=n_negative_fields
    )

    crop_random_image(
    path_raw_data=path_raw_data,
    n_images_to_crop=50,
    gdf_boxes_rugby=gdf_positive_crops_boxes,
    output_dir=output_dir,
    index=next_index
    )

def main():
    base = Path().resolve()
    path_raw_data = base / "data/raw/D65/data_ign/BDORTHO_2-0_RVB-0M20_JP2-E080_LAMB93_D065_2025-01-01/ORTHOHR/1_DONNEES_LIVRAISON_2026-02-00070/OHR_RVB_0M20_JP2-E080_LAMB93_D65-2025"
    output_dir = base / "data/raw/D65/images_negatives"

    gdf_positive_crops_boxes = geopandas.read_file(base / "data/raw/D65/geom_rugby_fields/crops_geom.json")

    gdf = geopandas.read_file(base / "data/raw/D65/osm/export_all_fields.geojson")
    gdf = gdf[["sport", "geometry"]]
    gdf_rugby_free = gdf[gdf["sport"].str.contains("football|soccer|athletics", na=False)]

    make_negative_data(
        gdf_rugby_free,
        gdf_positive_crops_boxes,
        path_raw_data,
        output_dir
    )


if __name__ == "__main__":
    main()

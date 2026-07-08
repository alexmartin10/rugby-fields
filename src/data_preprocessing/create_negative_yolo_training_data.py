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

from PIL import Image
from pathlib import Path
from rasterio.windows import Window
from rasterio.io import DatasetReader
from geopandas import GeoDataFrame

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

def make_file_name(departement: int, year, x_bound, y_bound):
    return f"{departement}-{year}-0{x_bound}-{y_bound}-LA93-0M20-E080.jp2"


def generate_yolo_format_crop_from_window(
        src: DatasetReader,
        window: Window,
        path_yolo_dataset: Path,
        crop_index: int,
):

    cropped_image = src.read((1, 2, 3), window=window) # shape: (bands, h, w)

    img = np.transpose(cropped_image, (1, 2, 0))  # -> (h, w, bands)

    Image.fromarray(img).save(path_yolo_dataset / f"images/img_{crop_index}.jpg")

    with open(path_yolo_dataset.joinpath(f"labels/img_{crop_index}.txt"), "w") as f:
        pass

    f.close()

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
        path_yolo_dataset: Path,
        index: int,
        crops_made_from_img,
        departement,
        year_orthophtos,
        img_size=2048,
    ):
    #multiply index by number of crops made with one image, here 4 (centered, zoomed, shifted x2), to
    #be at the right index
    index = index * crops_made_from_img

    img_size_zoomed = int(0.5 * img_size)

    x = geometry.centroid.x
    y = geometry.centroid.y

    x_km = int(x / 1000)
    y_km = int(y / 1000) + 1

    x_km = round_to_lower_multiple_of_5(x_km)
    y_km = round_to_higher_multiple_of_5(y_km)

    file_name = make_file_name(departement, year_orthophtos, x_km, y_km)

    with rasterio.open(path_raw_data.joinpath(file_name)) as src:
        row, col = src.index(x, y)
        
        ## centered
        window_centered = clamp_window(col, row, img_size, src)

        generate_yolo_format_crop_from_window(
            src,
            window_centered,
            path_yolo_dataset,
            index
        )

        index += 1

        ## zoomed in
        window_centered_zoomed = clamp_window(col, row, img_size_zoomed, src)
        
        generate_yolo_format_crop_from_window(
            src,
            window_centered_zoomed,
            path_yolo_dataset,
            index
        )

        index += 1

    src.close()

def crop_random_image(
        path_raw_data: Path,
        n_images_to_crop: int,
        gdf_boxes_rugby: GeoDataFrame,
        path_yolo_dataset: Path,
        index: int,
        img_size: int = 2048
):
    orthophotos = list(path_raw_data.rglob("*.jp2"))

    c = 0

    while c < n_images_to_crop:
        path_image = random.choice(orthophotos)

        with rasterio.open(path_image) as src:
            row = random.randint(0, src.height - 1)
            col = random.randint(0, src.width - 1)

            window = clamp_window(col, row, img_size, src)

            transform = src.window_transform(window)

            xmin_img, ymax_img = transform * (0, 0)
            xmax_img, ymin_img = transform * (window.width, window.height)

            crop_geom = box(xmin_img, ymin_img, xmax_img, ymax_img)

            if not gdf_boxes_rugby.intersects(crop_geom).any():
                generate_yolo_format_crop_from_window(
                    src,
                    window,
                    path_yolo_dataset,
                    index
                )

                index += 1
                c += 1

def extract_all_crops_from_gdf(
        gdf: GeoDataFrame,
        path_raw_data: Path,
        path_yolo_dataset: Path,
        gdf_boxes_rugby: GeoDataFrame,
        index_start=0,
        n_negative_fields=215,
        crops_made_from_img=3
    ):
    """
    path_to_dir is the path to the directory where crops are stored. Starts from the project base
    directory. 
    """
    if gdf.crs.name != "RGF93 v2b / Lambert-93":
        gdf = gdf.to_crs('EPSG:9794')
        print("Coordinates system changed to Lambert 93.")

    n_fields = gdf.shape[0]

    fields_extracted = 0
    i = 0

    #condition to stop: no field left in gdf or enough fields extracted
    while i < n_fields and fields_extracted <= n_negative_fields:
        field = gdf.iloc[i]
        geom = field.geometry

        if not gdf_boxes_rugby.intersects(geom).any():

            crop_images(
                geom,
                path_raw_data,
                path_yolo_dataset,
                index=fields_extracted + index_start,
                crops_made_from_img=crops_made_from_img,
                departement=31,
                year_orthophtos=2025
            )
            fields_extracted += 1

            if fields_extracted % 10 == 0:
                print(f"{fields_extracted} extracted")
                print(f"{i - fields_extracted} fields passed")
        
        i += 1


def extract_crops_from_one(
        gdf: GeoDataFrame,
        path_raw_data: Path,
        path_yolo_dataset: Path,
        index_in_gdf: int
    ):
    """
    path_to_dir is the path to the directory where crops are stored. Starts from the project base
    directory. 
    """
    if gdf.crs.name != "RGF93 v2b / Lambert-93":
        gdf = gdf.to_crs('EPSG:9794')
        print("Coordinates system changed to Lambert 93.")

    field = gdf.iloc[index_in_gdf]
    geom = field.geometry

    crop_images(
        geom,
        path_raw_data,
        path_yolo_dataset,
        index=0,
        crops_made_from_img=3,
        gdf=gdf,
        departement=31,
        year_orthophtos=2025
    )

def main():
    p = Path().resolve()
    base = p.parent.parent
    path_raw_data = base.joinpath("data/raw/data-ign/D31/BDORTHO_2-0_RVB-0M20_JP2-E080_LAMB93_D031_2025-01-01/ORTHOHR/1_DONNEES_LIVRAISON_2026-04-00085/OHR_RVB_0M20_JP2-E080_LAMB93_D31-2025")
    path_yolo_dataset = base.joinpath("data/yolo_dataset_negative").resolve()    

    gdf_crops_rugby = geopandas.read_file(base / "data/yolo_images_geom/crops_geom.json")

    gdf = geopandas.read_file(base.joinpath("data/raw/osm/export_all_fields.geojson"))
    gdf = gdf[["sport", "geometry"]]
    gdf = gdf[gdf["sport"].str.contains("football|soccer|athletics", na=False)]

    extract_all_crops_from_gdf(
        gdf,
        path_raw_data,
        path_yolo_dataset,
        gdf_crops_rugby,
        crops_made_from_img=2
    )

    crop_random_image(
    path_raw_data=path_raw_data,
    n_images_to_crop=100,
    gdf_boxes_rugby=gdf_crops_rugby,
    path_yolo_dataset=path_yolo_dataset,
    index=430
    )
    # extract_crops_from_one(gdf, path_raw_data, p, 11)


if __name__ == "__main__":
    main()

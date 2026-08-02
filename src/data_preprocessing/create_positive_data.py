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
from shapely.geometry.polygon import Polygon

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


def make_crop_from_window(
        src: DatasetReader,
        window: Window,
        path_yolo_dataset: Path,
        crop_index: int
) -> Polygon:

    cropped_image = src.read((1, 2, 3), window=window) # shape: (bands, h, w)

    img = np.transpose(cropped_image, (1, 2, 0))  # -> (h, w, bands)

    Image.fromarray(img).save(path_yolo_dataset / f"images/img_{crop_index}.jpg")

    transform = src.window_transform(window)
    xmin_img, ymax_img = transform * (0, 0)
    xmax_img, ymin_img = transform * (window.height, window.width)

    crop_geom = box(xmin_img, ymin_img, xmax_img, ymax_img)

    #returned geometry is then used to check against in the process of building
    #negative data
    return crop_geom

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
    #multiply index by number of crops made with one image, here 3 (centered, zoomed, shifted ), to
    #be at the right index
    index = index * crops_made_from_img

    img_size_zoomed = int(0.5 * img_size)

    dx = int(random.uniform(-0.25, 0.25) * img_size)
    dy = int(random.uniform(-0.25, 0.25) * img_size)

    x = geometry.centroid.x
    y = geometry.centroid.y

    x_km = int(x / 1000)
    y_km = int(y / 1000) + 1

    x_km = round_to_lower_multiple_of_5(x_km)
    y_km = round_to_higher_multiple_of_5(y_km)

    file_name = make_file_name(departement, year_orthophtos, x_km, y_km)

    with rasterio.open(path_raw_data.joinpath(file_name)) as src:
        row, col = src.index(x, y)

        if img_size > min(src.width, src.height):
            raise ValueError("Crop image size must be lower than original image size.")
        ## centered
        window_centered = clamp_window(col, row, img_size, src)

        crop_geom_centered = make_crop_from_window(
            src,
            window_centered,
            path_yolo_dataset,
            index
        )

        index += 1

        ## zoomed in
        window_centered_zoomed = clamp_window(col, row, img_size_zoomed, src)
        
        crop_geom_zoomed = make_crop_from_window(
            src,
            window_centered_zoomed,
            path_yolo_dataset,
            index
        )

        index += 1

        ##shifted randomly
        window_shifted = clamp_window(col - dx, row - dy, img_size, src)

        crop_geom_shifted = make_crop_from_window(
            src,
            window_shifted,
            path_yolo_dataset,
            index
        )

    src.close()

    return [crop_geom_centered, crop_geom_zoomed, crop_geom_shifted]


def extract_all_crops_from_gdf(
        gdf: GeoDataFrame,
        path_raw_data: Path,
        path_save: Path,
        path_save_crop_geometries: Path,
        index_start=0,
        crops_made_from_img=3
    ):
    """
    path_save_crop_geometries is the path to which we save the gdf containing the geometry of all the
    images we cropped. We then use it to create the negative dataset (containing no rugby fields).

    """
    departement, year_orthophotos = get_departement_and_year(path_raw_data)
    all_geoms = []

    images_dir = path_save / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if gdf.crs.name != "RGF93 v2b / Lambert-93":
        gdf = gdf.to_crs('EPSG:9794')
        print("Coordinates system changed to Lambert 93.")

    n_fields = gdf.shape[0]

    for i in range(n_fields):
        field = gdf.iloc[i]
        geom = field.geometry

        geom_list = crop_images(
            geom,
            path_raw_data,
            path_save,
            index=i + index_start,
            crops_made_from_img=crops_made_from_img,
            departement=departement,
            year_orthophtos=year_orthophotos
        )

        all_geoms += geom_list

        if (i+1) % 10 == 0:
            print(f"{i+1} / {n_fields} done")
    
    gdf_boxes = GeoDataFrame(geometry=all_geoms, crs='EPSG:9794')
    gdf_boxes.to_file(path_save_crop_geometries, driver="GeoJSON")

def main():
    base = Path().resolve()
    path_raw_data = base / "data/raw/D65/data_ign/BDORTHO_2-0_RVB-0M20_JP2-E080_LAMB93_D065_2025-01-01/ORTHOHR/1_DONNEES_LIVRAISON_2026-02-00070/OHR_RVB_0M20_JP2-E080_LAMB93_D65-2025"
    path_save = base / "data/raw/D65/yolo_positives"
    path_save_crop_geom = base / "data/raw/D65/geom_rugby_fields/crops_geom.json"
    gdf = geopandas.read_file(base / "data/raw/D65/osm/export_rugby.geojson")
    gdf = gdf[["sport", "geometry"]]

    extract_all_crops_from_gdf(gdf, path_raw_data, path_save, path_save_crop_geom)


if __name__ == "__main__":
    main()
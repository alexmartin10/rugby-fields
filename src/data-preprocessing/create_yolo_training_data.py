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

def make_file_name(departement: int, year, x_bound, y_bound):
    return f"{departement}-{year}-0{x_bound}-{y_bound}-LA93-0M20-E080.jp2"

def calculate_yolo_coordinates(
    x_field_center: int,
    y_field_center: int,
    xmin_img: int,
    xmax_img: int,
    ymin_img:int,
    ymax_img: int,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int
):
    """
    x_field_center, y_field_center are the coordinates of the center of the field. Given by OSM.
    xmin_img, ymax_img are the coordinates of the point at the extreme North-West of the image,
    to put it simplier, the pixel (0, 0) of the image (top left).
    xmax_img, ymin_img, same but for the bottom right pixel, position (img_size, img_size)
    xmin, xmax, ymin, ymax are the bounds given by OSM.
    """
    #make sure bounds for the fields are inside the crop
    xmin = max(xmin, xmin_img)
    xmax = min(xmax, xmax_img)
    ymin = max(ymin, ymin_img)
    ymax = min(ymax, ymax_img)

    x_center = abs((x_field_center - xmin_img) / (xmax_img - xmin_img))
    y_center = abs((y_field_center - ymin_img) / (ymax_img - ymin_img))
    height = abs((ymax - ymin) / (ymax_img - ymin_img))
    width = abs((xmax - xmin) / (xmax_img - xmin_img))

    return x_center, y_center, height, width


def generate_yolo_format_crop_from_window(
        src: DatasetReader,
        window: Window,
        path_yolo_dataset: Path,
        crop_index: int,
        gdf: GeoDataFrame
) -> Polygon:

    cropped_image = src.read((1, 2, 3), window=window) # shape: (bands, h, w)

    img = np.transpose(cropped_image, (1, 2, 0))  # -> (h, w, bands)

    Image.fromarray(img).save(path_yolo_dataset / f"images/img_{crop_index}.jpg")

    transform = src.window_transform(window)
    xmin_img, ymax_img = transform * (0, 0)
    xmax_img, ymin_img = transform * (window.height, window.width)

    crop_geom = box(xmin_img, ymin_img, xmax_img, ymax_img)

    gdf_all_fields_on_crop = gdf[gdf.intersects(crop_geom)]

    with open(
        path_yolo_dataset.joinpath(f"labels/img_{crop_index}.txt"),
        "w"
    ) as f:
        for _, field in gdf_all_fields_on_crop.iterrows():
            visible = field.geometry.intersection(crop_geom)

            ratio_visible = visible.area / field.geometry.area

            #only annotate a field if we can see >40% of his surface
            if ratio_visible >= 0.4:
                xmin, ymin, xmax, ymax = visible.bounds
                xc = visible.centroid.x
                yc = visible.centroid.y
                x_center, y_center, height, widht = calculate_yolo_coordinates(
                    xc,
                    yc,
                    xmin_img,
                    xmax_img,
                    ymin_img,
                    ymax_img,
                    xmin,
                    xmax,
                    ymin,
                    ymax
                )
                f.write(f"0 {x_center:.6f} {y_center:.6f} {widht:.6f} {height:.6f}\n")
        
    f.close()

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
        gdf,
        departement,
        year_orthophtos,
        img_size=2048,
    ):
    #multiply index by number of crops made with one image, here 4 (centered, zoomed, shifted x2), to
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

        crop_geom_centered = generate_yolo_format_crop_from_window(
            src,
            window_centered,
            path_yolo_dataset,
            index,
            gdf
        )

        index += 1

        ## zoomed in
        window_centered_zoomed = clamp_window(col, row, img_size_zoomed, src)
        
        crop_geom_zoomed = generate_yolo_format_crop_from_window(
            src,
            window_centered_zoomed,
            path_yolo_dataset,
            index,
            gdf
        )

        index += 1

        ##shifted randomly
        window_shifted = clamp_window(col - dx, row - dy, img_size, src)

        crop_geom_shifted = generate_yolo_format_crop_from_window(
            src,
            window_shifted,
            path_yolo_dataset,
            index,
            gdf
        )

    src.close()

    return [crop_geom_centered, crop_geom_zoomed, crop_geom_shifted]


def extract_all_crops_from_gdf(
        gdf: GeoDataFrame,
        path_raw_data: Path,
        path_yolo_dataset: Path,
        path_save_crop_geometries: Path,
        index_start=0
    ):
    """
    path_save_crop_geometries is the path to which we save the gdf containing the geometry of all the
    images we cropped. We then use it to create the negative dataset (containing no rugby fields).

    """
    all_geoms = []

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
            path_yolo_dataset,
            index=i + index_start,
            crops_made_from_img=3,
            gdf=gdf,
            departement=31,
            year_orthophtos=2025
        )

        all_geoms += geom_list

        if (i+1) % 10 == 0:
            print(f"{i+1} / {n_fields} done")
    
    gdf_boxes = GeoDataFrame(geometry=all_geoms, crs='EPSG:9794')
    gdf_boxes.to_file(path_save_crop_geometries, driver="GeoJSON")

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
    path_yolo_dataset = base.joinpath("data/yolo_dataset").resolve()
    path_save_crop_geom = base / "data/yolo_images_geom/crops_geom.json"
    gdf = geopandas.read_file(base.joinpath("data/raw/osm/export_rugby.geojson"))
    gdf = gdf[["sport", "geometry"]]

    extract_all_crops_from_gdf(gdf, path_raw_data, path_yolo_dataset, path_save_crop_geom)
    # extract_crops_from_one(gdf, path_raw_data, p, 11)


if __name__ == "__main__":
    main()

import rasterio
import geopandas
import random
from shapely.geometry import box


from pathlib import Path
from rasterio.windows import Window
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

def calculate_yolo_coordinates(
        x_field_center: int,
        y_field_center: int,
        x_nw_image: int,
        y_nw_image:int,
        x_se_image: int,
        y_se_image: int,
        xmin: int,
        xmax: int,
        ymin: int,
        ymax: int
    ):
    """
    x_field_center, y_field_center are the coordinates of the center of the field. Can be computed via
    OSM.
    x_nw_image, y_nw_image are the coordinates of the point at the extreme North-West of the image,
    to put it simplier, the pixel (0, 0) of the image (top left).
    x_se_image, y_se_image, same but for the bottom right pixel, position (img_size, img_size)
    xmin, xmax, ymin, ymax are the bounds given by OSM.
    """
    x_center = abs((x_field_center - x_nw_image) / (x_se_image - x_nw_image))
    y_center = abs((y_field_center - y_se_image) / (y_nw_image - y_se_image))
    height = abs((ymax - ymin) / (y_nw_image - y_se_image))
    width = abs((xmax - xmin) / (x_se_image - x_nw_image))

    return x_center, y_center, height, width

def crop_images_from_geom(
        geometry,
        path_raw_data: Path,
        path_yolo_dataset: Path,
        index: int,
        crops_made_from_img,
        gdf,
        img_size=2048,
    ):
    #multiply index by number of crops made with one image, here 4 (centered, zoomed, shifted x2), to
    #be at the right index
    index = index * crops_made_from_img

    xmin, ymin, xmax, ymax = geometry.bounds

    img_size_zoomed = int(0.5 * img_size)

    dx = int(random.uniform(-0.25, 0.25) * img_size)
    dy = int(random.uniform(-0.25, 0.25) * img_size)

    field_center = geometry.centroid

    x, y = field_center.x, field_center.y
    print(x, y)

    x_km = int(x / 1000)
    y_km = int(y / 1000)

    x_km = round_to_lower_multiple_of_5(x_km)
    y_km = round_to_higher_multiple_of_5(y_km)

    file_name = make_file_name(31, 2025, x_km, y_km)

    ## centered

    with rasterio.open(path_raw_data.joinpath(file_name)) as src:
        row, col = src.index(x, y)
        profile = src.profile.copy()

        window_centered = Window(
            col - int(img_size / 2),
            row - int(img_size / 2),
            img_size,
            img_size
        )
        cropped_image_centered = src.read((1, 2, 3), window=window_centered)
        profile.update({
            "driver": "JPEG",
            "height": int(window_centered.height),
            "width": int(window_centered.width),
            "transform": src.window_transform(window_centered)
        })

        with rasterio.open(
            path_yolo_dataset.joinpath(f"images/img_{index}.jpg").resolve(),
            "w",
            **profile
        ) as dst:
            dst.write(cropped_image_centered)
            x1, y1 = dst.xy(0, 0)
            x2, y2 = dst.xy(img_size, img_size)
        
        dst.close()

        with open(
            path_yolo_dataset.joinpath(f"labels/img_{index}.txt"),
            "w"
        ) as f:
            x_center, y_center, height, widht = calculate_yolo_coordinates(
                x,
                y,
                x1,
                y1,
                x2,
                y2,
                xmin,
                xmax,
                ymin,
                ymax
            )
            f.write(f"0 {x_center:.6f} {y_center:.6f} {widht:.6f} {height:.6f}")
        
        f.close()

        index += 1

        ## zoomed in 
        profile = src.profile.copy()

        window_centered_zoomed = Window(
            col - int(img_size_zoomed / 2),
            row - int(img_size_zoomed / 2),
            img_size_zoomed,
            img_size_zoomed
        )
        cropped_image_centered_zoomed = src.read((1, 2, 3), window=window_centered_zoomed)
        profile.update({
            "driver": "JPEG",
            "height": int(window_centered_zoomed.height),
            "width": int(window_centered_zoomed.width),
            "transform": src.window_transform(window_centered_zoomed)
        })

        with rasterio.open(
            path_yolo_dataset.joinpath(f"images/img_{index}.jpg").resolve(),
            "w",
            **profile
        ) as dst:
            dst.write(cropped_image_centered_zoomed)
            x1, y1 = dst.xy(0, 0)
            x2, y2 = dst.xy(img_size_zoomed, img_size_zoomed)

        dst.close()
    
        with open(
            path_yolo_dataset.joinpath(f"labels/img_{index}.txt"),
            "w"
        ) as f:
            x_center, y_center, height, widht = calculate_yolo_coordinates(
                x,
                y,
                x1,
                y1,
                x2,
                y2,
                xmin,
                xmax,
                ymin,
                ymax
            )
            f.write(f"0 {x_center:.6f} {y_center:.6f} {widht:.6f} {height:.6f}")
        
        f.close()

        index += 1

        ##shifted randomly
        profile = src.profile.copy()

        window_shifted = Window(
            col - int(img_size / 2) - dx,
            row - int(img_size / 2) - dy,
            img_size,
            img_size
        )

        cropped_image_shifted = src.read((1, 2, 3), window=window_shifted)
        profile.update({
            "driver": "JPEG",
            "height": int(window_shifted.height),
            "width": int(window_shifted.width),
            "transform": src.window_transform(window_shifted)
        })

        with rasterio.open(
            path_yolo_dataset.joinpath(f"images/img_{index}.jpg").resolve(),
            "w",
            **profile
        ) as dst:
            dst.write(cropped_image_shifted)
            x1, y1 = dst.xy(0, 0)
            x2, y2 = dst.xy(img_size, img_size)
        dst.close()

        with open(
            path_yolo_dataset.joinpath(f"labels/img_{index}.txt"),
            "w"
        ) as f:
            x_center, y_center, height, widht = calculate_yolo_coordinates(
                x,
                y,
                x1,
                y1,
                x2,
                y2,
                xmin,
                xmax,
                ymin,
                ymax
            )
            f.write(f"0 {x_center:.6f} {y_center:.6f} {widht:.6f} {height:.6f}")
        
        f.close()


    src.close()


def extract_all_crops_from_gdf(gdf: GeoDataFrame):
    """
    path_to_dir is the path to the directory where crops are stored. Starts from the project base
    directory. 
    """
    if gdf.crs.name != "RGF93 v2b / Lambert-93":
        gdf = gdf.to_crs('EPSG:9794')
        print("Coordinates system changed to Lambert 93.")

    p = Path()
    project_base = p.resolve().parent.parent

    path_raw_data = project_base.joinpath("data/raw/data-ign/D31/BDORTHO_2-0_RVB-0M20_JP2-E080_LAMB93_D031_2025-01-01/ORTHOHR/1_DONNEES_LIVRAISON_2026-04-00085/OHR_RVB_0M20_JP2-E080_LAMB93_D31-2025")
    
    base_path_save_images = project_base.joinpath("data/images/fields").resolve()

    n_fields = gdf.shape[0]
    for i in range(n_fields):
        field = gdf.iloc[i]
        geom = field.geometry
        path_save_images = base_path_save_images.joinpath(f"field-{i+1}")
        if not path_save_images.is_dir():
            path_save_images.mkdir()
        crop_images_from_geom(
            geom,
            path_raw_data,
            path_save_images
        )

        if (i+1) % 10 == 0:
            print(f"{i+1} / {n_fields} done")

def main():
    gdf = geopandas.read_file(Path().resolve().parent.parent.joinpath("/data/raw/osm/export.geojson"))
    gdf = gdf[["sport", "geometry"]]
    gdf_rugby = gdf[gdf["sport"].str.contains("rugby", na=False)]

    ##added
    gdf_l93 = gdf_rugby.to_crs('EPSG:9794')
    field_1 = gdf_l93.iloc[0]
    geometry = field_1.geometry

    p = Path().resolve()
    crop_images_from_geom(geometry, p, p, 0, 3, gdf_l93)
    ##

    # extract_all_crops_from_gdf(gdf_rugby)

if __name__ == "__main__":
    main()

import rasterio
import geopandas
import random


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



def crop_images_from_geom(geometry, path_raw_data: Path, path_save_images: Path, img_size=1024):
    xmin, ymin, xmax, ymax = geometry.bounds

    img_size_zoomed = int(0.9 * img_size)
    imgs = {}
    dx = int(random.uniform(-0.1, 0.1) * img_size_zoomed)
    dy = int(random.uniform(-0.1, 0.1) * img_size_zoomed)

    field_center = geometry.centroid

    x, y = field_center.x, field_center.y

    x_km = int(x / 1000)
    y_km = int(y / 1000)

    x_km = round_to_lower_multiple_of_5(x_km)
    y_km = round_to_higher_multiple_of_5(y_km)

    file_name = make_file_name(31, 2025, x_km, y_km)

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
            "driver": "GTiff",
            "height": int(window_centered.height),
            "width": int(window_centered.width),
            "transform": src.window_transform(window_centered)
        })

        with rasterio.open(
            path_save_images.joinpath("cropped_image_centered.tif"),
            "w",
            **profile
        ) as dst:
            dst.write(cropped_image_centered)
        
        dst.close()

     
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
            "driver": "GTiff",
            "height": int(window_centered_zoomed.height),
            "width": int(window_centered_zoomed.width),
            "transform": src.window_transform(window_centered_zoomed)
        })

        with rasterio.open(
            path_save_images.joinpath("cropped_image_centered_zoomed.tif"),
            "w",
            **profile
        ) as dst:
            dst.write(cropped_image_centered_zoomed)
        
        dst.close()

        ##shifted vert
        profile = src.profile.copy()

        window_shifted_vert = Window(
            col - int(img_size_zoomed / 2),
            row - int(img_size_zoomed / 2) - dy,
            img_size_zoomed,
            img_size_zoomed
        )

        cropped_image_shifted_vert = src.read((1, 2, 3), window=window_shifted_vert)
        profile.update({
            "driver": "GTiff",
            "height": int(window_shifted_vert.height),
            "width": int(window_shifted_vert.width),
            "transform": src.window_transform(window_shifted_vert)
        })

        with rasterio.open(
            path_save_images.joinpath("cropped_image_shifted_vert.tif"),
            "w",
            **profile
        ) as dst:
            dst.write(cropped_image_shifted_vert)
        
        dst.close()


        ## shifted hor
        profile = src.profile.copy()

        window_shifted_hor = Window(
            col - int(img_size_zoomed / 2) - dx,
            row - int(img_size_zoomed / 2),
            img_size_zoomed,
            img_size_zoomed
        )

        cropped_image_shifted_hor = src.read((1, 2, 3), window=window_shifted_hor)
        profile.update({
            "driver": "GTiff",
            "height": int(window_shifted_hor.height),
            "width": int(window_shifted_hor.width),
            "transform": src.window_transform(window_shifted_hor)
        })

        with rasterio.open(
            path_save_images.joinpath("cropped_image_shifted_hor.tif"),
            "w",
            **profile
        ) as dst:
            dst.write(cropped_image_shifted_hor)
        
        dst.close()
        
        ## over

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
    gdf = geopandas.read_file("/home/alexandre-martin/Documents/rugby-fields/data/raw/osm/export.geojson")
    gdf = gdf[["sport", "geometry"]]
    gdf_rugby = gdf[gdf["sport"].str.contains("rugby", na=False)]

    extract_all_crops_from_gdf(gdf_rugby)

if __name__ == "__main__":
    main()
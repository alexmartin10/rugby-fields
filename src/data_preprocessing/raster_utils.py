import re
import rasterio
import numpy as np

from PIL import Image
from pathlib import Path
from functools import lru_cache
from rasterio.windows import Window
from rasterio.io import DatasetReader
from geopandas import GeoDataFrame
from shapely.geometry.base import BaseGeometry

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

@lru_cache
def get_departement_and_year(path_raw_data: Path) -> tuple[int, int]:
    """all files follow the same naming convention as given by IGN. We can use the first file
    to guess the departement and year
    
    Returns departement, year"""
    file = next(path_raw_data.rglob("*.jp2"))
    pattern = r"^(\d+)-(\d+)-*"
    m = re.search(pattern, file.name)
    return int(m.group(1)), int(m.group(2))


def make_file_name(departement: int, year: int, x_bound: int, y_bound: int) -> str:
    return f"{departement}-{year}-0{x_bound}-{y_bound}-LA93-0M20-E080.jp2"

def save_crop_image(
        src: DatasetReader,
        window: Window,
        output_dir: Path,
        crop_index: int,
):
    cropped_image = src.read((1, 2, 3), window=window) # shape: (bands, h, w)

    img = np.transpose(cropped_image, (1, 2, 0))  # -> (h, w, bands)

    Image.fromarray(img).save(output_dir / f"img_{crop_index}.jpg")

def change_gdf_crs(
        gdf: GeoDataFrame,
        target_crs: str
) -> GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("The input GeoDataFrame has no CRS.")

    return gdf.to_crs(target_crs)

def find_raster_file_for_geometry(
        geometry: BaseGeometry,
        path_raw_data: Path
) -> Path:
    departement, year = get_departement_and_year(path_raw_data)
    point = geometry.representative_point()
    x, y = point.x, point.y

    x_km = int(x / 1000)
    y_km = int(y / 1000) + 1

    x_km = round_to_lower_multiple_of_5(x_km)
    y_km = round_to_higher_multiple_of_5(y_km)

    file_name = make_file_name(departement, year, x_km, y_km)
        
    return path_raw_data / file_name


def clamp_window(
        col: int,
        row: int,
        size: int,
        src: DatasetReader
    ) -> Window:
    if size <= 0:
        raise ValueError("Size must be positive.")

    if size > src.width or size > src.height:
        raise ValueError(
            f"Window size ({size}) exceeds raster dimensions "
            f"({src.width}x{src.height})."
        )
    col_off = col - size // 2
    row_off = row - size // 2

    col_off = max(0, col_off)
    row_off = max(0, row_off)

    col_off = min(col_off, src.width - size)
    row_off = min(row_off, src.height - size)

    return Window(col_off, row_off, size, size)

def get_raster_crs(path_raw_data: Path):
    first_raster = next(
        path_raw_data.rglob("*.jp2"),
        None,
    )

    if first_raster is None:
        raise FileNotFoundError(
            f"No JP2 raster found in {path_raw_data}"
        )

    with rasterio.open(first_raster) as src:
        if src.crs is None:
            raise ValueError(
                f"Raster has no CRS: {first_raster}"
            )

        return src.crs
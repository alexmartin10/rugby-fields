from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

def convert_jp2_tile_to_jpg(path_to_jp2: Path, path_save: Path, window_size: int, overlap: float = 0.2):
    """
    Returns dict[index] = [row_start, col_start] for each cropped image.
    """
    if not 0 < overlap < 1:
        raise ValueError("Overlap must be stricly between 0 and 1")
    
    dict_index_pixels = {}

    with rasterio.open(path_to_jp2) as src:
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

        row = 0
        col = 0
        index = 0

        while row < height or col < width:

            row_start = int(row)
            row_stop = min(row + window_size, height)
            
            col_start = int(col)
            col_stop = min(col + window_size, width)

            window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
            image = src.read((1, 2, 3), window=window)

            img = np.transpose(image, (1, 2, 0))

            Image.fromarray(img, mode="RGB").save(path_save.joinpath(f"{index}.jpg"))

            col = col + window_size * (1 - overlap)

            if col >= width:
                row = row + window_size * (1 - overlap)
                if row <= height:
                    col = 0

            dict_index_pixels[index] = (row_start, col_start)

            index += 1

    return transform, crs, dict_index_pixels


def main():
    base = Path().resolve()
    path_to_jp2 = base / "data/raw/D33/data_ign/BDORTHO_2-0_RVB-0M20_JP2-E080_LAMB93_D033_2024-01-01/ORTHOHR/1_DONNEES_LIVRAISON_2024-10-00207/OHR_RVB_0M20_JP2-E080_LAMB93_D33-2024/33-2024-0410-6425-LA93-0M20-E080.jp2"
    path_save = base / "data/raw/D33/dalle_jpg"
    convert_jp2_tile_to_jpg(path_to_jp2, path_save, window_size=2048)

if __name__ == "__main__":
    main()
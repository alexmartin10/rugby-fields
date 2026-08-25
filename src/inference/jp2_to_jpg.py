from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

def compute_start_indices(
        height: int,
        width: int,
        window_size: int,
        target_overlap: float,
        verbose: bool
):
    """
    Create a list of pixels bounds used to have the lowest possible number of windows.
    """
    target_stride = window_size * (1 - target_overlap)

    width_limit = width - window_size
    if width_limit == 0:
        col_starts = np.asarray([0])

    else:
        # number of windows with the target overlap
        n_windows_col = max(
            int(np.rint(width_limit / target_stride)) + 1, # n windows closest to target
            int(np.ceil(width_limit / window_size)) + 1, # n windows to cover full image
        )
        actual_stride_col = width_limit / (n_windows_col - 1)
        actual_overlap_col = 1 - actual_stride_col / window_size
        if verbose:
            print(f"Target overlap (width): {target_overlap}.")
            print(f"Actual overlap (width): {actual_overlap_col}. \n")

        col_starts = np.linspace(
            0,
            width_limit,
            num=n_windows_col,
            dtype=int,
        )

    height_limit = height - window_size
    if height_limit == 0:
        row_starts = np.asarray([0])
    else:
        n_windows_row = max(
            int(np.rint(height_limit / target_stride)) + 1, # n windows closest to target
            int(np.ceil(height_limit / window_size)) + 1, # n windows to cover full image
        )
        actual_stride_row = height_limit / (n_windows_row - 1)
        actual_overlap_row = 1 - actual_stride_row / window_size
        if verbose:
            print(f"Target overlap (height): {target_overlap}.")
            print(f"Actual overlap (height): {actual_overlap_row}.")

        row_starts = np.linspace(
            0,
            height_limit,
            num=n_windows_row,
            dtype=int,
        )
    
    return row_starts, col_starts

def jp2_tile_to_jpg(
        path_to_jp2: Path,
        path_save: Path, 
        window_size: int, 
        overlap: float = 0.2,
        verbose: bool = False
    ):
    """
    Returns dict[index] = [row_start, col_start] for each cropped image.
    """
    if not 0 < overlap < 1:
        raise ValueError("Overlap must be stricly between 0 and 1")

    if window_size <= 1:
        raise ValueError("Window size must be > 1")
    
    dict_index_pixels = {}

    with rasterio.open(path_to_jp2) as src:
        height = src.height
        width = src.width

        if window_size > min(width, height):
            window_size = min(width, height)
            print(f"Window size changed to {window_size} because it was larger than \
                at least one the tile's dimension.")
            
        transform = src.transform
        crs = src.crs

        row_start_indices, col_start_indices = compute_start_indices(
            width=width,
            height=height,
            window_size=window_size,
            target_overlap=overlap,
            verbose=verbose
        )
        index = 0

        for row_start in row_start_indices:

            row_stop = row_start + window_size

            for col_start in col_start_indices:

                col_stop = col_start + window_size

                window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
                image = src.read((1, 2, 3), window=window)

                img = np.transpose(image, (1, 2, 0))

                Image.fromarray(img, mode="RGB").save(path_save.joinpath(f"{index}.jpg"))

                dict_index_pixels[index] = (row_start, col_start)

                index += 1

    return transform, crs, dict_index_pixels

def main():
    base = Path().resolve()
    path_to_jp2 = base / "data/raw/D33/data_ign/BDORTHO_2-0_RVB-0M20_JP2-E080_LAMB93_D033_2024-01-01/ORTHOHR/1_DONNEES_LIVRAISON_2024-10-00207/OHR_RVB_0M20_JP2-E080_LAMB93_D33-2024/33-2024-0410-6425-LA93-0M20-E080.jp2"
    path_save = base / "data/raw/D33/test_compute"
    jp2_tile_to_jpg(path_to_jp2, path_save, window_size=2048, verbose=True)

if __name__ == "__main__":
    main()
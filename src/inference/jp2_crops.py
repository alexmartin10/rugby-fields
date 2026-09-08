import logging
import numpy as np

from rasterio.windows import Window

logger = logging.getLogger(__name__)

def compute_start_indices(
        height: int,
        width: int,
        window_size: int,
        target_overlap: float,
        verbose: bool
):
    """
    Compute the row and column offsets of full-size crop windows.

    Windows span each image dimension from zero to its last valid start position.
    The number of windows is selected to keep the effective overlap close to
    ``target_overlap`` while ensuring that no uncovered gap remains.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        window_size: Side length of each square crop in pixels.
        target_overlap: Desired overlap ratio between adjacent crops.
        verbose: Whether to display the effective overlaps.

    Returns:
        A tuple ``(row_starts, col_starts)`` containing the top-left pixel
        offsets of every crop along each axis.
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

def yield_crop(
        src,
        window_size: int, 
        overlap: float,
        verbose: bool
    ):
    """
    Yield full-size BGR crops from an opened Rasterio dataset.

    The caller owns ``src`` and must keep it open for the entire iteration.
    When the requested window size exceeds one tile dimension, it is reduced to
    the smallest tile dimension so that every yielded crop remains valid.

    Args:
        src: Open Rasterio dataset containing at least three image bands.
        window_size: Requested side length of each square crop in pixels.
        overlap: Target overlap ratio between adjacent crops.
        verbose: Whether to display the effective overlaps.

    Yields:
        Tuples ``(image, col_start, row_start, index)``, where ``image`` is a
        BGR NumPy array, ``col_start`` and ``row_start`` are the crop's top-left
        offsets in the tile, and ``index`` is its sequential identifier.

    Raises:
        ValueError: If ``overlap`` is not strictly between zero and one, or if
            ``window_size`` is less than two pixels.
    """
    if not 0 < overlap < 1:
        raise ValueError("Overlap must be stricly between 0 and 1")

    if window_size <= 1:
        raise ValueError("Window size must be > 1")

    height = src.height
    width = src.width

    if window_size > min(width, height):
        window_size = min(width, height)
        logger.warning(
            "Window size changed to %d because it was larger than \
            at least one the tile's dimension.",
            window_size
        )

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
            image = src.read((1, 2, 3), window=window) # (bands, rows, columns)

            img = np.transpose(image, (1, 2, 0))

            image = img[:, :, [2, 1, 0]] # RGB -> BGR

            yield image, col_start, row_start, index

            index += 1

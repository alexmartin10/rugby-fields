import time
import json
import logging
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from datetime import date
from ultralytics import YOLO
from shapely.geometry import Polygon

from .jp2_crops import yield_crop

logger = logging.getLogger(__name__)

def configure_logging(log_path: str | Path) -> None:
    """
    Configure application logging to both a file and the console.

    The log file is opened in append mode so resumed executions continue
    writing to the existing run log.

    Args:
        log_path: Path to the run log file.
    """
    log_path = Path(log_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

def predict_tile_obb(
    path_tile: str | Path,
    model: YOLO,
    window_size: int,
    window_overlap: float,
    predict_args: dict,
    verbose: bool = False
) -> tuple[gpd.GeoDataFrame, int]:
    """
    Run OBB inference on in-memory crops generated from a JP2 tile.

    Each crop is read lazily from the opened tile and passed directly to YOLO as a
    BGR NumPy array. Predicted pixel coordinates are shifted back into the tile
    coordinate system and converted into georeferenced polygons.

    Args:
        path_tile: Path to the JP2 tile.
        model: Loaded Ultralytics YOLO model.
        window_size: Crop width and height in pixels.
        window_overlap: Target overlap ratio between adjacent crops.
        predict_args: Additional arguments forwarded to ``model.predict``.
        verbose: Whether to display crop-generation information.

    Returns:
        A tuple containing the raw georeferenced predictions and the number of
        generated crops.
    """
    path_tile = Path(path_tile)
    tile_id = path_tile.stem

    fields_pixels = []
    confidences = []
    indexes = []
    n_crops = 0

    with rasterio.open(path_tile) as src:
    
        crs, transform = src.crs, src.transform

        for image, col_start, row_start, index in yield_crop(
            src=src,
            window_size=window_size,
            overlap=window_overlap,
            verbose=verbose
        ):
            n_crops += 1

            results = model.predict(
                source=image,
                **predict_args
            )

            for result in results:
                if result.obb is None or result.obb.data.numel() == 0:
                    continue

                indexes.extend([index] * len(result.obb))

                # Shape: (number of boxes, four vertices, two coordinates).
                pixels_tile = result.obb.xyxyxyxy.numpy().copy()

                # Shift every vertex from crop coordinates to tile coordinates.
                pixels_tile[:, :, 0] += col_start
                pixels_tile[:, :, 1] += row_start

                fields_pixels.extend(pixels_tile)

                confidences.extend(result.obb.conf.numpy())

    geometries = []

    for field in fields_pixels:
        coordinates_l93 = [
            transform * (float(x), float(y))
            for x, y in field
        ]

        polygon = Polygon(coordinates_l93)

        geometries.append(polygon)

    crop_ids = [
        f"{tile_id}_{index}"
        for index in indexes
    ]

    df = pd.DataFrame(
        {
            "confidence": confidences,
            "geometry": geometries,
            "crop_id": crop_ids,
            "tile": str(path_tile),
        }
    )

    return gpd.GeoDataFrame(
        df,
        geometry="geometry",
        crs=crs,
    ), n_crops

def spatial_nms(
    gdf: gpd.GeoDataFrame,
    overlap_threshold:float,
) -> gpd.GeoDataFrame:
    """
    Remove overlapping predictions produced from different image crops.

    Predictions are processed from largest to smallest. A candidate is suppressed
    when its intersection with a previously kept polygon covers at least
    ``overlap_threshold`` of the candidate's own area. Predictions from the same
    crop are not compared.


    Args:
        gdf: Raw georeferenced predictions.
        overlap_threshold: Minimum overlap ratio required for suppression.

    Returns:
        A copy of the GeoDataFrame containing the retained predictions.

    Raises:
        ValueError: If the geometry or crop identifier column is missing.
    """
    if gdf.empty:
        return gdf.copy()

    required_columns = {"geometry", "crop_id"}
    missing_columns = required_columns - set(gdf.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing_columns)}"
        )

    gdf_area = gdf.copy()
    gdf_area["area"] = gdf_area.geometry.area

    gdf_area = (
        gdf_area
        .sort_values("area", ascending=False)
        .reset_index(drop=True)
    )

    spatial_index = gdf_area.sindex

    suppressed = np.zeros(len(gdf_area), dtype=bool)
    kept_positions = []

    for i, geom_i in enumerate(gdf_area.geometry):
        if suppressed[i]:
            continue

        kept_positions.append(i)
        crop_i = gdf_area.iloc[i]["crop_id"]

        candidate_positions = spatial_index.query(
            geom_i,
            predicate="intersects",
        )

        for j in candidate_positions:
            if j <= i or suppressed[j]:
                continue

            crop_j = gdf_area.iloc[j]["crop_id"]

            if crop_i == crop_j:
                continue

            geom_j = gdf_area.geometry.iloc[j]

            if geom_j.area == 0:
                continue

            intersection_area = geom_i.intersection(geom_j).area

            if intersection_area == 0:
                continue

            overlap_ratio = intersection_area / geom_j.area

            if overlap_ratio >= overlap_threshold:
                suppressed[j] = True

    return (
        gdf_area.iloc[kept_positions]
        .drop(columns="area")
        .copy()
    )

def increment_path(path: str | Path, sep:str = "_") -> Path:
    """
    Return an available path by appending an incrementing suffix if needed.

    Args:
        path: Desired path.
        sep: Separator inserted before the numeric suffix.

    Returns:
        The original path if available, otherwise an incremented path.
    """
    path = Path(path)
    if path.exists():
        for n in range(2, 9999):
            p = Path(f"{path}{sep}{n}")
            if not p.exists():
                break
        path = p

    return path

def create_run_directory(save_dir: str | Path = "outputs") -> Path:
    """
    Create a uniquely named directory for a prediction run.

    The directory name is based on the current date and contains a
    subdirectory for logs.

    Args:
        save_dir: Root directory in which prediction runs are stored.

    Returns:
        Path to the newly created run directory.
    """
    today = date.today()
    year, month, day = today.year, today.month, today.day
    run_dir = Path(save_dir) / "prediction" / f"{year}-{month}-{day}"
    run_dir = increment_path(run_dir)

    run_dir.mkdir(parents=True)

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    return run_dir

def predict_and_save_tile(
        tile: Path,
        raw_preds_dir: Path,
        model: YOLO,
        window_size: int,
        window_overlap: float,
        predict_args: dict, 
        verbose: bool
    ) -> None:
    """
    Predict one tile and atomically save its raw GeoParquet output.

    Predictions are first written to a partial file. The final ``.parquet``
    file only becomes visible after the write completes successfully.

    Args:
        tile: JP2 tile to process.
        raw_preds_dir: Directory receiving raw prediction files.
        model: Loaded Ultralytics YOLO model.
        window_size: Crop width and height in pixels.
        window_overlap: Target overlap ratio between adjacent crops.
        predict_args: Additional arguments forwarded to ``model.predict``.
        verbose: Whether to display crop-generation information.
    """
    logger.info("Processing tile %s", str(tile.stem))
        

    start = time.perf_counter()
    tmp_path = raw_preds_dir / f"{tile.stem}.parquet.part"
    gdf, n_crops = predict_tile_obb(
        path_tile=tile,
        model=model,
        window_size=window_size,
        window_overlap=window_overlap,
        predict_args=predict_args,
        verbose=verbose
    )
    
    gdf.to_parquet(tmp_path)

    final_path = raw_preds_dir / f"{tile.stem}.parquet"
    # Expose the final file only after GeoParquet writing succeeds.
    tmp_path.replace(final_path)

    elapsed = time.perf_counter() - start

    logger.info(
        "Tile completed: %d crops, %d raw predictions, %.1f s, saved to %s",
        n_crops,
        len(gdf),
        elapsed,
        final_path.name,
    )

def run_raw_predictions(
        input_path: str | Path,
        run_dir: str | Path,
        model: YOLO,
        window_size: int,
        window_overlap: float,
        predict_args: dict | None = None,
        verbose: bool = False,
    ) -> None:
    """
    Generate and save raw predictions for every JP2 tile in a directory.

    Only JP2 files located directly inside ``input_path`` are processed. The
    prediction configuration is saved before processing starts.

    Args:
        input_path: Directory containing the JP2 tiles.
        run_dir: Directory associated with the prediction run.
        model: Loaded Ultralytics YOLO model.
        window_size: Crop width and height in pixels.
        window_overlap: Target overlap ratio between adjacent crops.
        predict_args: Arguments forwarded to ``model.predict``. Default
            arguments are used when this is ``None``.
        verbose: Whether to display crop-generation information.
    """
    run_dir = Path(run_dir)
    input_path = Path(input_path)

    if predict_args is None:
        predict_args = {
            "conf": 0.25,
            "save": False,
            "verbose": False
        }

    # Persist the inference configuration required to resume the run.
    today = date.today()
    metadata = dict()
    metadata["model_path"] = str(Path(model.model_name).resolve())
    metadata["window_size"] = window_size
    metadata["window_overlap"] = window_overlap
    metadata["input_path"] = str(input_path.resolve())
    metadata["date"] = str(today)
    metadata["predict_args"] = predict_args.copy()
    metadata["verbose"] = verbose

    metadata_path = run_dir / "metadata_prediction.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    # Only direct children are supported, keeping tile stems unique in the run.
    all_tiles = sorted(input_path.glob("*.jp2"))

    logger.info("Found %d JP2 tiles", len(all_tiles))

    raw_preds_dir = run_dir / "raw_predictions"
    raw_preds_dir.mkdir(exist_ok=True)

    for tile in all_tiles:

        predict_and_save_tile(
            tile=tile,
            raw_preds_dir=raw_preds_dir,
            model=model,
            window_size=window_size,
            window_overlap=window_overlap,
            predict_args=predict_args,
            verbose=verbose
        )


def resume_run(run_dir: str | Path) -> int | None:
    """
    Resume an interrupted prediction run from its saved metadata.

    Completed tiles are identified by their finalized GeoParquet files.
    Partial files are ignored, causing their tiles to be processed again.

    Args:
        run_dir: Directory of the interrupted prediction run.

    Returns:
        The number of tiles processed during the resumed execution, or
        ``None`` when the run cannot be resumed because its metadata or model
        is unavailable or invalid.

    Raises:
        ValueError: If the original input directory no longer exists.
    """
    run_dir = Path(run_dir)
    raw_preds_dir = run_dir / "raw_predictions"
    metadata_path = run_dir / "metadata_prediction.json"

    raw_preds_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)

    except FileNotFoundError:
        logger.error(
            "No metadata file found at the expected location %s. Resume is canceled",
            str(metadata_path)
        )
        return

    except json.JSONDecodeError:
        logger.error("Invalid prediction metadata JSON")
        return

    try:
        input_path = Path(metadata["input_path"])
        config = {
            "window_size": metadata["window_size"],
            "window_overlap": metadata["window_overlap"],
            "predict_args": metadata["predict_args"],
            "verbose": metadata["verbose"]
        }
        model_path = metadata["model_path"]

    except KeyError as error:
        logger.error("Missing prediction metadata key: %s", error)
        return

    if not input_path.exists():
        logger.warning(
            "Data source directory for the original run %s doesn't exists anymore",
            str(input_path)
        )
        raise ValueError

    # Finalized Parquet files are the source of truth; partial files are ignored.
    completed_tiles = set([tile.stem for tile in raw_preds_dir.glob("*.parquet")])
    all_tiles = sorted(input_path.glob("*.jp2"))
    missing_tiles = [tile for tile in all_tiles if tile.stem not in completed_tiles]

    if len(missing_tiles) == 0:
        return 0

    try:
        model = YOLO(model_path)

    except FileNotFoundError:
        logger.error(
            "No YOLO model found at %s", metadata["model_path"]
        )
        return

    logger.info("Resuming prediction on %d JP2 files", len(missing_tiles))
    for tile in missing_tiles:
        predict_and_save_tile(
            tile=tile,
            raw_preds_dir=raw_preds_dir,
            model=model,
            **config
        )

    return len(missing_tiles)

def save_results_after_nms(
        run_dir: str | Path,
        nms_threshold: float
    ) -> Path | None:
    """
    Apply global spatial NMS to completed raw prediction files.

    This step reads only finalized GeoParquet files and does not invoke the
    YOLO model.

    Args:
        run_dir: Directory containing the raw predictions.
        nms_threshold: Overlap threshold passed to the spatial NMS.

    Returns:
        Path to the generated GeoPackage, or ``None`` when the completed tiles
        contain no predictions.

    Raises:
        ValueError: If no completed raw prediction file exists.
    """
    run_dir = Path(run_dir)
    raw_predictions_dir = run_dir / "raw_predictions"
    final_result_dir = run_dir / "final"
    final_result_dir.mkdir(exist_ok=True)

    # Post-processing relies exclusively on completed per-tile outputs.
    l_gdf = []
    for file in raw_predictions_dir.glob("*.parquet"):
        l_gdf.append(gpd.read_parquet(file))

    if not l_gdf:
        raise ValueError("No completed raw prediction files found")

    all_predictions = gpd.GeoDataFrame(
        pd.concat(l_gdf, ignore_index=True),
        geometry="geometry",
        crs=l_gdf[0].crs,
    )

    if len(all_predictions) == 0:
        logger.info("Zero predictions made on these tiles, skipping NMS")
        return 

    logger.info("Starting global NMS on %d predictions", len(all_predictions))
    start = time.perf_counter()

    gdf_after_nms = spatial_nms(all_predictions, nms_threshold)

    elapsed = time.perf_counter() - start
    logger.info(
        "NMS completed: %d predictions kept, %.1f s",
        len(gdf_after_nms),
        elapsed,
    )

    results_filename = final_result_dir / "predictions_obb.gpkg"

    gdf_after_nms.to_file(
        results_filename,
        driver="GPKG",        
    )
    logger.info("Final GeoPackage save : %s", str(results_filename))

    today = date.today()
    metadata = dict()
    metadata["date"] = str(today)
    metadata["nms_threshold"] = nms_threshold
    metadata["gpkg"] = str(results_filename.resolve())

    metadata_path = run_dir / "metadata_nms.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    return results_filename

def main():
    """Run the complete prediction and post-processing pipeline."""
    try:
        input_path = Path("data/benchmark_yield/test_dalles")
        
        run_dir = create_run_directory("outputs")

        configure_logging(run_dir / "logs" / "predict.log")

        model = YOLO("models/yolo26n-obb/v4_1024_100e/weights/best.pt")
        logger.info("Run started")

        start = time.perf_counter()

        run_raw_predictions(
            input_path=input_path, 
            run_dir=run_dir, 
            model=model, 
            window_size=2048, 
            window_overlap=0.2
        )
        elapsed = time.perf_counter() - start
        logger.info("Processed all tiles in %.1f s", elapsed)

        save_results_after_nms(run_dir=run_dir, nms_threshold=0.1)

        elapsed = time.perf_counter() - start
        logger.info("Run completed in %.1f s", elapsed)

    except KeyboardInterrupt:
        logger.warning("Run interrupted by user")
        raise
    except Exception:
        logger.exception("Run failed")
        raise

def resume_main():
    """Resume raw prediction generation for an existing run."""
    try:
        run_dir = Path("outputs/prediction/2026-9-5")
        configure_logging(run_dir / "logs" / "predict.log")
        logger.info("Resuming run %s", str(run_dir))

        start = time.perf_counter()

        n_tiles = resume_run(run_dir)

        elapsed = time.perf_counter() - start

        if n_tiles is not None:
            logger.info(
                "Resume completed : processed %d tiles in %.1f s",
                n_tiles,
                elapsed
            )

        if n_tiles == 0:
            logger.info("Run already completed, no new tile to process")

    except KeyboardInterrupt:
        logger.warning("Run interrupted by user")
        raise
    except Exception:
        logger.exception("Run failed")
        raise

if __name__ == "__main__":
    main()

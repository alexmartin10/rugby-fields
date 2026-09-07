import time
import json
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from datetime import date
from ultralytics import YOLO
from shapely.geometry import Polygon

from .jp2_to_jpg import jp2_tile_to_jpg

logger = logging.getLogger(__name__)

def configure_logging(log_path: str | Path) -> None:
    log_path = Path(log_path)

    # Make parent directory if it doesn't exist
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
    path_save_jpg: str | Path,
    model: YOLO,
    window_size: int,
    window_overlap: float,
    predict_args: dict,
    verbose: bool = False
) -> tuple[gpd.GeoDataFrame, int]:
    path_tile = Path(path_tile)
    tile_id = path_tile.stem
    path_save_jpg = Path(path_save_jpg)
    path_save_jpg.mkdir(parents=True, exist_ok=True)

    for jpg_path in path_save_jpg.glob("*.jpg"):
        jpg_path.unlink()

    transform, crs, dict_index_pixels = jp2_tile_to_jpg(
        path_to_jp2=path_tile,
        path_save=path_save_jpg,
        window_size=window_size,
        overlap=window_overlap,
        verbose=verbose
    )

    results = model.predict(
        source=path_save_jpg,
        **predict_args
    )

    fields_pixels = np.empty((0, 4, 2), dtype=float)
    confidence = np.empty(0, dtype=float)
    indexes = []

    for result in results:
        if result.obb is None or result.obb.data.numel() == 0: # no prediction
            continue

        index = int(Path(result.path).stem)
        indexes.extend([index] * len(result.obb))

        row_start, col_start = dict_index_pixels[index]

        # Shape : (nombre de boxes, 4 sommets, 2 coordonnées)
        pixels_tile = result.obb.xyxyxyxy.numpy().copy()

        # Décalage de tous les sommets vers le référentiel de la dalle
        pixels_tile[:, :, 0] += col_start
        pixels_tile[:, :, 1] += row_start

        fields_pixels = np.concatenate(
            [fields_pixels, pixels_tile],
            axis=0,
        )

        confidence = np.concatenate(
            [confidence, result.obb.conf.numpy()],
        )

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
            "confidence": confidence,
            "geometry": geometries,
            "crop_id": crop_ids,
            "tile": str(path_tile),
        }
    )

    return gpd.GeoDataFrame(
        df,
        geometry="geometry",
        crs=crs,
    ), len(dict_index_pixels)

def spatial_nms(
    gdf: gpd.GeoDataFrame,
    overlap_threshold:float,
) -> gpd.GeoDataFrame:
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
    path = Path(path)
    if path.exists():
        for n in range(2, 9999):
            p = Path(f"{path}{sep}{n}")
            if not p.exists():
                break
        path = p

    return path

def create_run_directory(save_dir: str | Path = "outputs") -> Path:
    #create output dir
    today = date.today()
    year, month, day = today.year, today.month, today.day
    run_dir = Path(save_dir) / "prediction" / f"{year}-{month}-{day}"
    run_dir = increment_path(run_dir)

    run_dir.mkdir(parents=True)

    #/logs
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    return run_dir

def predict_and_save_tile(
        tile: Path,
        raw_preds_dir: Path,
        path_save_jpg: str | Path,
        model: YOLO,
        window_size: int,
        window_overlap: float,
        predict_args: dict, 
        verbose: bool
    ) -> None:
    """
    config expects the arguments to be given to predict_tile_obb :
    path_save_jpg, window_size, window_overlap, predict_args, verbose
    """
    logger.info("Processing tile %s", str(tile.stem))
        

    start = time.perf_counter()
    tmp_path = raw_preds_dir / f"{tile.stem}.parquet.part"
    gdf, n_crops = predict_tile_obb(
        path_tile=tile,
        path_save_jpg=path_save_jpg,
        model=model,
        window_size=window_size,
        window_overlap=window_overlap,
        predict_args=predict_args,
        verbose=verbose
    )
    
    gdf.to_parquet(tmp_path)

    final_path = raw_preds_dir / f"{tile.stem}.parquet"
    tmp_path.replace(final_path) #renamed after writing is over

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
        path_save_jpg: str | Path,
        predict_args: dict | None = None,
        verbose: bool = False,
    ) -> None:
    """
    JP2 files expected to be read must all be directly in the input_path directory.
    """
    run_dir = Path(run_dir)
    input_path = Path(input_path)

    if predict_args is None:
        predict_args = {
            "conf": 0.25,
            "stream": True,
            "batch": 8,
            "save": False,
            "verbose": False
        }

    #save run metadata: model, conf, imgsz, overlap, source, date, ...
    today = date.today()
    metadata = dict()
    metadata["model_path"] = str(Path(model.model_name).resolve())
    metadata["window_size"] = window_size
    metadata["window_overlap"] = window_overlap
    metadata["input_path"] = str(input_path.resolve())
    metadata["path_save_jpg"] = str(Path(path_save_jpg).resolve())
    metadata["date"] = str(today)
    metadata["predict_args"] = predict_args.copy()
    metadata["verbose"] = verbose

    metadata_path = run_dir / "metadata_prediction.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    #save one raw prediction by tile, in geoparquet, in dir raw_predictions
    all_tiles = sorted(input_path.glob("*.jp2"))

    logger.info("Found %d JP2 tiles", len(all_tiles))

    raw_preds_dir = run_dir / "raw_predictions"
    raw_preds_dir.mkdir(exist_ok=True)

    for tile in all_tiles:

        predict_and_save_tile(
            tile=tile,
            raw_preds_dir=raw_preds_dir,
            path_save_jpg=path_save_jpg,
            model=model,
            window_size=window_size,
            window_overlap=window_overlap,
            predict_args=predict_args,
            verbose=verbose
        )


def resume_run(run_dir: str | Path) -> int | None:
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
            "path_save_jpg": metadata["path_save_jpg"],
            "window_size": metadata["window_size"],
            "window_overlap": metadata["window_overlap"],
            "predict_args": metadata["predict_args"],
            "verbose": metadata["verbose"]
        }
        model_path = metadata["model_path"]

    except KeyError as error:
        logger.error("Missing prediction metadata key: %s", error)
        return

    if not input_path.is_dir():
        logger.warning(
            "Data source directory for the original run %s doesn't exists anymore",
            str(input_path)
        )
        raise ValueError

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
    run_dir = Path(run_dir)
    raw_predictions_dir = run_dir / "raw_predictions"
    final_result_dir = run_dir / "final"
    final_result_dir.mkdir(exist_ok=True)

    #read parquet files and concatenate them in a gdf
    l_gdf = []
    for file in raw_predictions_dir.glob("*.parquet"): # only read completed files
        l_gdf.append(gpd.read_parquet(file))

    if not l_gdf: # in the case no finished parquet file exists
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

    #final: final geopkg, after spatial_nms
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

    #save nms metadata
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
    try:
        input_path = Path("data/raw/D33/test_dalles")
        
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
            window_overlap=0.2, 
            path_save_jpg="data/raw/D33/dalle_jpg"
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
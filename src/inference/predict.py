from ultralytics import YOLO
from pathlib import Path
import geopandas as gpd
import pandas as pd
import re
import numpy as np
from shapely.geometry import Polygon
import time
from datetime import date
import json
import os
import logging

from .jp2_to_jpg import jp2_tile_to_jpg

np.set_printoptions(suppress=True, precision=10)

logger = logging.getLogger(__name__)


def predict_tile_obb(
    path_tile: Path,
    path_save_jpg,
    model: YOLO,
    window_size: int,
    window_overlap: float,
    predict_args: dict,
    verbose: bool = False
) -> tuple[gpd.GeoDataFrame, int]:
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
        if result.obb is None or result.obb.data.numel() == 0:
            continue

        match = re.search(r"/(\d+)\.jpg$", result.path)

        if match is None:
            raise ValueError(
                f"Impossible de récupérer l'index du crop depuis : {result.path}"
            )

        index = int(match.group(1))
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

def create_run_directory(save_dir: str | Path = "outputs"):
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

def run_raw_predictions(
        input_path: str | Path,
        run_dir: str | Path,
        model: YOLO,
        window_size: int,
        window_overlap: float,
        path_save_jpg: str | Path,

        predict_args: dict = {
            "conf": 0.25,
            "stream": True,
            "batch": 8,
            "save": False,
            "verbose": False
        },
):
    run_dir = Path(run_dir)
    input_path = Path(input_path)

    #save run metadata: model, conf, imgsz, overlap, source, date, ...
    today = date.today()
    metadata = dict()
    metadata["model_path"] = model.model_name
    metadata["window_size"] = window_size
    metadata["window_overlap"] = window_overlap
    metadata["input_path"] = str(input_path)
    metadata["date"] = str(today)
    metadata["predict_args"] = predict_args.copy()

    metadata_path = run_dir / "metadata_prediction.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    #save one raw prediction by tile, in geoparquet, in dir raw_predictions
    all_tiles = sorted(input_path.rglob("*.jp2"))

    logger.info(f"Found {len(all_tiles)} JP2 tiles")

    raw_preds_dir = run_dir / "raw_predictions"
    raw_preds_dir.mkdir(exist_ok=True)

    for tile in all_tiles:
        logger.info(f"Processing tile {str(tile.stem)}")
        start = time.perf_counter()

        tmp_path = raw_preds_dir / f"{tile.stem}.parquet.part"
        gdf, n_crops = predict_tile_obb(
            tile, 
            path_save_jpg, 
            model, 
            window_size,
            window_overlap,
            predict_args
        )
        elapsed = time.perf_counter() - start
        logger.info(f"Tile completed : {n_crops} crops, {len(gdf)} raw predictions, {elapsed:.1f}s")

        gdf.to_parquet(tmp_path)
        os.rename(str(tmp_path), str(raw_preds_dir / f"{tile.stem}.parquet")) #renamed after writing is over
        logger.info(f"Raw prediction saved : {str(raw_preds_dir / f"{tile.stem}.parquet")}")


def save_results_after_nms(
        run_dir: str | Path,
        nms_threshold: float
):
    run_dir = Path(run_dir)
    raw_predictions_dir = run_dir / "raw_predictions"
    final_result_dir = run_dir / "final"
    final_result_dir.mkdir(exist_ok=True)

    #read parquet files and concatenate them in a gdf
    l_gdf = []
    for file in raw_predictions_dir.glob("*.parquet"): # only read completed files
        l_gdf.append(gpd.read_parquet(file))

    all_predictions = gpd.GeoDataFrame(
        pd.concat(l_gdf, ignore_index=True),
        geometry="geometry",
        crs=l_gdf[0].crs,
    )
    logger.info(f"Starting global NMS on {len(all_predictions)} predictions")
    start = time.perf_counter()

    #final: final geopkg, after spatial_nms
    gdf_after_nms = spatial_nms(all_predictions, nms_threshold)

    elapsed = time.perf_counter() - start
    logger.info(f"NMS completed : {len(gdf_after_nms)} predictions kept, {elapsed:.1f}s")

    gdf_after_nms.to_file(
        str(final_result_dir / "predictions_obb.gpkg"),
        driver="GPKG",        
    )
    logger.info(f"Final GeoPackage save : {str(final_result_dir / 'predictions_obb.gpkg')}")

    #save nms metadata
    today = date.today()
    metadata = dict()
    metadata["date"] = str(today)
    metadata["nms_threshold"] = nms_threshold
    metadata["gpkg"] = str(final_result_dir / "predictions_obb.gpkg")

    metadata_path = run_dir / "metadata_nms.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

def main():
    input_path = Path("data/raw/D33/test_dalles")
    model = YOLO("models/yolo26n-obb/v4_1024_100e/weights/best.pt")

    run_dir = create_run_directory("outputs")

    logging.basicConfig(
        filename=run_dir / "logs" / "predict.log",
        level=logging.INFO,
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    logger.info(f"Run started")

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
    logger.info(f"Processed all tiles in {elapsed:.1f}s")

    save_results_after_nms(run_dir=run_dir, nms_threshold=0.1)

    elapsed = time.perf_counter() - start
    logger.info(f"Run completed in {elapsed:.1f}s")

if __name__ == "__main__":
    main()
from ultralytics import YOLO
from pathlib import Path
import geopandas as gpd
import pandas as pd
import re
import numpy as np
from shapely.geometry import box

from jp2_to_jpg import convert_jp2_tile_to_jpg

np.set_printoptions(suppress=True, precision=10)

model = YOLO("models/yolo26n/v1_1024_100e/best.pt")

def predict_tile(path_tile: Path, path_save_jpg, model: YOLO, window_size):
    tile_id = path_tile.stem
    path_save_jpg = Path(path_save_jpg)
    path_save_jpg.mkdir(parents=True, exist_ok=True)

    for jpg_path in path_save_jpg.glob("*.jpg"):
        jpg_path.unlink()
        
    transform, crs, dict_index_pixels = convert_jp2_tile_to_jpg(path_tile, path_save_jpg, window_size)

    results = model.predict(
        source=path_save_jpg,
        conf=0.25,
        save=False,
    )

    fields_pixels = np.empty((0, 4))
    confidence = np.empty((0))
    indexes = []
    for r in results:
        if r.boxes.data.numel() != 0:
            #field detected in the image
            #get the index of the image
            match = re.search(r"/(\d+)\.jpg$", r.path)
            index = int(match.group(1))
            indexes += [index] * len(r.boxes)

            #get bound pixels for the cropped image
            #Attention : row corresponds to y coordinate, col to x
            row_start, col_start = dict_index_pixels[index]
            
            pixels_tile = r.boxes.xyxy.numpy() #(x1, y1, x2, y2)
            #add row_start and col_start to have pixels in the tile "referential"
            pixels_tile[:, 0::2] += col_start
            pixels_tile[:, 1::2] += row_start

            fields_pixels = np.concatenate([fields_pixels, pixels_tile])
            confidence = np.concatenate([confidence, r.boxes.conf.numpy()])
    
    geometries = []
    coordinates_L93 = fields_pixels.copy()
    for field in coordinates_L93:
        field[0:2] = transform * field[0:2]
        field[2:] = transform * field[2:]
        geometries.append(box(*field))
    crops_ids = [f"{tile_id}_{index}" for index in indexes]
    df = pd.DataFrame({"confidence": confidence, "geometry": geometries, "crop_id": crops_ids, "tile": str(path_tile)})

    return gpd.GeoDataFrame(df, geometry="geometry", crs=crs)

def spatial_nms(
    gdf: gpd.GeoDataFrame,
    overlap_threshold: float = 0.6,
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

def predict_all_tiles(orthophotos_dir, tmp_dir, model, window_size):
    orthophotos_dir = Path(orthophotos_dir).resolve()
    all_tiles = orthophotos_dir.rglob("*.jp2")

    gdfs = []

    for tile in all_tiles:
        gdf = predict_tile(tile, tmp_dir, model, window_size)
        gdfs.append(gdf)

    if not gdfs:
        return gpd.GeoDataFrame(
            columns=["confidence", "geometry", "crop_id", "tile"],
            geometry="geometry",
        )

    all_predictions = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        geometry="geometry",
        crs=gdfs[0].crs,
    )

    return spatial_nms(all_predictions)

def main():
    base = Path().resolve()
    path_to_jp2 = base / "data/raw/D33/test_dalles"
    path_save_jpg = base / "data/raw/D33/dalle_jpg"
    gdf = predict_all_tiles(path_to_jp2, path_save_jpg, model, window_size=2048)
    gdf.to_file(
    "predictions.gpkg",
    driver="GPKG",
    )

if __name__ == "__main__":
    main()
import geopandas as gpd
import numpy as np

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

        candidate_positions = spatial_index.query(
            geom_i,
            predicate="intersects",
        )

        for j in candidate_positions:
            if j <= i or suppressed[j]:
                print(f"field {i} stopped at j<i for candidate {j}")
                continue

            geom_j = gdf_area.geometry.iloc[j]

            if geom_j.area == 0:
                print(f"field {i} stopped at area == 0 for candidate {j}")
                continue

            intersection_area = geom_i.intersection(geom_j).area

            if intersection_area == 0:
                print(f"field {i} stopped at intersection == 0 for candidate {j}")
                continue

            overlap_ratio = intersection_area / geom_j.area

            if overlap_ratio >= overlap_threshold:
                print(f"field {i} stopped at overlap OK for candidate {j}")
                suppressed[j] = True

    return (
        gdf_area.iloc[kept_positions]
        .drop(columns="area")
        .copy()
    )


path = "predictions.gpkg"

gdf = gpd.read_file(path)
print(gdf.shape)

gdf_after = spatial_nms(gdf)
print(gdf_after.shape)

gdf_after.to_file(
    "predictions.gpkg",
    driver="GPKG",
    )
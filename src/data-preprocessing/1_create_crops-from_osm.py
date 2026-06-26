import rasterio
import geopandas

dataset = rasterio.open("31-2025-0490-6180-LA93-0M20-E080.jp2")

print(dataset.width)
print(dataset.height)
print(dataset.count)
print(dataset.bounds)
print(dataset.transform)

gdf = geopandas.read_file("/home/alexandre-martin/Documents/rugby-fields/data/raw/osm/export.geojson")

print(gdf.columns)
print(gdf[["sport", "leisure", "geometry"]].head())
print(gdf["sport"].value_counts(dropna=False))
import pandas as pd
import geopandas as gpd
import os

from datafill.config import PARSED_DATA_PATH, RIVER_DATA_PATH, RIVER_SHP_PATH
from datafill.data import parse_data

TARGET_CRS = "EPSG:32748"

def parse_river():
    if not os.path.exists(PARSED_DATA_PATH):
        print("Parsed data file not found. Running initial parse...")
        parse_data()

    df = pd.read_csv(PARSED_DATA_PATH)
    geometries = gpd.points_from_xy(df['long'], df['lat'])
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    gdf = gdf.to_crs(TARGET_CRS)
    river_gdf = gpd.read_file(RIVER_SHP_PATH)
    river_gdf = river_gdf.to_crs(TARGET_CRS)

    unified_river = river_gdf.union_all()

    gdf['distance_to_river'] = gdf.geometry.apply(lambda geo: geo.distance(unified_river))
    
    gdf.drop(columns="geometry").to_csv(RIVER_DATA_PATH, index=False)
    print(f"Successfully saved rain data with river distances to {RIVER_DATA_PATH}")


if __name__ == "__main__":
    parse_river()

import pandas as pd
import geopandas as gpd
import os

from trainer.config import CRS_TARGET, PARSED_DATA_PATH, RIVER_DATA_PATH, RIVER_SHP_PATH
from trainer.datafill.data import parse_data

def parse_river():
    if not os.path.exists(PARSED_DATA_PATH):
        print("Parsed data file not found. Running initial parse...")
        parse_data()

    df = pd.read_csv(PARSED_DATA_PATH)
    geometries = gpd.points_from_xy(df['long'], df['lat'])
    # CRS should be auto-detected from the shapefile
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326").to_crs(CRS_TARGET)
    river_gdf = gpd.read_file(RIVER_SHP_PATH).to_crs(CRS_TARGET)

    unified_river = river_gdf.union_all()

    gdf['distance_to_river'] = gdf.geometry.apply(lambda geo: geo.distance(unified_river))
    
    gdf.drop(columns="geometry").to_csv(RIVER_DATA_PATH, index=False)
    print(f"Successfully saved rain data with river distances to {RIVER_DATA_PATH}")


if __name__ == "__main__":
    parse_river()

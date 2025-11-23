import os
import pandas as pd
import geopandas as gpd
from shapely import Polygon

from datafill.config import RAW_DATA_PATH, PARSED_DATA_PATH

TARGET_CRS = "EPSG:32748"
MAX_CELL_SIZE = 10_000

def parse_data():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['long'], df['lat']),
        crs="EPSG:4326"
    )
    
    xmin = 94.12166580738058
    ymin = -7.620859226011504
    xmax = 107.34095494083526
    ymax = 7.0052286958747345
    aoi = Polygon([
        (xmin, ymin), (xmax, ymin),
        (xmax, ymax), (xmin, ymax)
    ])
    gdf_sub = gdf[gdf.geometry.within(aoi)]
    gdf_balanced = (
        gdf_sub
        .groupby("banjir", group_keys=False)
        .apply(lambda x: x.sample(gdf_sub["banjir"].value_counts().min(), random_state=42))
    )
    gdf_balanced.drop(columns="geometry").to_csv(PARSED_DATA_PATH, index=False)
    print("Parsed data saved to", PARSED_DATA_PATH)


if __name__ == "__main__":
    parse_data()
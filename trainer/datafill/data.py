import os

import geopandas as gpd
import pandas as pd
from shapely import Polygon

from trainer.config import (
    PARSED_DATA_PATH,
    RAW_DATA_PATH,
    SELECTED_FEATURES,
    TARGET_FEATURE,
)


def parse_data():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    df["landcover_class"] = df["landcover_class"].astype("category").cat.codes
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["long"], df["lat"]), crs="EPSG:4326"
    )

    cols = ["long", "lat"] + SELECTED_FEATURES + [TARGET_FEATURE, "geometry"]
    cols = [c for c in cols if c in gdf.columns]
    gdf = gdf[cols]

    xmin = 94.5
    ymin = -11.5
    xmax = 141.5
    ymax = 6.5
    aoi = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

    gdf_sub = gdf[gdf.geometry.within(aoi)]
    gdf_sub.drop(columns="geometry").to_csv(PARSED_DATA_PATH, index=False)
    print("Parsed data saved to", PARSED_DATA_PATH)


if __name__ == "__main__":
    parse_data()

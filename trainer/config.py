CRS_TARGET = "EPSG:32748"  # UTM zone 48S

DATA_PATH = "data"
DEM_PROJECTED_PATH = "dem_tiles_utm"

RAW_DATA_PATH = f"{DATA_PATH}/data_banjir.csv"
PARSED_DATA_PATH = f"{DATA_PATH}/parsed_data_banjir.csv"
RIVER_DATA_PATH = f"{DATA_PATH}/river_data_banjir.csv"
RIVER_SHP_PATH = f"{DATA_PATH}/rivers.shp"
JABODETABEK_SHP_PATH = f"{DATA_PATH}/jabodetabek_bound.shp"

OUT_MODEL_PATH = f"{DATA_PATH}/logreg_best.pkl"

SELECTED_FEATURES = [
    "avg_rainfall",
    "elevation",
    "slope",
    "distance_to_river",
]
TARGET_FEATURE = "banjir"



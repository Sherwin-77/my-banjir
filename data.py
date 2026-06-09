import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from trainer.config import (
    JABODETABEK_SHP_PATH,
    OUT_MODEL_PATH,
    RAW_DATA_PATH,
    SELECTED_FEATURES,
)

df = pd.read_csv(RAW_DATA_PATH)
grouped_name2 = df.groupby("NAME_2")

print("Unique Kabupaten/Kota:")
print(grouped_name2.size())

points = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["long"], df["lat"]), crs="EPSG:4326"
)
jabodetabek_bound = gpd.read_file(JABODETABEK_SHP_PATH)

if jabodetabek_bound.crs is None:
    jabodetabek_bound = jabodetabek_bound.set_crs(points.crs)

points = points.to_crs(jabodetabek_bound.crs)
bound_area = jabodetabek_bound.union_all()
in_bound_mask = points.geometry.intersects(bound_area)

in_bound_count = int(in_bound_mask.sum())
out_bound_count = int((~in_bound_mask).sum())

print(f"In Jabodetabek bound: {in_bound_count}")
print(f"Outside Jabodetabek bound: {out_bound_count}")

model = joblib.load(OUT_MODEL_PATH)
scaler = model.named_steps["scaler"]
poly = model.named_steps["poly"]
clf = model.named_steps["clf"]

coefs = clf.coef_.ravel()
scale = scaler.scale_
powers = poly.powers_

n_orig = len(SELECTED_FEATURES)

rows = []
for pi in range(len(coefs)):
    parts = []
    scaling = 1.0
    for j in range(n_orig):
        exp = powers[pi, j]
        if exp > 0:
            if exp == 1:
                parts.append(SELECTED_FEATURES[j])
            else:
                parts.append(f"{SELECTED_FEATURES[j]}^{exp}")
            scaling *= scale[j] ** exp
    if not parts:
        continue
    name = " * ".join(parts)
    coeff = coefs[pi] / scaling
    rows.append({"feature": name, "coefficient": coeff})

imp_df = pd.DataFrame(rows).sort_values("coefficient", key=abs, ascending=False)

print("\nFull polynomial feature coefficients (rescaled to original units):")
print(imp_df.to_string(index=False))

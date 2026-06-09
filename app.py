import io
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import altair as alt
import folium
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from folium.plugins import MarkerCluster
from pyproj import Transformer
from rasterio.warp import transform, transform_bounds
from shapely.geometry import Point
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from streamlit_folium import st_folium

from trainer.config import (
    CRS_TARGET,
    DEM_PROJECTED_PATH,
    OUT_MODEL_PATH,
    RAW_DATA_PATH,
    RIVER_DATA_PATH,
    RIVER_SHP_PATH,
    SELECTED_FEATURES,
    TARGET_FEATURE,
)

APP_TITLE = "Pemetaan Kerawanan Banjir - Flood Susceptibility Mapping"

INDONESIA_BOUNDS = {
    "lon_min": 94.5,
    "lat_min": -11.5,
    "lon_max": 141.5,
    "lat_max": 6.5,
}
MAP_CENTER = [-2.5, 118.0]
MAP_ZOOM = 5

EVAL_TEST_SIZE = 0.3
EVAL_RANDOM_STATE = 42


RESULT_COLUMNS = [
    "Point_ID",
    "latitude",
    "longitude",
    "avg_rainfall",
    "elevation",
    "slope",
    "distance_to_river",
    "Result",
    "Confidence %",
]
EDITABLE_COLUMNS = ["avg_rainfall", "elevation", "slope", "distance_to_river"]
NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "avg_rainfall",
    "elevation",
    "slope",
    "distance_to_river",
    "Confidence %",
]

COORDINATE_COLUMN_CANDIDATES = {
    "lat": ["lat", "latitude", "y", "y_coord"],
    "lon": ["long", "lon", "longitude", "x", "x_coord"],
}


class DemTileInfo(TypedDict):
    path: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


class DatasetCoordinateColumns(TypedDict):
    lat: str
    lon: str


def _empty_results_df() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def load_raw_dataset() -> pd.DataFrame:
    dataset_path = Path(RAW_DATA_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at '{dataset_path}'. Expected flood dataset CSV."
        )
    return pd.read_csv(dataset_path)


@st.cache_data(show_spinner=False)
def load_training_dataset() -> pd.DataFrame:
    dataset_path = Path(RIVER_DATA_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Model evaluation dataset not found at '{dataset_path}'. "
            "Generate it first (trainer.datafill.river.parse_river)."
        )
    return pd.read_csv(dataset_path)


def _prepare_eval_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    required_cols = [*SELECTED_FEATURES, TARGET_FEATURE]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns for evaluation: " + ", ".join(missing)
        )

    data = df[required_cols].copy()
    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=required_cols).reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid rows found after cleaning evaluation dataset.")

    data[TARGET_FEATURE] = data[TARGET_FEATURE].astype(int)
    data = data[data[TARGET_FEATURE].isin([0, 1])].reset_index(drop=True)
    if data.empty:
        raise ValueError("Target column must contain binary labels 0/1.")

    y = data[TARGET_FEATURE]
    if y.nunique() < 2:
        raise ValueError("Evaluation dataset must contain both classes (0 and 1).")

    return data[SELECTED_FEATURES], y


@st.cache_data(show_spinner=False)
def compute_model_evaluation_artifacts() -> dict:
    raw_df = load_training_dataset()
    x, y = _prepare_eval_data(raw_df)

    min_class_count = int(y.value_counts().min())
    stratify_arg = y if min_class_count >= 2 else None

    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=EVAL_TEST_SIZE,
        random_state=EVAL_RANDOM_STATE,
        stratify=stratify_arg,
    )

    if y_test.nunique() < 2:
        raise ValueError(
            "Test split contains only one class; unable to compute ROC AUC. "
            "Please use a larger/balanced dataset."
        )

    model = load_model()
    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba.")

    y_score_raw = model.predict_proba(x_test)
    if np.asarray(y_score_raw).ndim == 2 and np.asarray(y_score_raw).shape[1] > 1:
        y_score = np.asarray(y_score_raw)[:, 1]
    else:
        y_score = np.asarray(y_score_raw).ravel()

    y_pred = model.predict(x_test)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = float(roc_auc_score(y_test, y_score))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    report_rows = []
    label_map = {
        "0": "Not Flood (0)",
        "1": "Flood (1)",
        "accuracy": "Accuracy",
        "macro avg": "Macro Avg",
        "weighted avg": "Weighted Avg",
    }
    preferred_order = ["0", "1", "accuracy", "macro avg", "weighted avg"]
    keys = [k for k in preferred_order if k in report] + [
        k for k in report.keys() if k not in preferred_order
    ]

    total_support = int(
        round(float(report.get("weighted avg", {}).get("support", len(y_test))))
    )

    for key in keys:
        val = report[key]
        if isinstance(val, dict):
            precision = float(val.get("precision", 0.0))
            recall = float(val.get("recall", 0.0))
            f1 = float(val.get("f1-score", 0.0))
            support = int(round(float(val.get("support", 0.0))))
        else:
            precision = float(val)
            recall = float(val)
            f1 = float(val)
            support = total_support

        report_rows.append(
            {
                "Class": label_map.get(str(key), str(key)),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(f1, 4),
                "Support": support,
            }
        )

    report_df = pd.DataFrame(report_rows)

    roc_df = pd.DataFrame(
        {
            "False Positive Rate": np.asarray(fpr, dtype=float),
            "True Positive Rate": np.asarray(tpr, dtype=float),
        }
    )

    cm_df = pd.DataFrame(
        {
            "Actual": [
                "Not Flood (0)",
                "Not Flood (0)",
                "Flood (1)",
                "Flood (1)",
            ],
            "Predicted": [
                "Not Flood (0)",
                "Flood (1)",
                "Not Flood (0)",
                "Flood (1)",
            ],
            "Count": [
                int(cm[0, 0]),
                int(cm[0, 1]),
                int(cm[1, 0]),
                int(cm[1, 1]),
            ],
        }
    )

    accuracy = float((np.asarray(y_pred) == np.asarray(y_test)).mean())

    return {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "test_rows": int(len(y_test)),
        "positive_rows": int((y_test == 1).sum()),
        "roc_df": roc_df,
        "cm_df": cm_df,
        "report_df": report_df,
    }


def _fmt_float(value, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return f"{0.0:.{digits}f}"


@st.cache_resource(show_spinner=False)
def load_model():
    model_path = Path(OUT_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Train or copy the model first."
        )
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def load_river_geometry():
    river_path = Path(RIVER_SHP_PATH)
    if not river_path.exists():
        return None

    try:
        river_gdf = gpd.read_file(river_path)
        if river_gdf.empty:
            return None
        river_gdf = river_gdf.to_crs(CRS_TARGET)
        if hasattr(river_gdf, "union_all"):
            return river_gdf.union_all()
        return river_gdf.unary_union
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_lonlat_transformer() -> Transformer:
    return Transformer.from_crs("EPSG:4326", CRS_TARGET, always_xy=True)

@st.cache_resource(show_spinner=False)
def get_dem_tile_index() -> List[DemTileInfo]:
    dem_dir = Path(DEM_PROJECTED_PATH)
    if not dem_dir.exists():
        return []

    tiles: List[DemTileInfo] = []
    for tile_path in sorted(dem_dir.glob("*.tif")):
        try:
            with rasterio.open(tile_path) as src:
                if src.crs is None:
                    continue
                lon_min, lat_min, lon_max, lat_max = transform_bounds(
                    src.crs,
                    "EPSG:4326",
                    src.bounds.left,
                    src.bounds.bottom,
                    src.bounds.right,
                    src.bounds.top,
                    densify_pts=21,
                )
                tiles.append(
                    {
                        "path": str(tile_path),
                        "lon_min": float(min(lon_min, lon_max)),
                        "lon_max": float(max(lon_min, lon_max)),
                        "lat_min": float(min(lat_min, lat_max)),
                        "lat_max": float(max(lat_min, lat_max)),
                    }
                )
        except Exception:
            continue
    return tiles


def ensure_state() -> None:
    if "results_df" not in st.session_state:
        st.session_state.results_df = _empty_results_df()
    if "pending_point" not in st.session_state:
        st.session_state.pending_point = None
    if "last_click_signature" not in st.session_state:
        st.session_state.last_click_signature = None
    if "resources_ready" not in st.session_state:
        st.session_state.resources_ready = False


def _normalize_point_ids(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "Point_ID" not in data.columns:
        data["Point_ID"] = ""

    current = data["Point_ID"].fillna("").astype(str).str.strip().tolist()
    used = set()
    normalized = []
    next_num = 1

    for point_id in current:
        if point_id and point_id not in used:
            normalized.append(point_id)
            used.add(point_id)
            continue

        while f"P-{next_num}" in used:
            next_num += 1
        generated = f"P-{next_num}"
        normalized.append(generated)
        used.add(generated)
        next_num += 1

    data["Point_ID"] = normalized
    return data


def _normalize_result_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_results_df()

    data = df.copy()
    for col in RESULT_COLUMNS:
        if col not in data.columns:
            data[col] = np.nan if col in NUMERIC_COLUMNS else ""

    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    data["Result"] = data["Result"].fillna("").astype(str)
    data = _normalize_point_ids(data)
    return data[RESULT_COLUMNS]


def _next_point_id(df: pd.DataFrame) -> str:
    if df is None or df.empty or "Point_ID" not in df.columns:
        return "P-1"

    used = set(df["Point_ID"].fillna("").astype(str).str.strip())
    num = 1
    while f"P-{num}" in used:
        num += 1
    return f"P-{num}"


def _within_indonesia_bounds(lon: float, lat: float) -> bool:
    return (
        INDONESIA_BOUNDS["lon_min"] <= lon <= INDONESIA_BOUNDS["lon_max"]
        and INDONESIA_BOUNDS["lat_min"] <= lat <= INDONESIA_BOUNDS["lat_max"]
    )


def _find_tile_for_lonlat(lon: float, lat: float) -> Optional[DemTileInfo]:
    for tile in get_dem_tile_index():
        if (
            tile["lon_min"] <= lon <= tile["lon_max"]
            and tile["lat_min"] <= lat <= tile["lat_max"]
        ):
            return tile
    return None


def _compute_center_slope_deg(
    patch: np.ndarray, pixel_x: float, pixel_y: float
) -> float:
    if patch.shape != (3, 3):
        return 0.0
    if np.isnan(patch).all():
        return 0.0

    filled = patch.astype("float32", copy=True)
    if np.isnan(filled).any():
        mean_val = np.nanmean(filled)
        if not np.isfinite(mean_val):
            return 0.0
        filled = np.where(np.isnan(filled), mean_val, filled)

    dz_dy, dz_dx = np.gradient(filled, pixel_y, pixel_x)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    val = float(slope_deg[1, 1])
    return val if np.isfinite(val) else 0.0


def _sample_dem_features(lon: float, lat: float) -> Tuple[float, float, bool]:
    tile = _find_tile_for_lonlat(lon, lat)
    if tile is None:
        return 0.0, 0.0, False

    try:
        with rasterio.open(tile["path"]) as src:
            if src.crs is None:
                return 0.0, 0.0, False

            transformed = transform("EPSG:4326", src.crs, [lon], [lat])
            if len(transformed) < 2 or not transformed[0] or not transformed[1]:
                return 0.0, 0.0, False

            x, y = float(transformed[0][0]), float(transformed[1][0])
            sample = next(src.sample([(x, y)]))
            elevation = float(sample[0]) if len(sample) else 0.0

            nodata_values = src.nodatavals or ()
            elev_nodata = nodata_values[0] if len(nodata_values) > 0 else src.nodata
            slope_nodata = nodata_values[1] if len(nodata_values) > 1 else src.nodata

            if elev_nodata is not None and np.isclose(elevation, elev_nodata):
                elevation = 0.0
            if not np.isfinite(elevation):
                elevation = 0.0

            if src.count >= 2 and len(sample) >= 2:
                slope = float(sample[1])
                if slope_nodata is not None and np.isclose(slope, slope_nodata):
                    slope = 0.0
                if not np.isfinite(slope):
                    slope = 0.0
            else:
                # Backward compatibility for legacy 1-band DEM tiles.
                row, col = src.index(x, y)
                patch = src.read(
                    1,
                    window=((row - 1, row + 2), (col - 1, col + 2)),
                    boundless=True,
                    fill_value=np.nan,
                ).astype("float32")

                if elev_nodata is not None:
                    patch = np.where(np.isclose(patch, elev_nodata), np.nan, patch)

                pixel_x = abs(float(src.transform.a)) or 1.0
                pixel_y = abs(float(src.transform.e)) or 1.0
                slope = _compute_center_slope_deg(patch, pixel_x, pixel_y)

            return round(elevation, 2), round(slope, 2), True
    except Exception:
        return 0.0, 0.0, False


def _distance_to_river_meters(lon: float, lat: float) -> float:
    river_geom = load_river_geometry()
    if river_geom is None:
        return 0.0

    try:
        transformer = get_lonlat_transformer()
        x, y = transformer.transform(lon, lat)
        distance = Point(x, y).distance(river_geom)
        if not np.isfinite(distance):
            return 0.0
        return round(float(distance), 2)
    except Exception:
        return 0.0


def _predict_rows(df: pd.DataFrame) -> pd.DataFrame:
    data = _normalize_result_df(df)
    if data.empty:
        return data

    missing = [c for c in SELECTED_FEATURES if c not in data.columns]
    if missing:
        raise ValueError("Missing model input columns: " + ", ".join(missing))

    model = load_model()
    x = data[SELECTED_FEATURES].astype(float)

    pred = model.predict(x)
    proba = model.predict_proba(x)
    if proba.ndim == 2 and proba.shape[1] > 1:
        proba_flood = proba[:, 1]
    else:
        proba_flood = np.asarray(proba).ravel()

    data["Result"] = np.where(pred == 1, "Flood", "Not Flood")
    confidence = np.where(pred == 1, proba_flood, 1.0 - proba_flood) * 100.0
    data["Confidence %"] = np.round(confidence, 2)

    return data[RESULT_COLUMNS]


def _popup_html(row: pd.Series) -> str:
    return (
        "<div style='font-size:13px;'>"
        f"<b>{row['Point_ID']}</b><br>"
        f"Lat, Lon: {_fmt_float(row['latitude'], 5)}, {_fmt_float(row['longitude'], 5)}<br>"
        f"Rainfall: {_fmt_float(row['avg_rainfall'])} mm<br>"
        f"Elevation: {_fmt_float(row['elevation'])} m<br>"
        f"Slope: {_fmt_float(row['slope'])} deg<br>"
        f"Distance to River: {_fmt_float(row['distance_to_river'])} m<br>"
        f"Result: <b>{row['Result']}</b><br>"
        f"Confidence: {_fmt_float(row['Confidence %'])}%"
        "</div>"
    )


def _build_points_map(points_df: pd.DataFrame, show_bounds: bool) -> folium.Map:
    fmap = folium.Map(
        location=MAP_CENTER,
        zoom_start=MAP_ZOOM,
        tiles="CartoDB positron",
        control_scale=True,
    )

    if show_bounds:
        folium.Rectangle(
            bounds=[
                [INDONESIA_BOUNDS["lat_min"], INDONESIA_BOUNDS["lon_min"]],
                [INDONESIA_BOUNDS["lat_max"], INDONESIA_BOUNDS["lon_max"]],
            ],
            color="#118ab2",
            weight=2,
            fill=False,
            tooltip="Indonesia bounding area",
        ).add_to(fmap)

    if points_df is None or points_df.empty:
        return fmap

    for _, row in points_df.iterrows():
        color = "red" if str(row["Result"]).strip().lower() == "flood" else "green"
        folium.Marker(
            location=[float(row["latitude"]), float(row["longitude"])],
            tooltip=f"{row['Point_ID']} | {row['Result']} ({_fmt_float(row['Confidence %'])}%)",
            popup=folium.Popup(_popup_html(row), max_width=360),
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(fmap)

    return fmap


@st.dialog("Point Variables")
def render_point_dialog() -> None:
    pending = st.session_state.get("pending_point")
    if not pending:
        return

    st.write(
        f"Latitude: {_fmt_float(pending['latitude'], 6)} | "
        f"Longitude: {_fmt_float(pending['longitude'], 6)}"
    )
    if not pending.get("dem_available", False):
        st.warning(
            "No local DEM tile matched this point. Elevation and slope default to 0."
        )

    key_suffix = pending.get("signature", "current").replace(".", "_").replace(",", "_")
    with st.form(f"dialog_form_{key_suffix}"):
        rainfall = st.number_input(
            "Average Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=float(pending["avg_rainfall"]),
            step=1.0,
            key=f"dlg_rain_{key_suffix}",
        )
        elevation = st.number_input(
            "Elevation (m)",
            value=float(pending["elevation"]),
            step=0.1,
            key=f"dlg_elev_{key_suffix}",
        )
        slope = st.number_input(
            "Slope (degree)",
            value=float(pending["slope"]),
            step=0.1,
            key=f"dlg_slope_{key_suffix}",
        )
        distance = st.number_input(
            "Distance to River (m)",
            value=float(pending["distance_to_river"]),
            step=0.1,
            key=f"dlg_dist_{key_suffix}",
        )

        col_save, col_cancel = st.columns(2)
        with col_save:
            save_clicked = st.form_submit_button("Save Point & Predict", type="primary")
        with col_cancel:
            cancel_clicked = st.form_submit_button("Cancel")

    if cancel_clicked:
        st.session_state.pending_point = None
        st.rerun()

    if save_clicked:
        new_row = pd.DataFrame(
            [
                {
                    "Point_ID": _next_point_id(st.session_state.results_df),
                    "latitude": pending["latitude"],
                    "longitude": pending["longitude"],
                    "avg_rainfall": rainfall,
                    "elevation": elevation,
                    "slope": slope,
                    "distance_to_river": distance,
                    "Result": "",
                    "Confidence %": 0.0,
                }
            ]
        )

        predicted_row = _predict_rows(new_row)
        merged = pd.concat(
            [st.session_state.results_df, predicted_row], ignore_index=True
        )
        st.session_state.results_df = _normalize_result_df(merged)
        st.session_state.pending_point = None
        st.rerun()


def render_map_input_tab(default_rainfall_mm: float) -> None:
    st.subheader("Map Input")
    st.caption(
        "Click inside Indonesia to calculate elevation, slope, and river distance. "
        "You can edit the values before prediction."
    )

    map_obj = _build_points_map(st.session_state.results_df, show_bounds=True)
    map_state = st_folium(
        map_obj,
        key="interactive_prediction_map",
        width=None,
        height=620,
        returned_objects=["last_clicked"],
    )

    clicked = (map_state or {}).get("last_clicked")
    if clicked:
        lat = float(clicked.get("lat", 0.0))
        lon = float(clicked.get("lng", 0.0))
        signature = f"{lat:.6f},{lon:.6f}"

        if signature != st.session_state.last_click_signature:
            st.session_state.last_click_signature = signature
            if not _within_indonesia_bounds(lon, lat):
                st.warning(
                    "Clicked point is outside Indonesia bounds. Please pick another point."
                )
            else:
                with st.spinner("Calculating point variables..."):
                    elevation, slope, dem_available = _sample_dem_features(lon, lat)
                    distance = _distance_to_river_meters(lon, lat)

                st.session_state.pending_point = {
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "avg_rainfall": float(default_rainfall_mm),
                    "elevation": float(elevation),
                    "slope": float(slope),
                    "distance_to_river": float(distance),
                    "dem_available": bool(dem_available),
                    "signature": signature,
                }

    if st.session_state.pending_point:
        render_point_dialog()


def render_results_tab() -> None:
    st.subheader("Mapping Results")
    if st.session_state.results_df.empty:
        st.info("No flood points yet. Add points from the map tab first.")
        return

    st.caption(
        "Edit rainfall/elevation/slope/distance values, then recalculate predictions."
    )
    non_editable = [c for c in RESULT_COLUMNS if c not in EDITABLE_COLUMNS]

    edited_df = st.data_editor(
        st.session_state.results_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        disabled=non_editable,
        key="results_editor",
    )

    col_recalc, col_clear = st.columns([1, 1])
    with col_recalc:
        if st.button("Recalculate Predictions", type="primary"):
            try:
                normalized = _normalize_result_df(edited_df)
                st.session_state.results_df = _predict_rows(normalized)
                st.success("Predictions updated.")
            except Exception as exc:
                st.error(str(exc))
    with col_clear:
        if st.button("Clear All Results"):
            st.session_state.results_df = _empty_results_df()
            st.session_state.pending_point = None
            st.info("All prediction points cleared.")

    st.download_button(
        "Download Results (CSV)",
        data=st.session_state.results_df.to_csv(index=False).encode("utf-8"),
        file_name="flood_mapping_results.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download Results (XLSX)",
        data=_to_excel_bytes(st.session_state.results_df),
        file_name="flood_mapping_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("### Result Map")
    st.caption(
        "Click a marker to view variables and prediction details for that point."
    )
    result_map = _build_points_map(st.session_state.results_df, show_bounds=False)
    st_folium(result_map, key="result_map", width=None, height=520)


def render_model_evaluation_tab() -> None:
    st.subheader("Model Evaluation")
    st.caption(
        "ROC AUC curve, confusion matrix, and classification report "
        "for the trained model on hold-out test data."
    )

    try:
        with st.spinner("Computing evaluation artifacts..."):
            artifacts = compute_model_evaluation_artifacts()
    except Exception as exc:
        st.error(str(exc))
        return

    roc_auc = float(artifacts["roc_auc"])
    accuracy = float(artifacts["accuracy"])
    test_rows = int(artifacts["test_rows"])
    positive_rows = int(artifacts["positive_rows"])

    col_auc, col_acc, col_rows, col_pos = st.columns(4)
    col_auc.metric("ROC AUC", f"{roc_auc:.4f}")
    col_acc.metric("Accuracy", f"{accuracy:.4f}")
    col_rows.metric("Test rows", f"{test_rows:,}")
    col_pos.metric("Flood rows (test)", f"{positive_rows:,}")

    st.markdown("### ROC Curve")
    roc_df = artifacts["roc_df"]
    baseline_df = pd.DataFrame(
        {
            "False Positive Rate": [0.0, 1.0],
            "True Positive Rate": [0.0, 1.0],
        }
    )

    roc_curve_chart = (
        alt.Chart(roc_df)
        .mark_line(color="#e63946", strokeWidth=3)
        .encode(
            x=alt.X(
                "False Positive Rate:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="False Positive Rate",
            ),
            y=alt.Y(
                "True Positive Rate:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="True Positive Rate",
            ),
            tooltip=[
                alt.Tooltip("False Positive Rate:Q", format=".4f"),
                alt.Tooltip("True Positive Rate:Q", format=".4f"),
            ],
        )
    )

    baseline_chart = (
        alt.Chart(baseline_df)
        .mark_line(color="#9e9e9e", strokeDash=[6, 6])
        .encode(
            x="False Positive Rate:Q",
            y="True Positive Rate:Q",
        )
    )

    st.altair_chart(
        (baseline_chart + roc_curve_chart).properties(
            title=f"ROC Curve (AUC = {roc_auc:.4f})", height=360
        ),
        use_container_width=True,
    )

    st.markdown("### Confusion Matrix")
    cm_df = artifacts["cm_df"]
    cm_order = ["Not Flood (0)", "Flood (1)"]

    cm_heatmap = (
        alt.Chart(cm_df)
        .mark_rect()
        .encode(
            x=alt.X("Predicted:N", sort=cm_order, title="Predicted"),
            y=alt.Y("Actual:N", sort=cm_order, title="Actual"),
            color=alt.Color("Count:Q", title="Count", scale=alt.Scale(scheme="blues")),
            tooltip=["Actual:N", "Predicted:N", alt.Tooltip("Count:Q", format=",")],
        )
    )

    cm_text = (
        alt.Chart(cm_df)
        .mark_text(fontSize=15)
        .encode(
            x=alt.X("Predicted:N", sort=cm_order),
            y=alt.Y("Actual:N", sort=cm_order),
            text=alt.Text("Count:Q", format=","),
            color=alt.value("black"),
        )
    )

    st.altair_chart(
        (cm_heatmap + cm_text).properties(height=280),
        use_container_width=True,
    )

    st.markdown("### Classification Report")
    report_df = artifacts["report_df"].copy()
    st.dataframe(report_df, use_container_width=True, hide_index=True)


def preload_resources() -> None:

    if st.session_state.resources_ready:
        return

    with st.spinner("Loading model and spatial layers..."):
        load_model()
        load_river_geometry()
        get_dem_tile_index()

    st.session_state.resources_ready = True


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Click map points, adjust variables, and view flood confidence directly on map and table."
    )

    ensure_state()

    try:
        preload_resources()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if load_river_geometry() is None:
        st.warning(
            "River geometry could not be loaded. Distance to river will default to 0."
        )

    if len(get_dem_tile_index()) == 0:
        st.warning("No DEM tiles found. Elevation and slope will default to 0.")

    with st.sidebar:
        st.header("Settings")
        default_rainfall = st.number_input(
            "Default rainfall for new points (mm)",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=1.0,
        )
        st.caption(
            "Indonesia bounds filter: "
            f"lon {INDONESIA_BOUNDS['lon_min']} to {INDONESIA_BOUNDS['lon_max']}, "
            f"lat {INDONESIA_BOUNDS['lat_min']} to {INDONESIA_BOUNDS['lat_max']}"
        )

    tab_map, tab_result = st.tabs(
        [
            "Map Input",
            "Mapping Results",
        ]
    )
    with tab_map:
        render_map_input_tab(default_rainfall)
    with tab_result:
        render_results_tab()


if __name__ == "__main__":
    main()

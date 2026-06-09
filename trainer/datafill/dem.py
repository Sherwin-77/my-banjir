import os
from pathlib import Path
from typing import Optional

import dotenv
import ee
import geemap
import rasterio
from rasterio.crs import CRS

from trainer.config import CRS_TARGET, DEM_PROJECTED_PATH

# --- CONFIG ---
west, south, east, north = 94.5, -11.5, 141.5, 6.5
tile_deg = 2.0
gee_dem_dataset = "COPERNICUS/DEM/GLO30"
gee_dem_band = "DEM"
gee_scale_meters = 30
output_dir = Path(DEM_PROJECTED_PATH)
# ----------------

output_dir.mkdir(parents=True, exist_ok=True)

dotenv.load_dotenv()


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


def is_valid_tif(path: Path, target_crs: Optional[str] = None, min_bands: int = 1) -> bool:
    """
    Returns True only if:
    - file exists
    - file size > 0
    - rasterio can successfully open it
    - band count >= min_bands
    - if target_crs is given, the CRS matches target_crs
    """
    if not path.exists():
        return False
    if path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as dst:
            dst_crs = dst.crs
            band_count = dst.count
        if band_count < min_bands:
            return False
        if target_crs is not None:
            return dst_crs == CRS.from_user_input(target_crs)
        return True
    except Exception:
        return False


def initialize_gee():
    try:
        ee.Initialize(project=os.getenv("EARTH_ENGINE_PROJECT_ID"))
    except Exception as exc:
        raise RuntimeError(
            "Google Earth Engine is not initialized. Run `earthengine authenticate` once, then rerun this script."
        ) from exc


def build_dem_with_slope_image() -> ee.Image:
    dem = ee.ImageCollection(gee_dem_dataset).select(gee_dem_band).mosaic().rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    return ee.Image.cat([dem, slope]).toFloat()


def download_one_tile(dem_with_slope: ee.Image, lon0: float, lat0: float, lon1: float, lat1: float, tile_path: Path):
    region = ee.Geometry.BBox(lon0, lat0, lon1, lat1)
    geemap.download_ee_image(
        image=dem_with_slope,
        filename=str(tile_path),
        region=region,
        crs=CRS_TARGET,
        scale=gee_scale_meters,
        overwrite=True,
    )


def download_tiles():
    use_gee = False
    dem_with_slope = None

    lon_steps = list(frange(west, east, tile_deg))
    lat_steps = list(frange(south, north, tile_deg))
    print(f"Will create {len(lon_steps) * len(lat_steps)} tiles")

    for lon0 in lon_steps:
        for lat0 in lat_steps:
            lon1 = min(lon0 + tile_deg, east)
            lat1 = min(lat0 + tile_deg, north)

            tile_path = output_dir / f"tile_{lon0:.3f}_{lat0:.3f}.tif"

            if is_valid_tif(tile_path, target_crs=CRS_TARGET, min_bands=2):
                print(f"[OK] Existing valid projected 2-band tile: {tile_path}")
                continue
            if tile_path.exists():
                print(f"[FIX] Removing corrupted/incomplete tile: {tile_path}")
                tile_path.unlink()

            if not use_gee:
                initialize_gee()
                dem_with_slope = build_dem_with_slope_image()
                use_gee = True


            print(f"Downloading GEE tile {tile_path.name} for bounds {lon0},{lat0},{lon1},{lat1}")

            try:
                download_one_tile(dem_with_slope, lon0, lat0, lon1, lat1, tile_path)  # type: ignore
            except Exception as exc:
                print(f"[ERROR] Download failed for tile {tile_path}: {exc}")
                if tile_path.exists():
                    tile_path.unlink()
                continue


if __name__ == "__main__":
    download_tiles()

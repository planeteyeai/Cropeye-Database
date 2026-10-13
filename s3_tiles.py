"""
Upload plot analysis tile PNGs to S3 and return public HTTPS URLs.
"""
import os
from typing import List, Optional

import boto3
import ee
import requests
from dotenv import load_dotenv

from ee_utils import ee_get_info

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "").strip().strip('"')
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip().strip('"')
AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME", "cropeye").strip().strip('"')
AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME", "ap-south-1").strip().strip('"')
S3_TILES_PREFIX = os.getenv("S3_TILES_PREFIX", "tiles").strip().strip('"').strip("/")
TILE_PNG_MAX_DIMENSION = int(os.getenv("TILE_PNG_MAX_DIMENSION", "768"))
# Set S3_TILE_UPLOAD_ENABLED=true in .env to re-enable S3 PNG uploads
S3_TILE_UPLOAD_ENABLED = (
    os.getenv("S3_TILE_UPLOAD_ENABLED", "false").strip().strip('"').lower()
    in ("1", "true", "yes", "on")
)

_s3_client = None


def _get_s3_client():
    global _s3_client
    if not S3_TILE_UPLOAD_ENABLED:
        return None

    if _s3_client is not None:
        return _s3_client

    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("⚠ AWS credentials missing — S3 tile upload disabled", flush=True)
        return None

    _s3_client = boto3.client(
        "s3",
        region_name=AWS_S3_REGION_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    return _s3_client


def public_tile_url(s3_key: str) -> str:
    return f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{s3_key}"


def build_tile_key(plot_id, analysis_type: str, analysis_date: str) -> str:
    safe_type = str(analysis_type).replace(" ", "_").lower()
    return f"{S3_TILES_PREFIX}/{plot_id}/{safe_type}/{analysis_date}.png"


def upload_png_bytes(png_bytes: bytes, s3_key: str) -> Optional[str]:
    client = _get_s3_client()
    if not client or not png_bytes:
        return None

    try:
        client.put_object(
            Bucket=AWS_STORAGE_BUCKET_NAME,
            Key=s3_key,
            Body=png_bytes,
            ContentType="image/png",
        )
        url = public_tile_url(s3_key)
        print(f"☁ Uploaded tile → {url}", flush=True)
        return url
    except Exception as e:
        print(f"❌ S3 upload failed for {s3_key}: {e}", flush=True)
        return None


def mask_visual_to_geometry(vis_image, geometry) -> ee.Image:
    """
    Pixels outside the plot polygon are masked so PNG export has a transparent
    background (matches GEE tile overlay on the map — no black letterboxing).
    """
    plot_mask = ee.Image.constant(1).clip(geometry).mask()
    return vis_image.updateMask(plot_mask)


def thumb_dimensions_for_geometry(
    geometry, max_dim: int = TILE_PNG_MAX_DIMENSION
) -> List[int]:
    """
    Non-square [width, height] so getThumbURL does not pad the plot with a
    square canvas (which caused black bars / wrong framing).
    """
    bounds = ee.Geometry(geometry).bounds(0.001)
    rect = ee_get_info(bounds, label="thumb-bounds") or {}
    coords = rect.get("coordinates", [[[]]])[0]
    if len(coords) < 4:
        return [max_dim, max_dim]

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    width_deg = max(lons) - min(lons)
    height_deg = max(lats) - min(lats)

    if width_deg <= 0 or height_deg <= 0:
        return [max_dim, max_dim]

    aspect = width_deg / height_deg
    if aspect >= 1:
        w = max_dim
        h = max(1, round(max_dim / aspect))
    else:
        h = max_dim
        w = max(1, round(max_dim * aspect))
    return [w, h]


def export_ee_image_png(
    vis_image,
    geometry,
    dimensions: int = TILE_PNG_MAX_DIMENSION,
) -> Optional[bytes]:
    """
    Export a visualized (RGB) Earth Engine image clipped to geometry as PNG bytes.
    Masked outside polygon → transparent; aspect ratio matches plot bounds.
    """
    try:
        masked = mask_visual_to_geometry(vis_image, geometry)
        dims = thumb_dimensions_for_geometry(geometry, max_dim=dimensions)

        thumb_url = masked.getThumbURL(
            {
                "region": geometry,
                "dimensions": dims,
                "format": "png",
            }
        )
        resp = requests.get(thumb_url, timeout=180)
        resp.raise_for_status()
        if not resp.content or len(resp.content) < 100:
            print("⚠ Empty/invalid PNG from Earth Engine thumb", flush=True)
            return None
        return resp.content
    except Exception as e:
        print(f"❌ EE PNG export failed: {e}", flush=True)
        return None


def export_and_upload_tile(
    vis_image,
    geometry,
    plot_id,
    analysis_type: str,
    analysis_date: str,
    dimensions: int = TILE_PNG_MAX_DIMENSION,
) -> Optional[str]:
    """
    Visualized EE image → PNG → S3. Returns public HTTPS URL or None.
    Disabled when S3_TILE_UPLOAD_ENABLED is not true.
    """
    if not S3_TILE_UPLOAD_ENABLED:
        return None

    if not plot_id or not analysis_date:
        return None

    png_bytes = export_ee_image_png(vis_image, geometry, dimensions=dimensions)
    if not png_bytes:
        return None

    s3_key = build_tile_key(plot_id, analysis_type, analysis_date)
    return upload_png_bytes(png_bytes, s3_key)


def resolve_tile_url(
    vis_image,
    geometry,
    plot_id,
    analysis_type: str,
    analysis_date: str,
    fallback_map_id: bool = True,
) -> Optional[str]:
    """
    Prefer durable S3 PNG URL when enabled; otherwise use ephemeral GEE MapID.
    """
    if S3_TILE_UPLOAD_ENABLED:
        s3_url = export_and_upload_tile(
            vis_image, geometry, plot_id, analysis_type, analysis_date
        )
        if s3_url:
            return s3_url

    if not fallback_map_id:
        return None

    try:
        return vis_image.getMapId()["tile_fetcher"].url_format
    except Exception as e:
        print(f"⚠ getMapId fallback failed: {e}", flush=True)
        return None


def month_half_windows(year: int, month: int):
    """
    Return [(start, end), ...] for first half (1–15) and second half (16–end).
    Ends are exclusive (Earth Engine filterDate style).
    """
    from datetime import date
    from dateutil.relativedelta import relativedelta

    first = date(year, month, 1)
    mid = date(year, month, 16)
    next_month = first + relativedelta(months=1)
    return [
        (first.isoformat(), mid.isoformat()),
        (mid.isoformat(), next_month.isoformat()),
    ]


def pick_preferred_sensor_result(results, prefer_s2: bool = True):
    """
    From a list of FeatureCollections, keep one: Sentinel-2 preferred, else Sentinel-1.
    """
    if not results:
        return None

    if isinstance(results, dict):
        results = [results]

    if not prefer_s2:
        return results

    def sensor_of(geojson):
        try:
            return str(
                geojson["features"][0]["properties"].get("sensor_used")
                or geojson["features"][0]["properties"].get("sensor")
                or ""
            ).lower()
        except Exception:
            return ""

    s2 = [r for r in results if "s2" in sensor_of(r) or "sentinel-2" in sensor_of(r)]
    s1 = [r for r in results if "s1" in sensor_of(r) or "sentinel-1" in sensor_of(r)]

    if s2:
        return [s2[0]]
    if s1:
        return [s1[0]]
    return [results[0]]

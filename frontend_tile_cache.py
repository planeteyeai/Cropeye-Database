"""
Snapshot exact cropeye-apis (frontend) tiles → S3 PNG + DB.

Does NOT change gee_growth analysis logic. Calls the live apis Admin HTTP
endpoints (same responses the frontend gets), stitches those GEE XYZ tiles
into one PNG (keeps pest black pixels), uploads to S3, stores full response
with S3 tile_url in analysis_results.

Rule: at most 1 stored result per half-month per analysis type (= 2 / month).
"""
from __future__ import annotations

import io
import math
import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from psycopg2.extras import Json, RealDictCursor
from shapely.geometry import shape
from shapely.ops import unary_union

from db import get_connection
from s3_tiles import (
    S3_TILE_UPLOAD_ENABLED,
    build_tile_key,
    month_half_windows,
    upload_png_bytes,
)
from shared_scene import plan_month_dense_dates, plan_month_shared_dates, scene_day_window

SHARED_DATE_TYPES = ("growth", "water_uptake", "soil_moisture")

load_dotenv()

APIS_BASE_DEFAULT = os.getenv("FRONTEND_APIS_BASE", "http://localhost:8003").strip()

APIS_ROUTES = {
    "growth": "analyze_Growth",
    "water_uptake": "wateruptake",
    "soil_moisture": "SoilMoisture",
    "pest_detection": "pest-detection",
}


def _run_query(query, params=None, fetch=False, fetchone=False):
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            if fetchone:
                row = cur.fetchone()
                conn.commit()
                return row
            if fetch:
                rows = cur.fetchall()
                conn.commit()
                return rows
            conn.commit()
            return None
    finally:
        conn.close()


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int(
        (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    )
    return x, y


def tile_bounds(x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
    n = 2.0**zoom
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0

    def y_to_lat(ty):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))

    return west, y_to_lat(y + 1), east, y_to_lat(y)


def expand_template(template: str, z: int, x: int, y: int) -> str:
    return (
        template.replace("{z}", str(z))
        .replace("{Z}", str(z))
        .replace("{x}", str(x))
        .replace("{X}", str(x))
        .replace("{y}", str(y))
        .replace("{Y}", str(y))
    )


# Frontend map/card frame: longest side capped (matches 472/1B growth reference ~768×364).
FRONTEND_TILE_MAX_DIM = int(os.getenv("FRONTEND_TILE_MAX_DIM", "768"))


def fit_png_max_dim(
    png_bytes: bytes,
    max_dim: int = FRONTEND_TILE_MAX_DIM,
    *,
    lock_size: Optional[Tuple[int, int]] = None,
) -> bytes:
    """
    Scale tile PNG for frontend frame fit.
    - If lock_size=(w,h) is set, force exact pixel size (per-plot lock).
    - Else scale so longest side == max_dim (keeps aspect). Default 768.
    Does NOT clear opaque black (pest shades must stay).
    """
    img = Image.open(io.BytesIO(png_bytes))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    if lock_size is not None:
        tw, th = int(lock_size[0]), int(lock_size[1])
        if tw > 0 and th > 0 and img.size != (tw, th):
            img = img.resize((tw, th), Image.Resampling.LANCZOS)
    else:
        w, h = img.size
        longest = max(w, h)
        if longest > 0 and longest != max_dim:
            scale = max_dim / float(longest)
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def choose_zoom(minx, miny, maxx, maxy, target_px: int = 1024) -> int:
    lat_c = (miny + maxy) / 2.0
    m_per_deg_lon = 111_320.0 * max(0.2, math.cos(math.radians(lat_c)))
    longest_m = max(
        max(1.0, (maxx - minx) * m_per_deg_lon),
        max(1.0, (maxy - miny) * 111_320.0),
    )
    zoom = 17
    for z in range(12, 20):
        mpp = 156_543.03 * math.cos(math.radians(lat_c)) / (2**z)
        zoom = z
        if longest_m / mpp >= target_px:
            break
    return zoom


def stitch_gee_tiles_exact(
    tile_template: str,
    geom_geojson: Dict[str, Any],
    max_dim: int = 768,
) -> bytes:
    """
    Exact frontend GEE tiles → PNG. Keeps black pest pixels (do NOT clear near-black).
    Transparent only outside the plot polygon mask.
    """
    geom = shape(geom_geojson)
    if geom.is_empty:
        raise ValueError("Empty geometry")
    if geom.geom_type == "MultiPolygon":
        geom = unary_union(geom)
    minx, miny, maxx, maxy = geom.bounds

    zoom = choose_zoom(minx, miny, maxx, maxy, target_px=max(max_dim * 2, 1024))
    x0, y0 = lonlat_to_tile(minx, maxy, zoom)
    x1, y1 = lonlat_to_tile(maxx, miny, zoom)
    if x1 - x0 > 10:
        mid = (x0 + x1) // 2
        x0, x1 = mid - 5, mid + 5
    if y1 - y0 > 10:
        mid = (y0 + y1) // 2
        y0, y1 = mid - 5, mid + 5

    cols = x1 - x0 + 1
    rows = y1 - y0 + 1
    canvas = Image.new("RGBA", (cols * 256, rows * 256), (0, 0, 0, 0))

    for iy, ty in enumerate(range(y0, y1 + 1)):
        for ix, tx in enumerate(range(x0, x1 + 1)):
            turl = expand_template(tile_template, zoom, tx, ty)
            try:
                r = requests.get(turl, timeout=90)
                r.raise_for_status()
                tile_img = Image.open(io.BytesIO(r.content)).convert("RGBA")
            except Exception:
                tile_img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
            canvas.paste(tile_img, (ix * 256, iy * 256), tile_img)

    west, _, _, north = tile_bounds(x0, y0, zoom)
    _, south, _, _ = tile_bounds(x0, y1, zoom)
    _, _, east, _ = tile_bounds(x1, y0, zoom)
    w, h = canvas.size

    def lon_to_px(lon: float) -> float:
        return (lon - west) / (east - west) * w

    def lat_to_py(lat: float) -> float:
        return (north - lat) / (north - south) * h

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in polys:
        exterior = [(lon_to_px(p[0]), lat_to_py(p[1])) for p in poly.exterior.coords]
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=255)
        for hole in poly.interiors:
            hr = [(lon_to_px(p[0]), lat_to_py(p[1])) for p in hole.coords]
            if len(hr) >= 3:
                draw.polygon(hr, fill=0)

    canvas.putalpha(mask)

    left = max(0, int(lon_to_px(minx)) - 4)
    right = min(w, int(lon_to_px(maxx)) + 4)
    top = max(0, int(lat_to_py(maxy)) - 4)
    bottom = min(h, int(lat_to_py(miny)) + 4)
    cropped = canvas if right <= left or bottom <= top else canvas.crop((left, top, right, bottom))
    cw, ch = cropped.size
    if cw < 1 or ch < 1:
        raise ValueError("Empty crop")

    if cw >= ch:
        nw, nh = max_dim, max(1, round(ch * max_dim / cw))
    else:
        nh, nw = max_dim, max(1, round(cw * max_dim / ch))
    cropped = cropped.resize((nw, nh), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    cropped.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def call_apis(
    apis_base: str,
    analysis_type: str,
    plot_name: str,
    end_date: str,
    start_date: Optional[str] = None,
    *,
    skip_cache: bool = False,
    tries: int = 4,
) -> Dict[str, Any]:
    route = APIS_ROUTES[analysis_type]
    url = (
        f"{apis_base.rstrip('/')}/{route}"
        f"?plot_name={requests.utils.quote(plot_name)}"
        f"&end_date={end_date}"
    )
    if start_date:
        url += f"&start_date={start_date}"
    if skip_cache:
        url += "&skip_cache=true"
    last: Optional[Exception] = None
    for i in range(max(1, tries)):
        try:
            res = requests.post(url, timeout=420)
            if res.status_code == 404:
                return {}
            if not res.ok:
                raise RuntimeError(f"apis {route} {res.status_code}: {res.text[:300]}")
            return res.json()
        except Exception as e:
            last = e
            print(f"   ↻ apis retry {i + 1}/{tries} ({route}): {e}", flush=True)
            time.sleep(min(45, 4 * (i + 1)))
    raise RuntimeError(last)

def _feature(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    feats = data.get("features") or []
    return feats[0] if feats else None


def _analysis_date(data: Dict[str, Any], feature: Dict[str, Any]) -> Optional[str]:
    props = feature.get("properties") or {}
    ps = data.get("pixel_summary") or {}
    nested = props.get("analysis_dates") or {}
    if not isinstance(nested, dict):
        nested = {}
    d = (
        props.get("latest_image_date")
        or props.get("image_date")
        or nested.get("latest_image_date")
        or ps.get("latest_image_date")
    )
    # Pest (frontend apis) returns image_dates[] instead of latest_image_date
    if not d:
        dates = props.get("image_dates") or ps.get("image_dates") or []
        if isinstance(dates, list) and dates:
            cleaned = sorted(str(x)[:10] for x in dates if x)
            d = cleaned[-1] if cleaned else None
    return str(d)[:10] if d else None


def _tile_url(feature: Dict[str, Any], data: Dict[str, Any]) -> Optional[str]:
    props = feature.get("properties") or {}
    return props.get("tile_url") or data.get("tile_url")


def _thumb_url(feature: Dict[str, Any], data: Dict[str, Any]) -> Optional[str]:
    """Seamless EE getThumbURL PNG when available (preferred over XYZ stitch)."""
    props = feature.get("properties") or {}
    u = props.get("thumb_url") or data.get("thumb_url")
    return str(u) if u else None


def _ensure_plot_row(plot_name: str, plot_data: Dict[str, Any]) -> int:
    props = plot_data.get("properties") or {}
    django_id = props.get("django_id")
    geom = plot_data.get("geometry")
    geom_geojson = None
    area_ha = None
    try:
        if geom is not None and hasattr(geom, "getInfo"):
            geom_geojson = geom.getInfo()
            area_ha = float(geom.area().divide(10000).getInfo())
    except Exception as e:
        print(f"⚠ geometry getInfo failed: {e}", flush=True)

    _run_query(
        """
        INSERT INTO plots
        (plot_name, geojson, area_hectares, django_plot_id, plantation_date, crop_type)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (plot_name) DO UPDATE SET
            geojson = COALESCE(EXCLUDED.geojson, plots.geojson),
            django_plot_id = COALESCE(EXCLUDED.django_plot_id, plots.django_plot_id),
            plantation_date = COALESCE(EXCLUDED.plantation_date, plots.plantation_date),
            crop_type = COALESCE(EXCLUDED.crop_type, plots.crop_type)
        """,
        (
            plot_name,
            Json(geom_geojson) if geom_geojson else None,
            area_ha,
            django_id,
            props.get("plantation_date"),
            props.get("crop_type_name") or props.get("crop_type"),
        ),
    )
    row = _run_query(
        "SELECT id FROM plots WHERE plot_name=%s",
        (plot_name,),
        fetchone=True,
    )
    if not row:
        raise RuntimeError(f"Could not resolve plot_id for {plot_name}")
    # plots.id is UUID in this DB (not an integer)
    return str(row["id"])


def _half_already_has(
    plot_id: str, analysis_type: str, half_start: str, half_end: str
) -> bool:
    row = _run_query(
        """
        SELECT id FROM analysis_results
        WHERE plot_id=%s AND analysis_type=%s
          AND analysis_date >= %s AND analysis_date < %s
          AND tile_url ILIKE '%%amazonaws.com%%'
        LIMIT 1
        """,
        (plot_id, analysis_type, half_start, half_end),
        fetchone=True,
    )
    return bool(row)


def _store(
    plot_id: str,
    analysis_type: str,
    analysis_date: str,
    sensor: str,
    s3_url: str,
    response: Dict[str, Any],
):
    feature = _feature(response)
    if feature and feature.get("properties") is not None:
        feature["properties"]["tile_url"] = s3_url
    _run_query(
        """
        INSERT INTO analysis_results
        (plot_id, analysis_type, analysis_date, sensor_used, tile_url, response_json)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (plot_id, analysis_type, analysis_date)
        DO UPDATE SET
            sensor_used = EXCLUDED.sensor_used,
            tile_url = EXCLUDED.tile_url,
            response_json = EXCLUDED.response_json
        """,
        (plot_id, analysis_type, analysis_date, sensor, s3_url, Json(response)),
    )


def _download_bytes(url: str, *, timeout: int = 300, tries: int = 4) -> bytes:
    last: Optional[Exception] = None
    for i in range(max(1, tries)):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            if r.content and len(r.content) > 100:
                return r.content
            raise RuntimeError("empty download")
        except Exception as e:
            last = e
            print(f"   ↻ download retry {i + 1}/{tries}: {e}", flush=True)
            time.sleep(min(30, 3 * (i + 1)))
    raise RuntimeError(last)


def _png_from_apis_response(
    data: Dict[str, Any],
    feature: Dict[str, Any],
    *,
    max_dim: int,
    lock_size: Optional[Tuple[int, int]],
    analysis_type: Optional[str] = None,
) -> bytes:
    """Thumb first; on timeout/fail fall back to XYZ stitch or raw URL.

    Pest prefers seamless thumb; stitch fallback uses smooth pest-presence tiles.
    """
    gee_url = _tile_url(feature, data)
    thumb_url = _thumb_url(feature, data)
    geom = feature.get("geometry")
    is_pest = analysis_type == "pest_detection"

    if thumb_url:
        try:
            # Pest thumbs can be slow; keep tries low so we can fall back to stitch.
            png = _download_bytes(
                thumb_url, timeout=180 if is_pest else 300, tries=2 if is_pest else 3
            )
            return fit_png_max_dim(png, max_dim=max_dim, lock_size=lock_size)
        except Exception as e:
            print(
                f"   ⚠ thumb failed ({e}); falling back to stitch/raw"
                + (" (pest smooth-presence tiles)" if is_pest else ""),
                flush=True,
            )

    if gee_url and geom and (
        "earthengine" in gee_url.lower() or "/tiles/" in gee_url.lower()
    ):
        png = stitch_gee_tiles_exact(gee_url, geom, max_dim=max_dim)
        return fit_png_max_dim(png, max_dim=max_dim, lock_size=lock_size)

    if gee_url:
        png = _download_bytes(gee_url, timeout=180, tries=3)
        return fit_png_max_dim(png, max_dim=max_dim, lock_size=lock_size)

    raise RuntimeError("no tile_url/thumb_url usable for PNG")


def _parse_month_start(value: Optional[str], *, field: str) -> Optional[date]:
    """Parse YYYY-MM or YYYY-MM-DD → first day of that month."""
    if value is None or str(value).strip() == "":
        return None
    raw = str(value).strip()[:10]
    try:
        if len(raw) == 7 and raw[4] == "-":
            d = datetime.strptime(raw, "%Y-%m").date().replace(day=1)
        else:
            d = datetime.strptime(raw[:10], "%Y-%m-%d").date().replace(day=1)
    except Exception as e:
        raise ValueError(f"{field} must be YYYY-MM or YYYY-MM-DD (got {value!r})") from e
    return d


def cache_frontend_tiles_for_plot(
    plot_name: str,
    plot_data: Dict[str, Any],
    *,
    apis_base: str = APIS_BASE_DEFAULT,
    months: Optional[int] = 2,
    force: bool = False,
    analysis_types: Optional[List[str]] = None,
    max_dim: int = 768,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
    scene_mode: str = "shared",
) -> Dict[str, Any]:
    """
    For one plot: snapshot frontend tiles → PNG → S3 → DB.

    scene_mode:
      - \"shared\" (default): up to 2 shared S2-first dates/month + pest half-months
      - \"dense\": up to 4 clearest S2/month (2 before/after day 15) + ALL S1 dates;
        pest uses every S1 date (day window)

    months=None → plantation→today (capped 36). Optional start_month/end_month limit window.
    """
    if max_dim is None or max_dim <= 0:
        max_dim = FRONTEND_TILE_MAX_DIM
    if not S3_TILE_UPLOAD_ENABLED:
        raise RuntimeError(
            "S3_TILE_UPLOAD_ENABLED must be true in .env to store frontend tiles"
        )
    mode = (scene_mode or "shared").strip().lower()
    if mode not in ("shared", "dense"):
        raise ValueError("scene_mode must be 'shared' or 'dense'")

    types = analysis_types or list(APIS_ROUTES.keys())
    plot_id = _ensure_plot_row(plot_name, plot_data)
    props = plot_data.get("properties") or {}

    today_month = date.today().replace(day=1)
    range_start = _parse_month_start(start_month, field="start_month")
    range_end = _parse_month_start(end_month, field="end_month")
    if range_start and range_end and range_start > range_end:
        raise ValueError("start_month must be <= end_month")

    plantation_day = None
    raw_plant = props.get("plantation_date")
    if raw_plant:
        try:
            plantation_day = datetime.strptime(str(raw_plant)[:10], "%Y-%m-%d").date()
        except Exception:
            plantation_day = None
    if plantation_day is None:
        row = _run_query(
            "SELECT plantation_date FROM plots WHERE id=%s OR plot_name=%s LIMIT 1",
            (plot_id, plot_name),
            fetchone=True,
        )
        if row and row.get("plantation_date"):
            try:
                plantation_day = (
                    row["plantation_date"]
                    if hasattr(row["plantation_date"], "year")
                    else datetime.strptime(str(row["plantation_date"])[:10], "%Y-%m-%d").date()
                )
                props["plantation_date"] = plantation_day.isoformat()
            except Exception:
                plantation_day = None

    # Newest month to process (loop starts here, walks backward)
    current_start = range_end if range_end is not None else today_month
    if current_start > today_month:
        current_start = today_month

    if months is not None:
        stop_month = current_start - relativedelta(months=max(0, months - 1))
    elif plantation_day:
        stop_month = plantation_day.replace(day=1)
        oldest = current_start - relativedelta(months=35)
        if stop_month < oldest:
            stop_month = oldest
    else:
        stop_month = current_start - relativedelta(months=1)

    if range_start is not None and stop_month < range_start:
        stop_month = range_start
    if range_start is not None and current_start < range_start:
        # empty window
        stop_month = current_start + relativedelta(months=1)

    if plantation_day and stop_month < plantation_day.replace(day=1):
        # still allow stop_month as computed for test window
        pass

    summary = {
        "plot_name": plot_name,
        "plot_id": plot_id,
        "apis_base": apis_base,
        "scene_mode": mode,
        "start_month": stop_month.isoformat(),
        "end_month": current_start.isoformat(),
        "stop_month": stop_month.isoformat(),
        "stored": [],
        "skipped": [],
        "errors": [],
    }

    geometry = plot_data.get("geometry")
    shared_types = [t for t in types if t in SHARED_DATE_TYPES]
    other_types = [t for t in types if t not in SHARED_DATE_TYPES]

    # Lock all tiles for this plot to one pixel size so frontend frame fit
    # stays consistent. Prefer existing growth PNG on S3 as the reference.
    lock_size: Optional[Tuple[int, int]] = None
    bucket = os.getenv("AWS_STORAGE_BUCKET_NAME", "cropeye").strip().strip('"')
    region = os.getenv("AWS_S3_REGION_NAME", "ap-south-1").strip().strip('"')
    prefix = os.getenv("S3_TILES_PREFIX", "tiles").strip().strip('"').strip("/")
    for d in ("2026-06-14", "2026-07-10", "2026-06-21", date.today().isoformat()):
        try:
            u = f"https://{bucket}.s3.{region}.amazonaws.com/{prefix}/{plot_id}/growth/{d}.png"
            rr = requests.get(u, timeout=20)
            if rr.status_code == 200 and len(rr.content) > 100:
                lock_size = Image.open(io.BytesIO(rr.content)).size
                print(f"🔒 Plot tile size lock from growth/{d}: {lock_size}", flush=True)
                break
        except Exception:
            continue

    def _snapshot_one(
        analysis_type: str,
        apis_end: str,
        *,
        start_date=None,
        half_label=None,
        tries: int = 4,
    ):
        nonlocal lock_size
        label = f"{analysis_type} {half_label or apis_end}"
        print(f"🔎 Frontend snapshot {plot_name}: {label}", flush=True)
        last_err: Optional[Exception] = None
        for attempt in range(max(1, tries)):
            try:
                data = call_apis(
                    apis_base,
                    analysis_type,
                    plot_name,
                    apis_end,
                    start_date=start_date,
                    skip_cache=force,
                    tries=3,
                )
                feature = _feature(data)
                if not feature:
                    msg = f"{label} (no features)"
                    summary["skipped"].append(msg)
                    print(f"⏭ Skip {msg}", flush=True)
                    return
                gee_url = _tile_url(feature, data)
                thumb_url = _thumb_url(feature, data)
                analysis_date = _analysis_date(data, feature)
                if not analysis_date or (not gee_url and not thumb_url):
                    msg = f"{label} (no tile/date)"
                    summary["skipped"].append(msg)
                    print(f"⏭ Skip {msg}", flush=True)
                    return
                geom = feature.get("geometry")
                if not geom and not thumb_url:
                    raise RuntimeError("no geometry for stitch fallback")

                png = _png_from_apis_response(
                    data,
                    feature,
                    max_dim=max_dim,
                    lock_size=lock_size,
                    analysis_type=analysis_type,
                )
                if lock_size is None:
                    try:
                        lock_size = Image.open(io.BytesIO(png)).size
                        print(f"🔒 Plot tile size lock set to {lock_size}", flush=True)
                    except Exception:
                        pass
                s3_key = build_tile_key(plot_id, analysis_type, analysis_date)
                s3_url = upload_png_bytes(png, s3_key)
                if not s3_url:
                    raise RuntimeError("S3 upload failed")
                sensor = str(
                    (feature.get("properties") or {}).get("sensor_used")
                    or (feature.get("properties") or {}).get("data_source")
                    or (feature.get("properties") or {}).get("sensor")
                    or "frontend_apis"
                )
                _store(plot_id, analysis_type, analysis_date, sensor, s3_url, data)
                item = {
                    "analysis_type": analysis_type,
                    "analysis_date": analysis_date,
                    "sensor": sensor,
                    "half": half_label or analysis_date,
                    "s3_url": s3_url,
                }
                summary["stored"].append(item)
                print(
                    f"   ✅ {analysis_type} {analysis_date} ({sensor}) → {s3_url}",
                    flush=True,
                )
                return
            except Exception as e:
                last_err = e
                print(
                    f"   ↻ snapshot retry {attempt + 1}/{tries} {label}: {e}",
                    flush=True,
                )
                time.sleep(min(45, 5 * (attempt + 1)))
        msg = f"{label}: {last_err}"
        summary["errors"].append(msg)
        print(f"❌ {msg}", flush=True)
    current = current_start
    while current >= stop_month:
        pest_s1_dates: List[str] = []
        # --- growth / water / soil ---
        if shared_types and geometry is not None:
            planned = []
            plan_err: Optional[Exception] = None
            for plan_try in range(4):
                try:
                    if mode == "dense":
                        dense = plan_month_dense_dates(
                            geometry, current.year, current.month
                        )
                        # Up to 4 clearest cloud-free S2; if none, up to 4 S1.
                        # (Avoid storing every S1 pass for growth/water/soil.)
                        s2_pick = dense.get("s2") or []
                        s1_pick = dense.get("s1") or []
                        planned = s2_pick if s2_pick else s1_pick[:4]
                        pest_s1_dates = [s["date"] for s in s1_pick]
                    else:
                        planned = plan_month_shared_dates(
                            geometry, current.year, current.month
                        )
                        pest_s1_dates = []
                    plan_err = None
                    break
                except Exception as e:
                    plan_err = e
                    print(
                        f"   ↻ plan month retry {plan_try + 1}/4 {current}: {e}",
                        flush=True,
                    )
                    time.sleep(min(30, 4 * (plan_try + 1)))
            if plan_err is not None:
                summary["errors"].append(
                    f"plan {current.year}-{current.month:02d}: {plan_err}"
                )
                print(f"❌ plan month {current}: {plan_err}", flush=True)
                planned = []
                pest_s1_dates = []
            for scene in planned:
                img_date = scene["date"]
                if plantation_day and date.fromisoformat(img_date) < plantation_day:
                    continue
                if date.fromisoformat(img_date) > date.today():
                    continue
                day_start, day_end = scene_day_window(img_date)
                # EE exclusive end is day_end; apis end_date uses same
                for analysis_type in shared_types:
                    # skip check: already have this exact date
                    if not force:
                        row = _run_query(
                            """
                            SELECT id FROM analysis_results
                            WHERE plot_id=%s AND analysis_type=%s AND analysis_date=%s
                              AND tile_url ILIKE '%%amazonaws.com%%'
                            LIMIT 1
                            """,
                            (plot_id, analysis_type, img_date),
                            fetchone=True,
                        )
                        if row:
                            msg = f"{analysis_type} {img_date} (S3 already in DB)"
                            summary["skipped"].append(msg)
                            print(f"✔ Skip {msg}", flush=True)
                            continue
                    _snapshot_one(
                        analysis_type,
                        day_end,
                        start_date=day_start,
                        half_label=f"{img_date} {mode}/{scene['sensor']}",
                    )

        # --- pest ---
        if other_types:
            if mode == "dense":
                # All S1 acquisition dates in the month (day windows)
                if not pest_s1_dates and geometry is not None:
                    try:
                        dense = plan_month_dense_dates(
                            geometry, current.year, current.month
                        )
                        pest_s1_dates = [s["date"] for s in (dense.get("s1") or [])]
                    except Exception as e:
                        summary["errors"].append(
                            f"pest plan {current.year}-{current.month:02d}: {e}"
                        )
                        pest_s1_dates = []
                for img_date in pest_s1_dates:
                    if plantation_day and date.fromisoformat(img_date) < plantation_day:
                        continue
                    if date.fromisoformat(img_date) > date.today():
                        continue
                    day_start, day_end = scene_day_window(img_date)
                    for analysis_type in other_types:
                        if not force:
                            row = _run_query(
                                """
                                SELECT id FROM analysis_results
                                WHERE plot_id=%s AND analysis_type=%s AND analysis_date=%s
                                  AND tile_url ILIKE '%%amazonaws.com%%'
                                LIMIT 1
                                """,
                                (plot_id, analysis_type, img_date),
                                fetchone=True,
                            )
                            if row:
                                msg = f"{analysis_type} {img_date} (S3 already in DB)"
                                summary["skipped"].append(msg)
                                print(f"✔ Skip {msg}", flush=True)
                                continue
                        _snapshot_one(
                            analysis_type,
                            day_end,
                            start_date=day_start,
                            half_label=f"{img_date} dense/s1",
                        )
            else:
                for half_start, half_end in month_half_windows(
                    current.year, current.month
                ):
                    start_d = date.fromisoformat(half_start)
                    end_d = date.fromisoformat(half_end)
                    if start_d > date.today():
                        continue
                    hs = half_start
                    if plantation_day:
                        if end_d <= plantation_day:
                            continue
                        if start_d < plantation_day:
                            hs = plantation_day.isoformat()
                    apis_end = (end_d - timedelta(days=1)).isoformat()
                    if apis_end < hs:
                        apis_end = hs
                    for analysis_type in other_types:
                        label = f"{analysis_type} {hs}→{half_end}"
                        if not force and _half_already_has(
                            plot_id, analysis_type, hs, half_end
                        ):
                            summary["skipped"].append(f"{label} (S3 already in DB)")
                            print(f"✔ Skip {label}", flush=True)
                            continue
                        _snapshot_one(
                            analysis_type, apis_end, half_label=f"{hs}→{half_end}"
                        )

        current -= relativedelta(months=1)

    summary["stored_count"] = len(summary["stored"])
    summary["skipped_count"] = len(summary["skipped"])
    summary["error_count"] = len(summary["errors"])
    return summary


# ---------------------------------------------------------------------------
# Bulk all-plots job (background). Resume-friendly: skips existing S3 rows.
# ---------------------------------------------------------------------------
_BULK_STATUS: Dict[str, Any] = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "total": 0,
    "index": 0,
    "current_plot": None,
    "ok": 0,
    "failed": 0,
    "stored_tiles": 0,
    "skipped_tiles": 0,
    "error_tiles": 0,
    "failed_plots": [],
    "last_error": None,
    "stop_requested": False,
    "params": {},
}


def get_bulk_tile_cache_status() -> Dict[str, Any]:
    return dict(_BULK_STATUS)


def request_stop_bulk_tile_cache() -> Dict[str, Any]:
    _BULK_STATUS["stop_requested"] = True
    return {"status": "stop_requested", **get_bulk_tile_cache_status()}


def cache_frontend_tiles_all_plots(
    plot_items: List[Tuple[str, Dict[str, Any]]],
    *,
    apis_base: str = APIS_BASE_DEFAULT,
    months: Optional[int] = None,
    force: bool = False,
    analysis_types: Optional[List[str]] = None,
    sleep_sec: float = 3.0,
    max_dim: int = 768,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
) -> None:
    """
    Process plots one-by-one. months=None → plantation→today (capped 36).
    Optional start_month / end_month (YYYY-MM) limit the inclusive month window.
    force=False skips dates that already have an S3 tile_url in DB.
    """
    global _BULK_STATUS
    if _BULK_STATUS.get("running"):
        raise RuntimeError("bulk tile cache already running")

    _BULK_STATUS = {
        "running": True,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "total": len(plot_items),
        "index": 0,
        "current_plot": None,
        "ok": 0,
        "failed": 0,
        "stored_tiles": 0,
        "skipped_tiles": 0,
        "error_tiles": 0,
        "failed_plots": [],
        "last_error": None,
        "stop_requested": False,
        "params": {
            "apis_base": apis_base,
            "months": months,
            "force": force,
            "analysis_types": analysis_types,
            "sleep_sec": sleep_sec,
            "max_dim": max_dim,
            "start_month": start_month,
            "end_month": end_month,
        },
    }

    print(
        f"🚀 Bulk frontend tile cache started: {len(plot_items)} plots "
        f"(months={months}, start_month={start_month}, end_month={end_month}, "
        f"force={force}, apis={apis_base})",
        flush=True,
    )

    try:
        for i, (plot_name, plot_data) in enumerate(plot_items, 1):
            if _BULK_STATUS.get("stop_requested"):
                print("🛑 Bulk tile cache stop requested — exiting loop", flush=True)
                break
            _BULK_STATUS["index"] = i
            _BULK_STATUS["current_plot"] = plot_name
            if i > 1 and sleep_sec > 0:
                time.sleep(sleep_sec)
            print(
                f"\n[{i}/{len(plot_items)}] Frontend tiles → {plot_name}",
                flush=True,
            )
            try:
                result = cache_frontend_tiles_for_plot(
                    plot_name,
                    plot_data,
                    apis_base=apis_base,
                    months=months,
                    force=force,
                    analysis_types=analysis_types,
                    max_dim=max_dim,
                    start_month=start_month,
                    end_month=end_month,
                )
                stored_n = int(result.get("stored_count") or 0)
                skip_n = int(result.get("skipped_count") or 0)
                err_n = int(result.get("error_count") or 0)
                _BULK_STATUS["stored_tiles"] += stored_n
                _BULK_STATUS["skipped_tiles"] += skip_n
                _BULK_STATUS["error_tiles"] += err_n
                if err_n and not stored_n:
                    _BULK_STATUS["failed"] += 1
                    _BULK_STATUS["failed_plots"].append(
                        {
                            "plot_name": plot_name,
                            "errors": (result.get("errors") or [])[:8],
                        }
                    )
                else:
                    _BULK_STATUS["ok"] += 1
                print(
                    f"✅ {plot_name}: stored={stored_n} skipped={skip_n} errors={err_n}",
                    flush=True,
                )
            except Exception as e:
                _BULK_STATUS["failed"] += 1
                _BULK_STATUS["last_error"] = str(e)
                _BULK_STATUS["failed_plots"].append(
                    {"plot_name": plot_name, "errors": [str(e)]}
                )
                print(f"🔥 {plot_name} failed: {e}", flush=True)
                # brief pause then continue — do not abort the whole run
                time.sleep(min(60, sleep_sec * 3 + 5))
    finally:
        _BULK_STATUS["running"] = False
        _BULK_STATUS["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _BULK_STATUS["current_plot"] = None
        print(
            f"\n✅ Bulk frontend tile cache done. "
            f"ok={_BULK_STATUS['ok']} failed={_BULK_STATUS['failed']} "
            f"stored_tiles={_BULK_STATUS['stored_tiles']} "
            f"skipped={_BULK_STATUS['skipped_tiles']} "
            f"error_tiles={_BULK_STATUS['error_tiles']} "
            f"total={_BULK_STATUS['total']}",
            flush=True,
        )

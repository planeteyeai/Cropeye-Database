"""
Shared Sentinel-2-first scene selection for growth / water_uptake / soil_moisture.

Rules:
  1. Use Sentinel-2 only when CLOUDY_PIXEL_PERCENTAGE < 45.
  2. Use Sentinel-1 only if no qualifying S2 scene exists in the window.
  3. All three analysis types must call the same picker → same date + sensor.
  4. Month planning (up to 2 tiles / month):
     - Per half: S2 (<45% cloud) if available, else S1.
     - If both halves have a scene → one date from each half.
     - If only one half has scenes → up to 2 dates from that half (same sensor preference).
"""
from __future__ import annotations

from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import ee

# Scene-level cloud filter for shared growth / water / soil S2 selection.
S2_MAX_CLOUD_PCT = 45


def _to_date(d) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


def _exclusive_end(d: date) -> str:
    return (d + timedelta(days=1)).isoformat()


def _unique_sorted_dates_from_collection(coll) -> List[str]:
    n = int(coll.size().getInfo() or 0)
    if n == 0:
        return []
    dates = (
        coll.aggregate_array("system:time_start")
        .map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
        .getInfo()
        or []
    )
    return sorted({str(d)[:10] for d in dates if d})


def list_s2_dates(geometry, start, end, max_cloud_pct: float = S2_MAX_CLOUD_PCT) -> List[str]:
    """Distinct S2 dates in [start, end) with CLOUDY_PIXEL_PERCENTAGE < max_cloud_pct."""
    start_s = str(start)[:10]
    end_s = str(end)[:10]
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(geometry)
        .filterDate(start_s, end_s)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .sort("system:time_start", True)
    )
    return _unique_sorted_dates_from_collection(coll)


def list_s1_dates(geometry, start, end) -> List[str]:
    """All distinct S1 IW VH acquisition dates in [start, end)."""
    start_s = str(start)[:10]
    end_s = str(end)[:10]
    coll = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start_s, end_s)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .sort("system:time_start", True)
    )
    return _unique_sorted_dates_from_collection(coll)


def pick_shared_scene(
    geometry,
    start,
    end,
) -> Optional[Dict[str, str]]:
    """
    Pick one shared scene for a window (typically one half-month).
    Returns {"date": "YYYY-MM-DD", "sensor": "s2"|"s1"} or None.
    """
    s2 = list_s2_dates(geometry, start, end)
    if s2:
        return {"date": s2[-1], "sensor": "s2"}
    s1 = list_s1_dates(geometry, start, end)
    if s1:
        return {"date": s1[-1], "sensor": "s1"}
    return None


def _pick_up_to_two(dates: List[str], sensor: str) -> List[Dict[str, str]]:
    if not dates:
        return []
    if len(dates) == 1:
        return [{"date": dates[-1], "sensor": sensor}]
    return [
        {"date": dates[-2], "sensor": sensor},
        {"date": dates[-1], "sensor": sensor},
    ]


def _pick_up_to_two_from_half(geometry, start, end) -> List[Dict[str, str]]:
    s2 = list_s2_dates(geometry, start, end)
    if s2:
        return _pick_up_to_two(s2, "s2")
    s1 = list_s1_dates(geometry, start, end)
    return _pick_up_to_two(s1, "s1")


def list_s2_scenes_ranked(
    geometry, start, end, max_cloud_pct: float = S2_MAX_CLOUD_PCT
) -> List[Dict[str, Any]]:
    """
    S2 scenes in [start, end) with cloud < max_cloud_pct.
    Returns [{date, cloud}, ...] sorted by cloud ascending, then date.
    """
    start_s = str(start)[:10]
    end_s = str(end)[:10]
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(geometry)
        .filterDate(start_s, end_s)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .sort("system:time_start", True)
    )
    n = int(coll.size().getInfo() or 0)
    if n == 0:
        return []
    fc = ee.FeatureCollection(
        coll.map(
            lambda img: ee.Feature(
                None,
                {
                    "date": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"),
                    "cloud": img.get("CLOUDY_PIXEL_PERCENTAGE"),
                },
            )
        )
    )
    feats = fc.getInfo().get("features") or []
    scenes: List[Dict[str, Any]] = []
    seen = set()
    for f in feats:
        props = f.get("properties") or {}
        d = str(props.get("date") or "")[:10]
        if not d or d in seen:
            continue
        seen.add(d)
        try:
            cloud = float(props.get("cloud"))
        except Exception:
            cloud = 999.0
        scenes.append({"date": d, "cloud": cloud})
    scenes.sort(key=lambda s: (s["cloud"], s["date"]))
    return scenes


def plan_month_s2_four(geometry, year: int, month: int) -> List[Dict[str, str]]:
    """
    Up to 4 most accurate (lowest cloud) S2 scenes per month:
      - prefer 2 in days 1–15 and 2 in days 16–end
      - if a half is short, fill remaining slots from other leftover S2 in the month
    """
    first = date(year, month, 1)
    next_month = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
    ranked = list_s2_scenes_ranked(geometry, first.isoformat(), next_month.isoformat())
    if not ranked:
        return []

    h1 = [s for s in ranked if _to_date(s["date"]).day <= 15]
    h2 = [s for s in ranked if _to_date(s["date"]).day >= 16]
    picked: List[Dict[str, Any]] = []
    picked.extend(h1[:2])
    picked.extend(h2[:2])
    if len(picked) < 4:
        picked_dates = {s["date"] for s in picked}
        for s in ranked:
            if len(picked) >= 4:
                break
            if s["date"] not in picked_dates:
                picked.append(s)
                picked_dates.add(s["date"])
    picked.sort(key=lambda s: s["date"])
    return [{"date": s["date"], "sensor": "s2"} for s in picked]


def plan_month_dense_dates(geometry, year: int, month: int) -> Dict[str, List[Dict[str, str]]]:
    """
    Dense monthly plan for high-frequency tile store:
      - s2: up to 4 clearest S2 (2 first half + 2 second half when available)
      - s1: ALL S1 IW VH dates in the month
      - shared: S2 picks + all S1 (for growth/water/soil day windows)
    """
    first = date(year, month, 1)
    next_month = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
    s2 = plan_month_s2_four(geometry, year, month)
    s1_dates = list_s1_dates(geometry, first.isoformat(), next_month.isoformat())
    s1 = [{"date": d, "sensor": "s1"} for d in s1_dates]
    # Prefer S2 on a date if both exist (avoid duplicate day for shared types)
    s2_dates = {s["date"] for s in s2}
    s1_for_shared = [s for s in s1 if s["date"] not in s2_dates]
    shared = sorted(s2 + s1_for_shared, key=lambda s: s["date"])
    return {"s2": s2, "s1": s1, "shared": shared}


def plan_month_shared_dates(geometry, year: int, month: int) -> List[Dict[str, str]]:
    """
    Up to 2 shared scenes for a calendar month (growth/water/soil all use these).
    Per half: S2 with cloud < 45%, else S1.
    """
    first = date(year, month, 1)
    mid = date(year, month, 16)
    next_month = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)

    h1_start, h1_end = first.isoformat(), mid.isoformat()
    h2_start, h2_end = mid.isoformat(), next_month.isoformat()

    h1 = pick_shared_scene(geometry, h1_start, h1_end)
    h2 = pick_shared_scene(geometry, h2_start, h2_end)

    if h1 and h2:
        return [h1, h2]
    if h1 and not h2:
        return _pick_up_to_two_from_half(geometry, h1_start, h1_end)
    if h2 and not h1:
        return _pick_up_to_two_from_half(geometry, h2_start, h2_end)
    return []


def scene_day_window(image_date: str) -> Tuple[str, str]:
    """Inclusive image day as EE filterDate [start, end) window."""
    d = _to_date(image_date)
    return d.isoformat(), _exclusive_end(d)


def load_s2_image_on_date(geometry, image_date: str, max_cloud_pct: float = S2_MAX_CLOUD_PCT):
    start, end = scene_day_window(image_date)
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(geometry)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .sort("CLOUDY_PIXEL_PERCENTAGE", True)
    )
    if int(coll.size().getInfo() or 0) == 0:
        return None
    return ee.Image(coll.first()).clip(geometry)


def load_s1_vh_on_date(geometry, image_date: str):
    start, end = scene_day_window(image_date)
    coll = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VH"])
        .sort("system:time_start", False)
    )
    if int(coll.size().getInfo() or 0) == 0:
        return None
    return ee.Image(coll.first()).clip(geometry)


def load_s1_vh_pair(geometry, latest_date: str, lookback_days: int = 45):
    """Latest VH on latest_date + previous VH before that date (for water delta)."""
    latest = load_s1_vh_on_date(geometry, latest_date)
    if latest is None:
        return None, None
    d = _to_date(latest_date)
    prev_start = (d - timedelta(days=lookback_days)).isoformat()
    coll = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(prev_start, latest_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VH"])
        .sort("system:time_start", False)
    )
    if int(coll.size().getInfo() or 0) == 0:
        return latest, None
    return latest, ee.Image(coll.first()).clip(geometry)


def load_s1_vv_on_date(geometry, image_date: str):
    start, end = scene_day_window(image_date)
    coll = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select(["VV"])
        .sort("system:time_start", False)
    )
    if int(coll.size().getInfo() or 0) == 0:
        return None
    return ee.Image(coll.first()).clip(geometry)

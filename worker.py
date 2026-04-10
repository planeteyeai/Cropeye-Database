from datetime import date, timedelta
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue

from psycopg2.extras import RealDictCursor, Json

from gee_growth import (
    run_growth_analysis_by_plot,
    run_water_uptake_analysis_by_plot,
    run_soil_moisture_analysis_by_plot,
    run_pest_detection_analysis_by_plot
)

from db import get_connection
from Admin import run_monthly_backfill_for_plot
from shared_services import PlotSyncService

# =====================================================
# INIT
# =====================================================

plot_sync_service = PlotSyncService()
plot_dict = {}

task_queue = PriorityQueue()

MAX_PARALLEL_ANALYSIS = 3
GLOBAL_LIMIT = 4

semaphore = threading.Semaphore(GLOBAL_LIMIT)

# =====================================================
# DB
# =====================================================

def run_query(query, params=None, fetchone=False, fetchall=False):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        if params:
            params = tuple(Json(p) if isinstance(p, dict) else p for p in params)

        cursor.execute(query, params)

        if fetchone:
            result = cursor.fetchone()
        elif fetchall:
            result = cursor.fetchall()
        else:
            result = None

        conn.commit()
        return result

    except Exception as e:
        conn.rollback()
        print("🔥 DB error:", e, flush=True)
        return None

    finally:
        cursor.close()
        conn.close()

def build_plot_data_from_dict(plot_name):

    if plot_name not in plot_dict:
        return None

    data = plot_dict[plot_name]
    geom_obj = data.get("geometry")

    if not geom_obj:
        print("⚠ Missing geometry", flush=True)
        return None

    # ✅ HANDLE MULTIPLE FORMATS
    try:
        if hasattr(geom_obj, "getInfo"):
            geom_geojson = geom_obj.getInfo()
        else:
            geom_geojson = geom_obj  # already geojson
    except Exception as e:
        print("❌ Geometry getInfo failed:", e, flush=True)
        return None

    if not geom_geojson:
        return None

    # ✅ HANDLE FeatureCollection
    if geom_geojson.get("type") == "FeatureCollection":
        features = geom_geojson.get("features", [])
        if not features:
            print("❌ Empty FeatureCollection", flush=True)
            return None
        geom_geojson = features[0].get("geometry")

    # ✅ FINAL VALIDATION
    if "type" not in geom_geojson:
        print("❌ Invalid geometry format", flush=True)
        return None

    props = data.get("properties", {})

    return {
        "geometry": geom_geojson,
        "properties": {
            "plot_name": plot_name,
            "crop_type": props.get("crop_type_name") or "generic",
            "django_id": props.get("django_id")
        }
    }
# =====================================================
# 🔥 NEW: EXTRACT TILE + SENSOR
# =====================================================

def extract_metadata(geojson):
    try:
        features = geojson.get("features", [])
        if not features:
            return None, None

        props = features[0]["properties"]

        tile_url = props.get("tile_url") or props.get("tiles_url")
        sensor = props.get("sensor_used") or props.get("sensor")

        return tile_url, sensor

    except Exception:
        return None, None

# =====================================================
# STORE
# =====================================================

def store_results(results, analysis_type, plot_id, plot_name):

    if not results:
        print(f"⚠ No results for {analysis_type}", flush=True)
        return

    if isinstance(results, dict):
        results = [results]

    for geojson in results:

        features = geojson.get("features")
        if not features:
            continue

        props = features[0]["properties"]

        analysis_date = props.get("analysis_image_date") or props.get("latest_image_date")
        sensor = props.get("sensor_used") or props.get("sensor")

        if not analysis_date:
            print(f"⚠ Missing date in {analysis_type}", flush=True)
            continue

        # fallback sensor
        sensor = sensor or "unknown"

        final_type = f"{analysis_type}_{sensor.lower().replace('-', '')}"

        # 🔥 INSERT analysis
        run_query(
            """
            INSERT INTO analysis_results
            (plot_id, analysis_type, analysis_date, response_json)
            VALUES (%s,%s,%s,%s)
            ON CONFLICT (plot_id, analysis_type, analysis_date)
            DO UPDATE SET response_json = EXCLUDED.response_json
            """,
            (plot_id, final_type, analysis_date, Json(geojson))
        )

        # 🔥 UPDATE plots table with tile + sensor
        tile_url, sensor_used = extract_metadata(geojson)

        if tile_url or sensor_used:
            run_query(
                """
                UPDATE plots
                SET tile_url = COALESCE(%s, tile_url),
                    sensor_used = COALESCE(%s, sensor_used)
                WHERE id = %s
                """,
                (tile_url, sensor_used, plot_id)
            )

# =====================================================
# ANALYSIS
# =====================================================

def safe_analysis(fn, name, plot_name, plot_data, start, end, plot_id):
    try:
        print(f"🔎 Running {name}", flush=True)
        result = fn(plot_name, plot_data, start, end)
        store_results(result, name, plot_id, plot_name)
    except Exception as e:
        print(f"🔥 {name} failed: {e}", flush=True)

def run_today_analysis(plot_name, plot_data, plot_id):

    end = date.today()
    start = (end - timedelta(days=30)).isoformat()
    end = end.isoformat()

    with semaphore:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_ANALYSIS) as ex:

            ex.submit(safe_analysis, run_growth_analysis_by_plot, "growth", plot_name, plot_data, start, end, plot_id)
            ex.submit(safe_analysis, run_water_uptake_analysis_by_plot, "water", plot_name, plot_data, start, end, plot_id)
            ex.submit(safe_analysis, run_soil_moisture_analysis_by_plot, "soil", plot_name, plot_data, start, end, plot_id)
            ex.submit(safe_analysis, run_pest_detection_analysis_by_plot, "pest", plot_name, plot_data, start, end, plot_id)

# =====================================================
# PROCESS
# =====================================================

def process_plot(plot_name):

    global plot_dict

    print(f"🚀 Processing: {plot_name}", flush=True)

    plot_dict = plot_sync_service.get_plots_dict(force_refresh=True)

    plot_data = build_plot_data_from_dict(plot_name)
    if not plot_data:
        print(f"❌ No plot data", flush=True)
        return

    run_query(
        """
        INSERT INTO plots (plot_name, geojson)
        VALUES (%s,%s)
        ON CONFLICT (plot_name)
        DO UPDATE SET geojson = EXCLUDED.geojson
        """,
        (plot_name, Json(plot_data["geometry"]))
    )

    row = run_query("SELECT id FROM plots WHERE plot_name=%s", (plot_name,), fetchone=True)
    if not row:
        return

    plot_id = row["id"]

    run_today_analysis(plot_name, plot_data, plot_id)

    threading.Thread(
        target=run_monthly_backfill_for_plot,
        args=(plot_name, plot_data),
        daemon=True
    ).start()

# =====================================================
# WORKER
# =====================================================

def worker():
    while True:
        priority, timestamp, plot_name = task_queue.get()

        print(f"⚙️ Worker picked: {plot_name}", flush=True)

        try:
            process_plot(plot_name)
        except Exception as e:
            print(f"🔥 Worker error: {e}", flush=True)

        task_queue.task_done()

# =====================================================
# DAILY
# =====================================================

def daily_scheduler():

    global plot_dict

    while True:

        print("🕛 DAILY FETCH", flush=True)

        try:
            plot_dict = plot_sync_service.get_plots_dict(force_refresh=True)
        except Exception as e:
            print("❌ Fetch failed:", e)
            time.sleep(3600)
            continue

        for p in list(plot_dict.keys()):
            task_queue.put((10, time.time(), p))

        print(f"📦 Added {len(plot_dict)} plots", flush=True)

        time.sleep(86400)

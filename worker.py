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

# =====================================================
# ANALYSIS
# =====================================================

def safe_analysis(fn, name, plot_name, plot_data, start, end, plot_id):
    try:
        print(f"🔎 Running {name}", flush=True)
        result = fn(plot_name, plot_data, start, end)
        if result:
            store_results(result, name, plot_id)
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
# STORE RESULTS
# =====================================================

def extract_metadata(geojson):
    try:
        props = geojson["features"][0]["properties"]
        return props.get("tile_url"), props.get("sensor_used")
    except:
        return None, None

def store_results(results, analysis_type, plot_id):

    if isinstance(results, dict):
        results = [results]

    for geojson in results:

        features = geojson.get("features")
        if not features:
            continue

        props = features[0]["properties"]

        analysis_date = props.get("analysis_image_date") or props.get("latest_image_date")
        sensor = props.get("sensor_used") or "unknown"

        tile_url, sensor_used = extract_metadata(geojson)

        if not analysis_date:
            continue

        final_type = f"{analysis_type}_{sensor.lower()}"

        run_query(
            """
            INSERT INTO analysis_results
            (plot_id, analysis_type, analysis_date, response_json, tile_url, sensor_used)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (plot_id, analysis_type, analysis_date)
            DO UPDATE SET 
                response_json = EXCLUDED.response_json,
                tile_url = EXCLUDED.tile_url,
                sensor_used = EXCLUDED.sensor_used
            """,
            (plot_id, final_type, analysis_date, Json(geojson), tile_url, sensor_used)
        )

# =====================================================
# PROCESS (NO REFETCH BUG)
# =====================================================

def process_plot(plot_name):

    global plot_dict

    print(f"🚀 Processing: {plot_name}", flush=True)

    if plot_name not in plot_dict:
        print(f"❌ Not in memory: {plot_name}")
        return

    plot_data = plot_dict[plot_name]
    geom = plot_data.get("geometry")

    if not geom:
        print(f"⛔ No geometry: {plot_name}")
        return

    row = run_query(
        "SELECT id FROM plots WHERE plot_name=%s",
        (plot_name,),
        fetchone=True
    )

    if not row:
        print(f"❌ Not in DB: {plot_name}")
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
            plot_dict.clear()
            plot_dict.update(plot_sync_service.get_plots_dict(force_refresh=True))
        except Exception as e:
            print("❌ Fetch failed:", e)
            time.sleep(3600)
            continue

        for p, data in plot_dict.items():

            if not data.get("geometry"):
                continue

            task_queue.put((10, time.time(), p))

        print("📦 Daily queue added", flush=True)

        time.sleep(86400)

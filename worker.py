from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

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
# FASTAPI
# =====================================================

@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=daily_scheduler, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔥 TEMPORARY FIX (important)
    allow_credentials=False,  # must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)
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
# PROCESS
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

    # ✅ UPSERT FIRST (fix for "Not in DB")
    try:
        geom_geojson = geom.getInfo()
        area_ha = float(geom.area().divide(10000).getInfo())
        props = plot_data.get("properties", {})
    except Exception as e:
        print(f"❌ Geometry error: {e}")
        return

    run_query(
        """
        INSERT INTO plots 
        (plot_name, geojson, area_hectares, django_plot_id, plantation_date, crop_type)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (plot_name)
        DO UPDATE SET
            geojson = EXCLUDED.geojson,
            area_hectares = EXCLUDED.area_hectares,
            django_plot_id = EXCLUDED.django_plot_id,
            plantation_date = EXCLUDED.plantation_date,
            crop_type = EXCLUDED.crop_type
        """,
        (
            plot_name,
            Json(geom_geojson),
            area_ha,
            props.get("django_id"),
            props.get("plantation_date"),
            props.get("crop_type_name"),
        )
    )

    row = run_query(
        "SELECT id FROM plots WHERE plot_name=%s",
        (plot_name,),
        fetchone=True
    )

    plot_id = row["id"]

    run_today_analysis(plot_name, plot_data, plot_id)

    task_queue.put((20, time.time(), f"backfill::{plot_name}"))

# =====================================================
# WORKER
# =====================================================

def worker():
    while True:
        priority, timestamp, item = task_queue.get()

        try:
            if isinstance(item, str) and item.startswith("backfill::"):
                plot_name = item.split("::")[1]
                print(f"🧠 Backfill: {plot_name}")

                if plot_name in plot_dict:
                    run_monthly_backfill_for_plot(plot_name, plot_dict[plot_name])

            else:
                print(f"⚙️ Worker picked: {item}")
                process_plot(item)

        except Exception as e:
            print(f"🔥 Worker error: {e}", flush=True)

        task_queue.task_done()

# =====================================================
# DAILY SCHEDULER
# =====================================================

def daily_scheduler():

    global plot_dict

    while True:

        print("🕛 DAILY FETCH", flush=True)

        try:
            new_data = plot_sync_service.get_plots_dict(force_refresh=True)
        except Exception as e:
            print("❌ Fetch failed:", e)
            time.sleep(3600)
            continue

        new_plot_names = set(new_data.keys())

        existing_rows = run_query(
            "SELECT plot_name FROM plots",
            fetchall=True
        ) or []

        existing_plots = {row["plot_name"] for row in existing_rows}

        newly_added = new_plot_names - existing_plots

        print(f"🆕 New plots detected: {len(newly_added)}")

        plot_dict.clear()
        plot_dict.update(new_data)

        for p in newly_added:
            if plot_dict[p].get("geometry"):
                task_queue.put((1, time.time(), p))  # HIGH priority

        for p in new_plot_names:
            if p in newly_added:
                continue
            if plot_dict[p].get("geometry"):
                task_queue.put((10, time.time(), p))

        print("📦 Queue updated", flush=True)

        time.sleep(86400)

# =====================================================
# TRIGGER (NO INPUT REQUIRED)
# =====================================================

@app.post("/trigger-new-plot")
async def trigger_new():

    print("🚀 Manual trigger (AUTO DETECT NEW PLOTS)")

    try:
        new_data = plot_sync_service.get_plots_dict(force_refresh=True)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    new_plot_names = set(new_data.keys())

    existing_rows = run_query(
        "SELECT plot_name FROM plots",
        fetchall=True
    ) or []

    existing_plots = {row["plot_name"] for row in existing_rows}

    newly_added = new_plot_names - existing_plots

    plot_dict.clear()
    plot_dict.update(new_data)

    for p in newly_added:
        if plot_dict[p].get("geometry"):
            task_queue.put((1, time.time(), p))

    print(f"🆕 Trigger detected {len(newly_added)} new plots")

    return {
        "status": "queued",
        "new_plots": len(newly_added)
    }

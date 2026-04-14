from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from datetime import date, timedelta
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

WORKER_COUNT = 6
MAX_PARALLEL_ANALYSIS = 5
GLOBAL_LIMIT = 10

semaphore = threading.Semaphore(GLOBAL_LIMIT)

active_tasks = set()
active_lock = threading.Lock()

# =====================================================
# FASTAPI
# =====================================================

@asynccontextmanager
async def lifespan(app):
    print("🚀 Starting workers + scheduler", flush=True)

    for i in range(WORKER_COUNT):
        threading.Thread(target=worker, daemon=True).start()
        print(f"👷 Worker-{i+1} started")

    threading.Thread(target=daily_scheduler, daemon=True).start()

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
        print(f"🔎 {name} → {plot_name}", flush=True)

        result = fn(plot_name, plot_data, start, end)

        if result:
            store_results(result, name, plot_id)
        else:
            print(f"⚠ No result for {name}", flush=True)

    except Exception as e:
        print(f"🔥 {name} failed: {e}", flush=True)


def run_today_analysis(plot_name, plot_data, plot_id):
    end = date.today()
    start = (end - timedelta(days=30)).isoformat()
    end = end.isoformat()

    futures = []

    with semaphore:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_ANALYSIS) as ex:

            futures.append(ex.submit(safe_analysis, run_growth_analysis_by_plot, "growth", plot_name, plot_data, start, end, plot_id))
            futures.append(ex.submit(safe_analysis, run_water_uptake_analysis_by_plot, "water", plot_name, plot_data, start, end, plot_id))
            futures.append(ex.submit(safe_analysis, run_soil_moisture_analysis_by_plot, "soil", plot_name, plot_data, start, end, plot_id))
            futures.append(ex.submit(safe_analysis, run_pest_detection_analysis_by_plot, "pest", plot_name, plot_data, start, end, plot_id))

            for _ in as_completed(futures):
                pass

# =====================================================
# STORE
# =====================================================

def store_results(results, analysis_type, plot_id):
    if isinstance(results, dict):
        results = [results]

    for geojson in results:
        features = geojson.get("features")
        if not features:
            continue

        props = features[0]["properties"]

        analysis_date = (
            props.get("analysis_image_date")
            or props.get("latest_image_date")
            or props.get("analysis_dates", {}).get("latest_image_date")
        )

        if not analysis_date:
            continue

        run_query(
            """
            INSERT INTO analysis_results
            (plot_id, analysis_type, analysis_date, response_json)
            VALUES (%s,%s,%s,%s)
            ON CONFLICT (plot_id, analysis_type, analysis_date)
            DO UPDATE SET response_json = EXCLUDED.response_json
            """,
            (plot_id, analysis_type, analysis_date, Json(geojson))
        )

# =====================================================
# PROCESS
# =====================================================

def process_plot(plot_name):

    if plot_name not in plot_dict:
        print(f"❌ Not in memory: {plot_name}")
        return

    print(f"🚀 Processing: {plot_name}", flush=True)

    plot_data = plot_dict[plot_name]
    geom = plot_data.get("geometry")

    if not geom:
        return

    try:
        geom_geojson = geom.getInfo()
        area_ha = float(geom.area().divide(10000).getInfo())
        props = plot_data.get("properties", {})
    except Exception as e:
        print(f"❌ Geometry error: {e}")
        return

    existing = run_query(
        "SELECT id FROM plots WHERE plot_name=%s",
        (plot_name,),
        fetchone=True
    )

    is_new = existing is None

    run_query(
        """
        INSERT INTO plots
        (plot_name, geojson, area_hectares, django_plot_id, plantation_date, crop_type)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (plot_name)
        DO UPDATE SET geojson = EXCLUDED.geojson
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

    if not row:
        return

    plot_id = row["id"]

    run_today_analysis(plot_name, plot_data, plot_id)

    if is_new:
        print(f"⚡ Backfill NEW plot: {plot_name}", flush=True)
        run_monthly_backfill_for_plot(plot_name, plot_data)

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

        new_data = plot_sync_service.get_plots_dict(force_refresh=True)

        plot_dict.clear()
        plot_dict.update(new_data)

        for p in new_data.keys():
            task_queue.put((10, time.time(), p))

        time.sleep(86400)

# =====================================================
# CLEAR QUEUE
# =====================================================

def clear_queue():
    while not task_queue.empty():
        try:
            task_queue.get_nowait()
            task_queue.task_done()
        except:
            break

# =====================================================
# TRIGGER (FINAL PERFECT VERSION)
# =====================================================

@app.post("/trigger-new-plot")
async def trigger_new():

    def background():
        global plot_dict

        print("🚀 Manual trigger")

        # ✅ IMPORTANT: clear old queue
        clear_queue()

        max_attempts = 20  # ~1 min
        found_new = set()

        for attempt in range(max_attempts):

            print(f"🔄 Attempt {attempt+1}", flush=True)

            new_data = plot_sync_service.get_plots_dict(force_refresh=True)
            new_plot_names = set(new_data.keys())

            existing_rows = run_query(
                "SELECT plot_name FROM plots",
                fetchall=True
            ) or []

            existing_plots = {row["plot_name"] for row in existing_rows}

            new_plots = new_plot_names - existing_plots

            if new_plots:
                found_new = new_plots
                plot_dict.clear()
                plot_dict.update(new_data)
                break

            time.sleep(3)

        if not found_new:
            print("❌ No new plots found", flush=True)
            return

        print(f"✅ Found new plots: {len(found_new)}", flush=True)

        for p in found_new:
            task_queue.put((1, time.time(), p))
            task_queue.put((2, time.time(), f"backfill::{p}"))

        print(f"🚀 Queued {len(found_new)} new plots", flush=True)

    threading.Thread(target=background, daemon=True).start()

    return {"status": "started"}

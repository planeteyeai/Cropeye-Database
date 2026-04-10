from datetime import date, timedelta
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
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

# 🔥 PRIORITY QUEUE
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

# =====================================================
# CORS
# =====================================================

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
# BUILD
# =====================================================

def build_plot_data_from_dict(plot_name):

    if plot_name not in plot_dict:
        print(f"❌ Not in dict: {plot_name}", flush=True)
        return None

    data = plot_dict[plot_name]
    geom_obj = data.get("geometry")

    if not geom_obj:
        return None

    try:
        geom_geojson = geom_obj.getInfo()
    except Exception:
        return None

    if not geom_geojson:
        return None

    props = data.get("properties", {})

    return {
        "geometry": geom_geojson,
        "geom_type": geom_geojson.get("type"),
        "original_coords": geom_geojson.get("coordinates"),
        "properties": {
            "plot_name": plot_name,
            "crop_type": props.get("crop_type_name") or "generic",
            "django_id": props.get("django_id")
        }
    }

# =====================================================
# STORE
# =====================================================

def store_results(results, analysis_type, plot_id):

    if not results:
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

        if not analysis_date or not sensor:
            continue

        final_type = f"{analysis_type}_{sensor.lower().replace('-', '')}"

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

# =====================================================
# ANALYSIS
# =====================================================

def run_today_analysis(plot_name, plot_data, plot_id):

    end = date.today()
    start = (end - timedelta(days=30)).isoformat()
    end = end.isoformat()

    with semaphore:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_ANALYSIS) as ex:

            ex.submit(lambda: store_results(run_growth_analysis_by_plot(plot_name, plot_data, start, end), "growth", plot_id))
            ex.submit(lambda: store_results(run_water_uptake_analysis_by_plot(plot_name, plot_data, start, end), "water", plot_id))
            ex.submit(lambda: store_results(run_soil_moisture_analysis_by_plot(plot_name, plot_data, start, end), "soil", plot_id))
            ex.submit(lambda: store_results(run_pest_detection_analysis_by_plot(plot_name, plot_data, start, end), "pest", plot_id))

# =====================================================
# PROCESS
# =====================================================

def process_plot(plot_name):

    global plot_dict

    print(f"🚀 Processing: {plot_name}", flush=True)

    # 🔄 Always refresh before processing (important)
    plot_dict = plot_sync_service.get_plots_dict(force_refresh=True)

    plot_data = build_plot_data_from_dict(plot_name)
    if not plot_data:
        print(f"❌ Skipping {plot_name} (no data)", flush=True)
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

    # 🔥 backfill runs async but DOES NOT block worker
    threading.Thread(
        target=run_monthly_backfill_for_plot,
        args=(plot_name, plot_data),
        daemon=True
    ).start()

# =====================================================
# WORKER (CORE ENGINE)
# =====================================================

def worker():

    while True:
        priority, timestamp, plot_name = task_queue.get()

        print(f"⚙️ Worker picked: {plot_name} (priority {priority})", flush=True)

        try:
            process_plot(plot_name)
        except Exception as e:
            print(f"🔥 Worker error: {e}", flush=True)

        task_queue.task_done()

# =====================================================
# API (PRIORITY INSERT)
# =====================================================

@app.post("/trigger-new-plot")
async def trigger_new(request: Request):

    body = await request.json()
    plot_name = body.get("plot_name")

    if not plot_name:
        return {"status": "error", "message": "plot_name required"}

    # 🔥 HIGHEST PRIORITY
    task_queue.put((0, time.time(), plot_name))

    print(f"🚨 PRIORITY TASK ADDED: {plot_name}", flush=True)

    return {"status": "queued with priority", "plot": plot_name}

# =====================================================
# DAILY (LOW PRIORITY)
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

            # 🟡 LOW PRIORITY
            task_queue.put((10, time.time(), p))

        print(f"📦 Added {len(plot_dict)} plots to queue", flush=True)

        time.sleep(86400)
        

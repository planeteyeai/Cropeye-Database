from datetime import date, timedelta
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from fastapi import FastAPI
from pydantic import BaseModel
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
from shared_services import run_plot_sync

app = FastAPI()

MAX_PARALLEL_ANALYSIS = 3
GLOBAL_LIMIT = 4

semaphore = threading.Semaphore(GLOBAL_LIMIT)

known_plot_ids = set()
priority_queue = Queue()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =====================================================
# HEALTH
# =====================================================

@app.get("/")
def health():
    return {"status": "alive"}

# =====================================================
# REQUEST MODEL
# =====================================================

class PlotRequest(BaseModel):
    plot_name: str

# =====================================================
# DB HELPER
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
# INITIAL LOAD
# =====================================================

def initial_load():

    print("🔄 Initial sync...", flush=True)

    try:
        run_plot_sync()
    except Exception as e:
        print("❌ Sync failed:", e, flush=True)

    rows = run_query("SELECT id FROM plots", fetchall=True) or []

    for r in rows:
        known_plot_ids.add(r["id"])

    print(f"✅ Loaded {len(known_plot_ids)} plots", flush=True)

# =====================================================
# GEOJSON BUILDER
# =====================================================

def build_plot_data(row):

    geojson = row.get("geojson")
    if not geojson:
        return None

    geometry = None

    if isinstance(geojson, dict):
        if geojson.get("type") in ["Polygon", "MultiPolygon"]:
            geometry = geojson
        elif geojson.get("type") == "Feature":
            geometry = geojson.get("geometry")
        elif geojson.get("type") == "FeatureCollection":
            features = geojson.get("features", [])
            if features:
                geometry = features[0].get("geometry")

    if not geometry or not geometry.get("coordinates"):
        return None

    django_id = row.get("django_plot_id")
    if not django_id:
        return None

    return {
        "geometry": geometry,
        "geom_type": geometry.get("type"),
        "original_coords": geometry.get("coordinates"),
        "properties": {
            "django_id": django_id,
            "plot_name": row.get("plot_name"),
            "crop_type": row.get("crop_type"),
        }
    }

# =====================================================
# STORE RESULTS
# =====================================================

def store_results(results, analysis_type, plot_id):

    try:
        if not results:
            return

        if isinstance(results, dict):
            results = [results]

        for geojson in results:

            features = geojson.get("features")
            if not features:
                continue

            props = features[0].get("properties", {})

            analysis_date = (
                props.get("analysis_image_date")
                or props.get("latest_image_date")
            )

            sensor = props.get("sensor_used") or props.get("sensor")

            if not analysis_date or not sensor:
                continue

            final_analysis_type = f"{analysis_type}_{sensor.lower().replace('-', '')}"

            run_query(
                """
                INSERT INTO analysis_results
                (plot_id, analysis_type, analysis_date, response_json)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT (plot_id, analysis_type, analysis_date)
                DO UPDATE SET response_json = EXCLUDED.response_json
                """,
                (plot_id, final_analysis_type, analysis_date, Json(geojson))
            )

            print(f"✅ Stored {final_analysis_type} for plot {plot_id}", flush=True)

    except Exception as e:
        print("🔥 store_results error:", e, flush=True)

# =====================================================
# ANALYSIS
# =====================================================

def run_today_analysis_for_plot(plot_name, plot_data, plot_id):

    try:
        with semaphore:

            print(f"🌅 Running TODAY analysis for {plot_name}", flush=True)

            end_date = date.today()
            start_date = (end_date - timedelta(days=30)).isoformat()
            end_date = end_date.isoformat()

            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_ANALYSIS) as executor:

                executor.submit(lambda: store_results(
                    run_growth_analysis_by_plot(plot_name, plot_data, start_date, end_date),
                    "growth", plot_id))

                executor.submit(lambda: store_results(
                    run_water_uptake_analysis_by_plot(plot_name, plot_data, start_date, end_date),
                    "water", plot_id))

                executor.submit(lambda: store_results(
                    run_soil_moisture_analysis_by_plot(plot_name, plot_data, start_date, end_date),
                    "soil", plot_id))

                executor.submit(lambda: store_results(
                    run_pest_detection_analysis_by_plot(plot_name, plot_data, start_date, end_date),
                    "pest", plot_id))

    except Exception as e:
        print(f"🔥 Analysis failed for {plot_name}: {e}", flush=True)

# =====================================================
# PROCESS PLOT
# =====================================================

def process_plot(plot_name):

    try:
        print(f"⚙️ Processing plot: {plot_name}", flush=True)

        row = run_query(
            """
            SELECT id, geojson, django_plot_id, plot_name, crop_type
            FROM plots
            WHERE plot_name=%s
            """,
            (plot_name,),
            fetchone=True
        )

        if not row:
            print("❌ Plot not found in DB", flush=True)
            return

        plot_data = build_plot_data(row)
        if not plot_data:
            print("❌ Invalid plot data", flush=True)
            return

        plot_id = row["id"]

        run_today_analysis_for_plot(plot_name, plot_data, plot_id)

        threading.Thread(
            target=run_monthly_backfill_for_plot,
            args=(plot_name, plot_data),
            daemon=True
        ).start()

    except Exception as e:
        print(f"🔥 process_plot error: {e}", flush=True)

# =====================================================
# API (FIXED)
# =====================================================

@app.post("/trigger-new-plot")
def trigger_new_plot(data: PlotRequest):

    print(f"🚀 Trigger received: {data.plot_name}", flush=True)

    threading.Thread(
        target=process_plot,
        args=(data.plot_name,),
        daemon=True
    ).start()

    return {"status": "processing started"}

# =====================================================
# DAILY JOB
# =====================================================

def daily_scheduler():

    while True:

        print("🕛 Running DAILY job", flush=True)

        rows = run_query(
            """
            SELECT plot_name 
            FROM plots 
            WHERE geojson IS NOT NULL
            AND geojson::text != '{}'
            AND django_plot_id IS NOT NULL
            """,
            fetchall=True
        ) or []

        for r in rows:
            process_plot(r["plot_name"])

        time.sleep(86400)

# =====================================================
# STARTUP (IMPORTANT FIX)
# =====================================================

@app.on_event("startup")
def startup_event():
    print("🚀 App started", flush=True)
    threading.Thread(target=initial_load, daemon=True).start()
    threading.Thread(target=daily_scheduler, daemon=True).start()

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("🌐 Worker starting...", flush=True)

    uvicorn.run(app, host="0.0.0.0", port=8000)

from datetime import date, timedelta
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

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
        print("❌ Sync failed:", e)

    rows = run_query("SELECT id FROM plots", fetchall=True) or []

    for r in rows:
        known_plot_ids.add(r["id"])

    print(f"✅ Loaded {len(known_plot_ids)} plots", flush=True)

# =====================================================
# GEOJSON FIX (FINAL)
# =====================================================

def build_plot_data(row):

    geojson = row.get("geojson")
    django_id = row.get("django_plot_id")  # ✅ FIXED

    if not geojson:
        print("⚠ Missing geojson")
        return None

    if not django_id:
        print("⚠ Missing django_plot_id")
        return None

    geometry = None

    if geojson.get("type") in ["Polygon", "MultiPolygon"]:
        geometry = geojson

    elif geojson.get("type") == "Feature":
        geometry = geojson.get("geometry")

    elif geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])
        if features:
            geometry = features[0].get("geometry")

    # ✅ STRICT VALIDATION
    if not geometry or "coordinates" not in geometry:
        print("⚠ Invalid geometry structure")
        return None

    if not geometry["coordinates"]:
        print("⚠ Empty coordinates")
        return None

    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "django_id": django_id  # ✅ FIXED
            }
        }]
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

            if not analysis_date:
                continue

            run_query(
                """
                INSERT INTO analysis_results
                (plot_id,analysis_type,analysis_date,response_json)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT DO NOTHING
                """,
                (plot_id, analysis_type, analysis_date, Json(geojson))
            )

            print(f"✅ Stored {analysis_type} for plot {plot_id}", flush=True)

    except Exception as e:
        print("🔥 store_results error:", e, flush=True)

# =====================================================
# ANALYSIS
# =====================================================

def run_today_analysis_for_plot(plot_name, plot_data, plot_id):

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

# =====================================================
# PROCESS PLOT (FINAL SAFE)
# =====================================================

def process_plot(plot_name):

    print(f"➡ Processing {plot_name}", flush=True)

    row = run_query(
        "SELECT id, geojson, django_plot_id FROM plots WHERE plot_name=%s",
        (plot_name,),
        fetchone=True
    )

    if not row:
        print(f"❌ Plot not found {plot_name}", flush=True)
        return

    plot_data = build_plot_data(row)

    if not plot_data:
        print(f"❌ Skipping {plot_name} due to invalid data", flush=True)
        return

    plot_id = row["id"]

    run_today_analysis_for_plot(plot_name, plot_data, plot_id)

    threading.Thread(
        target=run_monthly_backfill_for_plot,
        args=(plot_name, plot_data),
        daemon=True
    ).start()

# =====================================================
# API TRIGGER
# =====================================================

@app.post("/trigger-new-plot")
def trigger_new_plot(data: PlotRequest):

    print(f"🚀 PRIORITY trigger for {data.plot_name}", flush=True)

    priority_queue.put(data.plot_name)

    return {"status": "queued"}

# =====================================================
# WORKER LOOP
# =====================================================

def worker_loop():

    while True:

        try:

            if not priority_queue.empty():
                process_plot(priority_queue.get())
                continue

            rows = run_query("SELECT id, plot_name FROM plots", fetchall=True) or []

            current_ids = set(r["id"] for r in rows)
            new_ids = current_ids - known_plot_ids

            for r in rows:
                if r["id"] in new_ids:
                    process_plot(r["plot_name"])

            known_plot_ids.update(new_ids)

            time.sleep(5)

        except Exception as e:
            print("🔥 Worker error:", e, flush=True)
            time.sleep(5)

# =====================================================
# DAILY JOB
# =====================================================

def daily_scheduler():

    while True:

        print("🕛 Running DAILY job for ALL plots", flush=True)

        rows = run_query(
            "SELECT plot_name FROM plots WHERE geojson IS NOT NULL",
            fetchall=True
        ) or []

        for r in rows:
            priority_queue.put(r["plot_name"])

        time.sleep(86400)

# =====================================================
# STARTUP
# =====================================================

def start_background_jobs():

    time.sleep(5)

    initial_load()

    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=daily_scheduler, daemon=True).start()

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("🌐 Worker starting...", flush=True)

    threading.Thread(target=start_background_jobs, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000)

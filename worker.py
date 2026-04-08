from datetime import date, timedelta
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException
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
# APP INIT
# =====================================================

app = FastAPI()

plot_sync_service = PlotSyncService()
plot_dict = {}

priority_processing = False

MAX_PARALLEL_ANALYSIS = 3
GLOBAL_LIMIT = 4

semaphore = threading.Semaphore(GLOBAL_LIMIT)

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
# HEALTH
# =====================================================

@app.get("/")
def health():
    return {"status": "alive"}

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
# 🔥 REFRESH FROM DJANGO (NEW)
# =====================================================

@app.post("/refresh-from-django")
async def refresh_from_django():
    try:
        global plot_dict

        print("🔄 FULL REFRESH from Django...", flush=True)

        plot_dict = plot_sync_service.get_plots_dict(force_refresh=True)

        print(f"✅ Loaded {len(plot_dict)} plots from Django", flush=True)

        return {
            "status": "success",
            "plot_count": len(plot_dict)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# GEOJSON BUILDER
# =====================================================

def build_plot_data_from_dict(plot_name):

    global plot_dict

    if plot_name not in plot_dict:
        print("❌ Plot not in refreshed dict", flush=True)
        return None

    data = plot_dict[plot_name]

    try:
        geom = data.get("geometry")
        geom_geojson = geom.getInfo()

        props = data.get("properties", {})

        return {
            "geometry": geom_geojson,
            "geom_type": geom_geojson.get("type"),
            "original_coords": geom_geojson.get("coordinates"),
            "properties": {
                "django_id": props.get("django_id"),
                "plot_name": plot_name,
                "crop_type": props.get("crop_type_name"),
            }
        }

    except Exception as e:
        print("🔥 build_plot_data error:", e, flush=True)
        return None

# =====================================================
# STORE RESULTS
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

        props = features[0].get("properties", {})

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

        print(f"✅ Stored {final_type} for plot {plot_id}", flush=True)

# =====================================================
# ANALYSIS
# =====================================================

def run_today_analysis(plot_name, plot_data, plot_id):

    with semaphore:

        print(f"🌅 TODAY analysis: {plot_name}", flush=True)

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
# PROCESS (PRIORITY)
# =====================================================

def process_plot(plot_name):

    print(f"⚙️ PRIORITY processing: {plot_name}", flush=True)

    plot_data = build_plot_data_from_dict(plot_name)

    if not plot_data:
        return

    # Insert plot in DB first
    run_query(
        """
        INSERT INTO plots (plot_name, geojson)
        VALUES (%s, %s)
        ON CONFLICT (plot_name)
        DO UPDATE SET geojson = EXCLUDED.geojson
        """,
        (plot_name, Json(plot_data["geometry"]))
    )

    row = run_query(
        "SELECT id FROM plots WHERE plot_name=%s",
        (plot_name,),
        fetchone=True
    )

    plot_id = row["id"]

    run_today_analysis(plot_name, plot_data, plot_id)

    threading.Thread(
        target=run_monthly_backfill_for_plot,
        args=(plot_name, plot_data),
        daemon=True
    ).start()

# =====================================================
# 🚀 TRIGGER (FINAL FIXED)
# =====================================================

@app.post("/trigger-new-plot")
async def trigger_new_plot(request: Request):

    global priority_processing

    body = await request.json()
    plot_name = body.get("plot_name")

    if not plot_name:
        return {"status": "error", "message": "plot_name required"}

    print(f"🚀 PRIORITY TRIGGER: {plot_name}", flush=True)

    def pipeline():
        global priority_processing

        try:
            priority_processing = True

            # ✅ STEP 1: FORCE REFRESH
            print("🔄 Refreshing Django...", flush=True)
            global plot_dict
            plot_dict = plot_sync_service.get_plots_dict(force_refresh=True)

            # ✅ STEP 2: PROCESS IMMEDIATELY
            process_plot(plot_name)

            print(f"🔥 DONE: {plot_name}", flush=True)

        finally:
            priority_processing = False

    threading.Thread(target=pipeline, daemon=True).start()

    return {"status": "started", "plot": plot_name}

# =====================================================
# DAILY JOB
# =====================================================

def daily_scheduler():

    global priority_processing

    while True:

        if priority_processing:
            print("⏸ Skipping daily (priority running)", flush=True)
            time.sleep(10)
            continue

        print("🕛 DAILY RUN", flush=True)

        rows = run_query("SELECT plot_name FROM plots", fetchall=True) or []

        for r in rows:

            if priority_processing:
                break

            process_plot(r["plot_name"])

        time.sleep(86400)

# =====================================================
# STARTUP
# =====================================================

@app.on_event("startup")
def startup():
    print("🚀 Worker started", flush=True)
    threading.Thread(target=daily_scheduler, daemon=True).start()

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

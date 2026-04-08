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
from typing import Optional
from fastapi import Request
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
priority_processing = False
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
    if plot_name in known_plot_ids:
    print(f"⚠ Already processed recently: {plot_name}", flush=True)
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

def sync_single_plot_from_django(plot_name):

    print(f"🔄 Fetching single plot from Django: {plot_name}", flush=True)

    from shared_services import PlotSyncService

    service = PlotSyncService()

    plots = service.get_plots_dict(force_refresh=True)

    if plot_name not in plots:
        print("❌ Plot not found in Django API", flush=True)
        return None

    data = plots[plot_name]

    try:
        geom = data.get("geometry")
        geom_geojson = geom.getInfo()
        area_ha = float(geom.area().divide(10000).getInfo())

        props = data.get("properties", {})

        django_id = props.get("django_id")
        plantation_date = props.get("plantation_date")
        crop_type = props.get("crop_type_name")

        run_query(
            """
            INSERT INTO plots
            (plot_name, geojson, area_hectares, django_plot_id, plantation_date, crop_type)
            VALUES (%s, %s, %s, %s, %s, %s)
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
                str(django_id),
                plantation_date,
                crop_type
            )
        )

        print(f"✅ Plot synced: {plot_name}", flush=True)

        return True

    except Exception as e:
        print(f"🔥 Sync single plot failed: {e}", flush=True)
        return None


@app.post("/trigger-new-plot")
async def trigger_new_plot(request: Request):

    global priority_processing

    # ---------------------------
    # READ BODY SAFELY
    # ---------------------------
    try:
        body = await request.json()
        plot_name = body.get("plot_name")
    except Exception:
        return {"status": "error", "message": "Invalid JSON"}

    if not plot_name:
        return {"status": "error", "message": "plot_name required"}

    print(f"🚀 PRIORITY trigger: {plot_name}", flush=True)

    def full_pipeline():
        global priority_processing

        try:
            priority_processing = True   # 🚨 BLOCK OTHER JOBS

            # ---------------------------
            # FAST SYNC (ONLY THIS PLOT)
            # ---------------------------
            print(f"🔄 Syncing {plot_name}", flush=True)

            synced = sync_single_plot_from_django(plot_name)

            if not synced:
                print("❌ Sync failed", flush=True)
                priority_processing = False
                return

            print(f"✅ Sync done: {plot_name}", flush=True)

            # ---------------------------
            # PRIORITY PROCESS
            # ---------------------------
            process_plot(plot_name)

            print(f"🔥 PRIORITY DONE: {plot_name}", flush=True)

        except Exception as e:
            print(f"🔥 trigger error: {e}", flush=True)

        finally:
            priority_processing = False   # ✅ RESUME NORMAL

    threading.Thread(target=full_pipeline, daemon=True).start()

    return {
        "status": "priority processing started",
        "plot_name": plot_name
    }
# =====================================================
# DAILY JOB
# =====================================================
def daily_scheduler():

    global priority_processing

    while True:

        if priority_processing:
            print("⏸ Skipping daily job (priority running)", flush=True)
            time.sleep(10)
            continue

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

            if priority_processing:
                print("⏸ Breaking daily loop (priority triggered)", flush=True)
                break

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

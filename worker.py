from datetime import date, timedelta
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
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

app = FastAPI()

plot_sync_service = PlotSyncService()
plot_dict = {}

priority_processing = False
STOP_ALL = False   # 🔥 NEW

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
# FETCH
# =====================================================

def fetch_latest_plot():
    import requests

    url = f"{plot_sync_service.django_api_url}/api/plots/public/?ordering=-id&page_size=1"

    r = requests.get(url, timeout=10)
    data = r.json()

    results = data.get("results", [])
    if not results:
        return None

    processed = plot_sync_service._process_plots_response({"results": results})
    return processed

# =====================================================
# BUILD
# =====================================================

def build_plot_data_from_dict(plot_name):

    if plot_name not in plot_dict:
        return None

    data = plot_dict[plot_name]
    geom_obj = data.get("geometry")

    try:
        geom_geojson = geom_obj.getInfo()
    except:
        return None

    props = data.get("properties", {})

    crop_type = props.get("crop_type_name") or "generic"

    return {
        "geometry": geom_geojson,
        "properties": {
            "plot_name": plot_name,
            "crop_type": crop_type,
        }
    }

# =====================================================
# STORE
# =====================================================

def store_results(results, analysis_type, plot_id):

    if STOP_ALL:
        return

    if not results:
        return

    if isinstance(results, dict):
        results = [results]

    for geojson in results:

        if STOP_ALL:
            return

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

    if STOP_ALL:
        return

    with semaphore:

        end = date.today()
        start = (end - timedelta(days=30)).isoformat()
        end = end.isoformat()

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_ANALYSIS) as ex:

            ex.submit(lambda: store_results(run_growth_analysis_by_plot(plot_name, plot_data, start, end), "growth", plot_id))
            ex.submit(lambda: store_results(run_water_uptake_analysis_by_plot(plot_name, plot_data, start, end), "water", plot_id))
            ex.submit(lambda: store_results(run_soil_moisture_analysis_by_plot(plot_name, plot_data, start, end), "soil", plot_id))
            ex.submit(lambda: store_results(run_pest_detection_analysis_by_plot(plot_name, plot_data, start, end), "pest", plot_id))

# =====================================================
# PROCESS
# =====================================================

def process_plot(plot_name):

    if STOP_ALL:
        return

    print(f"🚀 Processing: {plot_name}", flush=True)

    plot_data = build_plot_data_from_dict(plot_name)
    if not plot_data:
        return

    run_query(
        "INSERT INTO plots (plot_name, geojson) VALUES (%s, %s) ON CONFLICT (plot_name) DO UPDATE SET geojson = EXCLUDED.geojson",
        (plot_name, Json(plot_data["geometry"]))
    )

    row = run_query("SELECT id FROM plots WHERE plot_name=%s", (plot_name,), fetchone=True)
    if not row:
        return

    plot_id = row["id"]

    run_today_analysis(plot_name, plot_data, plot_id)

    # 🔥 BACKFILL with STOP CHECK
    def safe_backfill():
        if STOP_ALL:
            return
        run_monthly_backfill_for_plot(plot_name, plot_data)

    threading.Thread(target=safe_backfill, daemon=True).start()

# =====================================================
# TRIGGER (PRIORITY)
# =====================================================

def trigger_pipeline():

    global priority_processing, STOP_ALL, plot_dict

    print("🔥 PRIORITY MODE ON", flush=True)

    STOP_ALL = True          # ⛔ STOP EVERYTHING
    priority_processing = True

    time.sleep(2)            # let threads stop

    latest = fetch_latest_plot()
    if not latest:
        STOP_ALL = False
        priority_processing = False
        return

    plot_name = list(latest.keys())[0]

    plot_dict[plot_name] = latest[plot_name]

    process_plot(plot_name)

    print(f"🔥 DONE PRIORITY {plot_name}", flush=True)

    STOP_ALL = False         # ✅ RESUME
    priority_processing = False

@app.post("/trigger-new-plot")
async def trigger_new():
    threading.Thread(target=trigger_pipeline, daemon=True).start()
    return {"status": "priority started"}

# =====================================================
# DAILY
# =====================================================

def daily_scheduler():

    global plot_dict

    while True:

        if priority_processing or STOP_ALL:
            time.sleep(5)
            continue

        print("🕛 DAILY START", flush=True)

        plot_dict = plot_sync_service.get_plots_dict(force_refresh=True)

        for p in list(plot_dict.keys()):

            if STOP_ALL:
                print("⛔ DAILY INTERRUPTED", flush=True)
                break

            process_plot(p)

        time.sleep(86400)

# =====================================================
# START
# =====================================================

@app.on_event("startup")
def startup():
    threading.Thread(target=daily_scheduler, daemon=True).start()

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

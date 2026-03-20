from datetime import date, timedelta
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI()

# =====================================================
# CONFIG
# =====================================================

MAX_PARALLEL_ANALYSIS = 4

known_plot_ids = set()

# Priority queue for new plots
priority_queue = []

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
        print("🔥 DB error:", e)
        return None

    finally:
        cursor.close()
        conn.close()

# =====================================================
# INITIAL LOAD
# =====================================================

print("🔄 Initial sync...")
run_plot_sync()

rows = run_query("SELECT id FROM plots", fetchall=True) or []

for r in rows:
    known_plot_ids.add(r["id"])

print(f"✅ Loaded {len(known_plot_ids)} plots")

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

# =====================================================
# ANALYSIS
# =====================================================

def run_today_analysis_for_plot(plot_name, plot_data, plot_id):

    print(f"🌅 Running TODAY analysis for {plot_name}")

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
# PROCESS PLOT
# =====================================================

def process_plot(plot_name):

    row = run_query(
        "SELECT id, geojson FROM plots WHERE plot_name=%s",
        (plot_name,),
        fetchone=True
    )

    if not row or not row["geojson"]:
        print(f"⚠ Skipping {plot_name}")
        return

    plot_id = row["id"]

    plot_data = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": row["geojson"],
            "properties": {}
        }]
    }

    run_today_analysis_for_plot(plot_name, plot_data, plot_id)

    threading.Thread(
        target=run_monthly_backfill_for_plot,
        args=(plot_name, plot_data)
    ).start()

# =====================================================
# INSTANT TRIGGER (FROM DJANGO)
# =====================================================

@app.post("/trigger-new-plot")
def trigger_new_plot(data: PlotRequest):

    print(f"🚀 PRIORITY trigger for {data.plot_name}")

    priority_queue.append(data.plot_name)

    return {"status": "queued"}

# =====================================================
# WORKER LOOP (PRIORITY FIRST)
# =====================================================

def worker_loop():

    while True:

        try:

            # 🔥 PRIORITY FIRST
            if priority_queue:
                plot_name = priority_queue.pop(0)
                process_plot(plot_name)
                continue

            # Normal DB scan
            rows = run_query(
                "SELECT id, plot_name FROM plots",
                fetchall=True
            ) or []

            current_ids = set(r["id"] for r in rows)

            new_ids = current_ids - known_plot_ids

            for r in rows:
                if r["id"] in new_ids:
                    process_plot(r["plot_name"])

            known_plot_ids.update(new_ids)

            time.sleep(10)

        except Exception as e:
            print("🔥 Worker error:", e)
            time.sleep(5)

# =====================================================
# DAILY JOB (ALL PLOTS)
# =====================================================

def daily_scheduler():

    while True:

        print("🕛 Running DAILY job for ALL plots")

        rows = run_query(
            "SELECT plot_name FROM plots",
            fetchall=True
        ) or []

        for r in rows:
            process_plot(r["plot_name"])

        time.sleep(86400)  # 24 hours

# =====================================================
# STARTUP
# =====================================================

def start_background_jobs():
    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=daily_scheduler, daemon=True).start()

start_background_jobs()

from datetime import date, timedelta
import time
import ee
from concurrent.futures import ThreadPoolExecutor
from psycopg2.extras import RealDictCursor, Json

from gee_growth import (
    run_growth_analysis_by_plot,
    run_water_uptake_analysis_by_plot,
    run_soil_moisture_analysis_by_plot,
    run_pest_detection_analysis_by_plot
)

from shared_services import PlotSyncService, run_plot_sync
from db import get_connection
from Admin import run_monthly_backfill_for_plot

# =====================================================
# CONFIG
# =====================================================

BATCH_SIZE = 20
REQUEST_DELAY = 3
MAX_WORKERS = 6
MAX_RETRY = 3

# =====================================================
# STEP 1 — INTERNAL PLOT SYNC
# =====================================================

print("🔄 Running plot sync...", flush=True)

try:
    sync_result = run_plot_sync()
    print("✅ Sync result:", sync_result, flush=True)
except Exception as e:
    print("❌ Plot sync failed:", str(e), flush=True)

plots = PlotSyncService().get_plots_dict(force_refresh=True)

# =====================================================
# DB HELPER
# =====================================================

def run_query(query, params=None, fetchone=False, fetchall=False):

    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:

        if params:
            new_params = []
            for p in params:
                if isinstance(p, dict):
                    new_params.append(Json(p))
                else:
                    new_params.append(p)
            params = tuple(new_params)

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

        sensor_used = props.get("sensor", "Unknown")

        tile_url = props.get("tile_url")

        if not analysis_date:
            continue

        run_query(
            """
            INSERT INTO satellite_images
            (plot_id,satellite,satellite_date)
            VALUES (%s,%s,%s)
            ON CONFLICT DO NOTHING
            """,
            (plot_id, sensor_used, analysis_date)
        )

        run_query(
            """
            INSERT INTO analysis_results
            (plot_id,analysis_type,analysis_date,sensor_used,tile_url,response_json)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (plot_id,analysis_type,analysis_date) DO NOTHING
            """,
            (plot_id, analysis_type, analysis_date, sensor_used, tile_url, Json(geojson))
        )

        print(f"✅ Stored {analysis_type} {analysis_date}", flush=True)

# =====================================================
# TODAY ANALYSIS
# =====================================================

def run_today_analysis_for_plot(plot_name, plot_data, plot_id):

    print(f"🌅 Running TODAY analysis for {plot_name}")

    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    start_date = start_date.isoformat()
    end_date = end_date.isoformat()

    try:

        res = run_growth_analysis_by_plot(plot_name, plot_data, start_date, end_date)
        store_results(res, "growth", plot_id)

        res = run_water_uptake_analysis_by_plot(plot_name, plot_data, start_date, end_date)
        store_results(res, "water_uptake", plot_id)

        res = run_soil_moisture_analysis_by_plot(plot_name, plot_data, start_date, end_date)
        store_results(res, "soil_moisture", plot_id)

        res = run_pest_detection_analysis_by_plot(plot_name, plot_data, start_date, end_date)
        store_results(res, "pest_detection", plot_id)

        print(f"✅ TODAY analysis complete {plot_name}")

    except Exception as e:
        print(f"🔥 TODAY analysis failed {plot_name}: {e}")

# =====================================================
# NEW PLOT BACKFILL
# =====================================================

def trigger_backfill_for_new_plots(plots):

    for plot_name, plot_data in plots.items():

        try:

            row = run_query(
                "SELECT id,backfill_completed FROM plots WHERE plot_name=%s",
                (plot_name,),
                fetchone=True
            )

            if not row:
                continue

            plot_id = row["id"]

            if not row["backfill_completed"]:

                print(f"🆕 New plot detected → {plot_name}")

                run_today_analysis_for_plot(plot_name, plot_data, plot_id)

                run_monthly_backfill_for_plot(plot_name, plot_data)

                run_query(
                    "UPDATE plots SET backfill_completed=TRUE WHERE id=%s",
                    (plot_id,)
                )

            else:

                print(f"✅ Backfill already done {plot_name}", flush=True)

        except Exception as e:
            print(f"🔥 Backfill failed {plot_name}: {e}", flush=True)

trigger_backfill_for_new_plots(plots)

# =====================================================
# CREATE QUEUE
# =====================================================

print("📦 Creating analysis queue...", flush=True)

analysis_types = ["growth","water_uptake","soil_moisture","pest_detection"]

for plot_name in plots.keys():

    row = run_query(
        "SELECT id FROM plots WHERE plot_name=%s",
        (plot_name,),
        fetchone=True
    )

    if not row:
        continue

    plot_id = row["id"]

    for analysis in analysis_types:

        run_query(
            """
            INSERT INTO analysis_queue
            (plot_id,plot_name,analysis_type,status,priority)
            VALUES (%s,%s,%s,'pending',1)
            ON CONFLICT (plot_id,analysis_type)
            DO UPDATE SET status='pending'
            """,
            (plot_id, plot_name, analysis)
        )

print("✅ Queue created", flush=True)

# =====================================================
# JOB FUNCTIONS
# =====================================================

today = date.today().isoformat()
start_date = (date.today() - timedelta(days=30)).isoformat()
pest_start = (date.today() - timedelta(days=15)).isoformat()

def fetch_jobs():

    return run_query(
        """
        SELECT * FROM analysis_queue
        WHERE status='pending'
        ORDER BY priority DESC
        LIMIT %s
        """,
        (BATCH_SIZE,),
        fetchall=True
    )

def lock_job(job_id):

    res = run_query(
        """
        UPDATE analysis_queue
        SET status='processing'
        WHERE id=%s AND status='pending'
        RETURNING id
        """,
        (job_id,),
        fetchone=True
    )

    return res is not None

def mark_completed(job_id):

    run_query(
        "UPDATE analysis_queue SET status='completed' WHERE id=%s",
        (job_id,)
    )

# =====================================================
# JOB PROCESSOR
# =====================================================

def process_job(job):

    job_id = job["id"]
    plot_name = job["plot_name"]
    analysis_type = job["analysis_type"]
    plot_id = job["plot_id"]

    if not lock_job(job_id):
        return

    try:

        plot_data = plots.get(plot_name)

        if analysis_type == "growth":
            results = run_growth_analysis_by_plot(plot_name, plot_data, start_date, today)

        elif analysis_type == "water_uptake":
            results = run_water_uptake_analysis_by_plot(plot_name, plot_data, start_date, today)

        elif analysis_type == "soil_moisture":
            results = run_soil_moisture_analysis_by_plot(plot_name, plot_data, start_date, today)

        elif analysis_type == "pest_detection":
            results = run_pest_detection_analysis_by_plot(plot_name, plot_data, pest_start, today)

        store_results(results, analysis_type, plot_id)

        mark_completed(job_id)

        time.sleep(REQUEST_DELAY)

    except Exception as e:

        print("🔥 Job failed:", e, flush=True)

# =====================================================
# WORKER LOOP
# =====================================================

print("🚀 PERSISTENT WORKER STARTED", flush=True)

while True:

    try:

        print("🔄 Syncing plots...", flush=True)

        run_plot_sync()

        plots = PlotSyncService().get_plots_dict(force_refresh=True)

        trigger_backfill_for_new_plots(plots)

        jobs = fetch_jobs()

        if jobs:

            print(f"📦 Processing {len(jobs)} jobs", flush=True)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                executor.map(process_job, jobs)

        else:

            print("⏳ No jobs — sleeping 5 minutes", flush=True)

        time.sleep(300)   # 5 minutes

    except Exception as e:

        print("🔥 Worker error:", e, flush=True)

        time.sleep(60)

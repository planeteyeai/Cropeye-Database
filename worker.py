from datetime import date, timedelta
import time
from concurrent.futures import ThreadPoolExecutor

from psycopg2.extras import RealDictCursor

from gee_growth import (
    run_growth_analysis_by_plot,
    run_water_uptake_analysis_by_plot,
    run_soil_moisture_analysis_by_plot,
    run_pest_detection_analysis_by_plot
)

from shared_services import PlotSyncService, run_plot_sync
from db import get_connection
from Admin import run_monthly_backfill_for_plot
from psycopg2.extras import Json
# =====================================================
# CONFIG
# =====================================================

BATCH_SIZE = 20
REQUEST_DELAY = 3
MAX_WORKERS = 5
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

def run_query(query, params=None):

    conn = get_connection()
    cursor = conn.cursor()

    try:

        if params:
            new_params = []

            for p in params:

                if isinstance(p, dict):
                    new_params.append(Json(p))   # ✅ convert dict → JSON
                else:
                    new_params.append(p)

            params = tuple(new_params)

        cursor.execute(query, params)

        conn.commit()

    except Exception as e:

        conn.rollback()
        print(f"🔥 DB error: {e}", flush=True)

    finally:

        cursor.close()
        conn.close()

    return result

# =====================================================
# STEP 2 — SMART NEW PLOT BACKFILL TRIGGER
# =====================================================

def get_plot_hash(plot_data):
    plantation = str(plot_data.get("plantation_date"))
    crop = str(plot_data.get("crop_type"))
    return f"{plantation}_{crop}"

def trigger_backfill_for_new_plots(plots):

    for plot_name, plot_data in plots.items():
        try:

            plot_row = run_query(
                "SELECT id, backfill_completed, last_backfill_hash FROM plots WHERE plot_name = %s",
                (plot_name,),
                fetchone=True
            )

            if not plot_row:
                continue

            plot_id = plot_row["id"]
            backfill_done = plot_row["backfill_completed"]
            old_hash = plot_row["last_backfill_hash"]

            new_hash = get_plot_hash(plot_data)

            if not backfill_done:
                print(f"🆕 New plot → backfill {plot_name}", flush=True)

                run_monthly_backfill_for_plot(plot_name, plot_data)

                run_query(
                    """
                    UPDATE plots
                    SET backfill_completed = TRUE,
                        last_backfill_hash = %s
                    WHERE id = %s
                    """,
                    (new_hash, plot_id)
                )

            elif old_hash != new_hash:
                print(f"🌱 Plot updated → rebackfill {plot_name}", flush=True)

                run_monthly_backfill_for_plot(plot_name, plot_data)

                run_query(
                    """
                    UPDATE plots
                    SET last_backfill_hash = %s
                    WHERE id = %s
                    """,
                    (new_hash, plot_id)
                )

            else:
                print(f"✅ Backfill already done {plot_name}", flush=True)

        except Exception as e:
            print(f"🔥 Backfill failed {plot_name}: {e}", flush=True)

trigger_backfill_for_new_plots(plots)

# =====================================================
# STEP 3 — CREATE ANALYSIS QUEUE
# =====================================================

print("📦 Creating analysis queue...", flush=True)

analysis_types = ["growth", "water_uptake", "soil_moisture", "pest_detection"]

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
            (plot_id, plot_name, analysis_type, status, priority)
            VALUES (%s,%s,%s,'pending',1)
            ON CONFLICT (plot_id,analysis_type)
            DO UPDATE SET status='pending'
            """,
            (plot_id, plot_name, analysis)
        )

print("✅ Queue created", flush=True)

# =====================================================
# DATE RANGE
# =====================================================

today = date.today().isoformat()
start_date = (date.today() - timedelta(days=30)).isoformat()
pest_start = (date.today() - timedelta(days=15)).isoformat()

# =====================================================
# QUEUE HELPERS
# =====================================================

def fetch_jobs():

    return run_query(
        """
        SELECT * FROM analysis_queue
        WHERE status='pending'
        ORDER BY priority DESC
        LIMIT %s
        """,
        (BATCH_SIZE,),
        fetch=True
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

def mark_failed(job_id, error="Unknown"):

    job = run_query(
        "SELECT retry_count FROM analysis_queue WHERE id=%s",
        (job_id,),
        fetchone=True
    )

    retry = job["retry_count"]

    if retry < MAX_RETRY:

        run_query(
            """
            UPDATE analysis_queue
            SET status='pending',
                retry_count=%s,
                last_error=%s
            WHERE id=%s
            """,
            (retry + 1, str(error), job_id)
        )

        print("🔁 Retrying job", flush=True)

    else:

        run_query(
            """
            UPDATE analysis_queue
            SET status='failed',
                last_error=%s
            WHERE id=%s
            """,
            (str(error), job_id)
        )

# =====================================================
# SCENE SKIP CHECK
# =====================================================

def is_scene_unchanged(plot_id, new_date):

    latest = run_query(
        """
        SELECT satellite_date
        FROM satellite_images
        WHERE plot_id=%s
        ORDER BY satellite_date DESC
        LIMIT 1
        """,
        (plot_id,),
        fetchone=True
    )

    if not latest:
        return False

    return str(latest["satellite_date"]) == str(new_date)

# =====================================================
# STORE RESULTS
# =====================================================

def store_results(results, analysis_type, plot_id):

    from psycopg2.extras import Json

    if not results:
        return

    if isinstance(results, dict):
        results = [results]

    for geojson in results:

        if not geojson.get("features"):
            continue

        props = geojson["features"][0]["properties"]

        analysis_date = props.get("analysis_image_date") or props.get("latest_image_date")
        sensor_used = props.get("sensor", "unknown")
        tile_url = props.get("tile_url")

        if not analysis_date:
            continue

        if is_scene_unchanged(plot_id, analysis_date):
            print("⏭ Scene unchanged — skipping")
            return

        run_query(
            """
            INSERT INTO satellite_images
            (plot_id, satellite, satellite_date)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (plot_id, sensor_used, analysis_date)
        )

        run_query(
            """
            INSERT INTO analysis_results
            (plot_id, analysis_type, analysis_date, sensor_used, tile_url, response_json)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                plot_id,
                analysis_type,
                analysis_date,
                sensor_used,
                tile_url,
                Json(geojson)   # ✅ FIX: convert dict → JSON
            )
        )

        print(f"✅ Stored {analysis_type} {analysis_date}", flush=True)

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

    print(f"⚙ Processing {plot_name} → {analysis_type}", flush=True)

    try:

        plot_data = plots.get(plot_name)

        if not plot_data:
            mark_failed(job_id, "Plot missing")
            return

        if analysis_type == "growth":
            results = run_growth_analysis_by_plot(plot_name, plot_data, start_date, today)

        elif analysis_type == "water_uptake":
            results = run_water_uptake_analysis_by_plot(plot_name, plot_data, start_date, today)

        elif analysis_type == "soil_moisture":
            results = run_soil_moisture_analysis_by_plot(plot_name, plot_data, start_date, today)

        elif analysis_type == "pest_detection":
            results = run_pest_detection_analysis_by_plot(plot_name, plot_data, pest_start, today)

        else:
            mark_failed(job_id, "Unknown analysis")
            return

        store_results(results, analysis_type, plot_id)

        mark_completed(job_id)

        time.sleep(REQUEST_DELAY)

    except Exception as e:

        if "Quota exceeded" in str(e):
            print("🧊 Cooling GEE 5 mins...")
            time.sleep(300)

        print("🔥 Job failed:", str(e), flush=True)

        mark_failed(job_id, str(e))

# =====================================================
# WORKER LOOP
# =====================================================

print("🚀 QUEUE WORKER STARTED", flush=True)

while True:

    jobs = fetch_jobs()

    if not jobs:
        print("✅ No pending jobs — worker exiting", flush=True)
        break

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_job, jobs)

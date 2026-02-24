from datetime import date, timedelta
import time

from gee_growth import (
    run_growth_analysis_by_plot,
    run_water_uptake_analysis_by_plot,
    run_soil_moisture_analysis_by_plot,
    run_pest_detection_analysis_by_plot
)

from shared_services import PlotSyncService, run_plot_sync
from db import supabase
from Admin import run_monthly_backfill_for_plot


# =====================================================
# CONFIG
# =====================================================

BATCH_SIZE = 20
SLEEP_TIME = 10


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
# STEP 2 — CREATE ANALYSIS QUEUE (CRON SCHEDULER)
# =====================================================

print("📦 Creating analysis queue...", flush=True)

analysis_types = [
    "growth",
    "water_uptake",
    "soil_moisture",
    "pest_detection"
]

for plot_name, plot_data in plots.items():

    plot_row = (
        supabase.table("plots")
        .select("id")
        .eq("plot_name", plot_name)
        .execute()
    )

    if not plot_row.data:
        continue

    plot_id = plot_row.data[0]["id"]

    for analysis in analysis_types:

        supabase.table("analysis_queue").upsert(
            {
                "plot_id": plot_id,
                "plot_name": plot_name,
                "analysis_type": analysis,
                "status": "pending"
            },
            on_conflict="plot_id,analysis_type"
        ).execute()

print("✅ Queue created", flush=True)


# =====================================================
# STEP 3 — MONTHLY BACKFILL CHECK
# =====================================================

print("🔁 Running monthly backfill...", flush=True)

for plot_name, plot_data in plots.items():
    try:
        run_monthly_backfill_for_plot(plot_name, plot_data)
    except Exception as e:
        print("🔥 Backfill failed:", plot_name, str(e), flush=True)


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
    jobs = (
        supabase.table("analysis_queue")
        .select("*")
        .eq("status", "pending")
        .limit(BATCH_SIZE)
        .execute()
    )
    return jobs.data or []


def mark_processing(job_id):
    supabase.table("analysis_queue").update(
        {
            "status": "processing"
        }
    ).eq("id", job_id).execute()


def mark_completed(job_id):
    supabase.table("analysis_queue").update(
        {
            "status": "completed"
        }
    ).eq("id", job_id).execute()


def mark_failed(job_id):
    supabase.table("analysis_queue").update(
        {
            "status": "failed"
        }
    ).eq("id", job_id).execute()


# =====================================================
# SAFE STORE FUNCTION
# =====================================================

def store_results(results, analysis_type, plot_id):

    if not results:
        return

    if isinstance(results, dict):
        results = [results]

    for geojson in results:

        if not isinstance(geojson, dict):
            continue

        if not geojson.get("features"):
            continue

        properties = geojson["features"][0]["properties"]

        analysis_date = (
            properties.get("latest_image_date")
            or properties.get("analysis_dates", {}).get("latest_image_date")
            or properties.get("analysis_dates", {}).get("analysis_end_date")
        )

        sensor_used = (
            properties.get("data_source")
            or properties.get("sensor")
            or properties.get("sensor_used")
            or "unknown"
        )

        tile_url = properties.get("tile_url")

        if not analysis_date:
            continue

        # ---------- Satellite ----------

        supabase.table("satellite_images").upsert(
            {
                "plot_id": plot_id,
                "satellite": sensor_used,
                "satellite_date": analysis_date,
            },
            on_conflict="plot_id,satellite,satellite_date"
        ).execute()

        # ---------- Analysis ----------

        supabase.table("analysis_results").upsert(
            {
                "plot_id": plot_id,
                "analysis_type": analysis_type,
                "analysis_date": analysis_date,
                "sensor_used": sensor_used,
                "tile_url": tile_url,
                "response_json": geojson,
            },
            on_conflict="plot_id,analysis_type,analysis_date,sensor_used"
        ).execute()

        print(
            f"✅ Stored {analysis_type} ({sensor_used}) {analysis_date}",
            flush=True
        )


# =====================================================
# STEP 4 — QUEUE WORKER LOOP
# =====================================================

print("🚀 QUEUE WORKER STARTED", flush=True)

while True:

    jobs = fetch_jobs()

    if not jobs:
        print("😴 No pending jobs. Sleeping...", flush=True)
        time.sleep(SLEEP_TIME)
        continue

    for job in jobs:

        job_id = job["id"]
        plot_name = job["plot_name"]
        analysis_type = job["analysis_type"]
        plot_id = job["plot_id"]

        print(f"\n⚙ Processing {plot_name} → {analysis_type}", flush=True)

        mark_processing(job_id)

        try:

            plot_data = plots.get(plot_name)

            if not plot_data:
                mark_failed(job_id)
                continue

            if analysis_type == "growth":
                results = run_growth_analysis_by_plot(
                    plot_name, plot_data,
                    start_date, today
                )

            elif analysis_type == "water_uptake":
                results = run_water_uptake_analysis_by_plot(
                    plot_name, plot_data,
                    start_date, today
                )

            elif analysis_type == "soil_moisture":
                results = run_soil_moisture_analysis_by_plot(
                    plot_name, plot_data,
                    start_date, today
                )

            elif analysis_type == "pest_detection":
                results = run_pest_detection_analysis_by_plot(
                    plot_name, plot_data,
                    pest_start, today
                )

            else:
                mark_failed(job_id)
                continue

            store_results(results, analysis_type, plot_id)

            mark_completed(job_id)

        except Exception as e:
            print("🔥 Job failed:", str(e), flush=True)
            mark_failed(job_id)

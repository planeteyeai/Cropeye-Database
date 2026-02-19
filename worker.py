from datetime import date, timedelta
from gee_growth import run_growth_analysis_by_plot
from shared_services import PlotSyncService, run_plot_sync
from db import supabase
from Admin import run_monthly_backfill_for_plot


# =====================================================
# STEP 1 — RUN INTERNAL PLOT SYNC
# =====================================================

print("🔄 Running internal plot sync before growth...", flush=True)

try:
    sync_result = run_plot_sync()
    print("✅ Sync result:", sync_result, flush=True)
except Exception as e:
    print("❌ Plot sync failed:", str(e), flush=True)


# =====================================================
# STEP 2 — RUN DAILY GROWTH WORKER
# =====================================================

print("🚀 DAILY GROWTH WORKER STARTED", flush=True)

today = date.today().isoformat()
start_date = (date.today() - timedelta(days=30)).isoformat()
end_date = today

plots = PlotSyncService().get_plots_dict(force_refresh=True)

print("🔁 Running monthly historical backfill check...", flush=True)

for plot_name, plot_data in plots.items():
    try:
        run_monthly_backfill_for_plot(plot_name, plot_data)
    except Exception as e:
        print("🔥 Backfill failed for", plot_name, str(e), flush=True)


for plot_name, plot_data in plots.items():
    print(f"\n--- Processing {plot_name} ---", flush=True)

    try:
        props = plot_data.get("properties") or {}
        django_id = props.get("django_id")

        if not django_id:
            print("❌ Missing django_id", flush=True)
            continue

        # ---------------- DB CHECK ----------------

        plot_row = (
            supabase.table("plots")
            .select("id")
            .eq("django_plot_id", django_id)
            .execute()
        )

        if not plot_row.data:
            print("❌ Plot not found in DB", flush=True)
            continue

        plot_id = plot_row.data[0]["id"]
        print("✔ Plot ID:", plot_id, flush=True)

        # ---------------- GEE ANALYSIS ----------------

        results = run_growth_analysis_by_plot(
            plot_name,
            plot_data=plot_data,
            start_date=start_date,
            end_date=end_date
        )

        print("✔ Growth analysis done", flush=True)

        for geojson in results:

            properties = geojson["features"][0]["properties"]

            analysis_date = properties["latest_image_date"]
            sensor_used = properties["data_source"]
            tile_url = properties["tile_url"]

            # ---------------- STORE SATELLITE METADATA ----------------

            supabase.table("satellite_images").upsert(
                {
                    "plot_id": plot_id,
                    "satellite": sensor_used,
                    "satellite_date": analysis_date,
                },
                on_conflict="plot_id,satellite,satellite_date"
            ).execute()

            print(
                f"   🛰 Satellite stored {sensor_used} ({analysis_date})",
                flush=True
            )

            # ---------------- STORE RESULTS ----------------

            supabase.table("analysis_results").upsert(
                {
                    "plot_id": plot_id,
                    "analysis_type": "growth",
                    "analysis_date": analysis_date,
                    "sensor_used": sensor_used,
                    "tile_url": tile_url,
                    "response_json": geojson,
                },
                on_conflict="plot_id,analysis_type,analysis_date,sensor_used"
            ).execute()

            print(
                f"   ✅ Stored {sensor_used} for {analysis_date}",
                flush=True
            )

    except Exception as e:
        print("🔥 ERROR:", str(e), flush=True)

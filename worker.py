from datetime import date, timedelta
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
# STEP 1 — RUN INTERNAL PLOT SYNC
# =====================================================

print("🔄 Running internal plot sync before analysis...", flush=True)

try:
    sync_result = run_plot_sync()
    print("✅ Sync result:", sync_result, flush=True)
except Exception as e:
    print("❌ Plot sync failed:", str(e), flush=True)


# =====================================================
# STEP 2 — DATE RANGE
# =====================================================

print("🚀 DAILY ANALYSIS WORKER STARTED", flush=True)

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


# =====================================================
# STEP 3 — MAIN LOOP
# =====================================================

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
            .eq("plot_name", plot_name)  # ⚠ Change if column name differs
            .execute()
        )

        if not plot_row.data:
            print("❌ Plot not found in DB", flush=True)
            continue

        plot_id = plot_row.data[0]["id"]
        print("✔ Plot ID:", plot_id, flush=True)

        # =====================================================
        # SAFE STORE FUNCTION
        # =====================================================

        def store_results(results, analysis_type):

            if not results:
                return

            # 🔥 Normalize dict → list
            if isinstance(results, dict):
                results = [results]

            if not isinstance(results, list):
                print(f"⚠ Unexpected format for {analysis_type}", flush=True)
                return

            for geojson in results:

                if not isinstance(geojson, dict):
                    continue

                if not geojson.get("features"):
                    continue

                properties = geojson["features"][0]["properties"]

                # ---------------- DATE EXTRACTION ----------------

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
                    print(f"⚠ Skipping {analysis_type} — no date", flush=True)
                    continue

                # ---------------- SATELLITE TABLE ----------------

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

                # ---------------- ANALYSIS TABLE ----------------

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
                    f"   ✅ Stored {analysis_type} ({sensor_used}) for {analysis_date}",
                    flush=True
                )

        # =====================================================
        # RUN ANALYSES
        # =====================================================

        growth_results = run_growth_analysis_by_plot(
            plot_name,
            plot_data=plot_data,
            start_date=start_date,
            end_date=end_date
        )
        print("✔ Growth analysis done", flush=True)
        store_results(growth_results, "growth")

        water_results = run_water_uptake_analysis_by_plot(
            plot_name,
            plot_data=plot_data,
            start_date=start_date,
            end_date=end_date
        )
        print("✔ Water uptake analysis done", flush=True)
        store_results(water_results, "water_uptake")

        soil_results = run_soil_moisture_analysis_by_plot(
            plot_name,
            plot_data=plot_data,
            start_date=start_date,
            end_date=end_date
        )
        print("✔ Soil moisture analysis done", flush=True)
        store_results(soil_results, "soil_moisture")

        pest_start = (date.today() - timedelta(days=15)).isoformat()

        pest_results = run_pest_detection_analysis_by_plot(
            plot_name,
            plot_data=plot_data,
            start_date=pest_start,
            end_date=end_date
        )
        print("✔ Pest detection analysis done", flush=True)
        store_results(pest_results, "pest_detection")

    except Exception as e:
        print("🔥 ERROR:", str(e), flush=True)

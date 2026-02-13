from datetime import date, timedelta
from gee_growth import run_growth_analysis_by_plot
from shared_services import PlotSyncService
from db import supabase

print("🚀 DAILY GROWTH WORKER STARTED")

today = date.today().isoformat()
start_date = (date.today() - timedelta(days=30)).isoformat()

plots = PlotSyncService().get_plots_dict(force_refresh=True)

for plot_name, plot_data in plots.items():
    print(f"\n--- Processing {plot_name} ---", flush=True)

    try:
        props = plot_data.get("properties") or {}
        django_id = props.get("django_id")

        if not django_id:
            print("❌ Missing django_id", flush=True)
            continue

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

        sat_row = (
            supabase.table("satellite_images")
            .select("id,satellite,satellite_date")
            .eq("plot_id", plot_id)
            .order("satellite_date", desc=True)
            .limit(1)
            .execute()
        )

        if not sat_row.data:
            print("❌ No satellite image found", flush=True)
            continue

        satellite_image = sat_row.data[0]
        print("✔ Satellite found:", satellite_image["id"], flush=True)

        result = run_growth_analysis_by_plot(
            plot_data=plot_data,
            start_date=start_date,
            end_date=end_date
        )

        print("✔ Growth analysis done", flush=True)

        response = supabase.table("analysis_results").upsert(
            {
                "plot_id": plot_id,
                "satellite_image_id": satellite_image["id"],
                "analysis_type": "growth",
                "analysis_date": result["analysis_date"],
                "sensor_used": result["sensor"],
                "tile_url": result["tile_url"],
                "response_json": result["response_json"],
            },
            on_conflict="plot_id,satellite_image_id,analysis_type"
        ).execute()

        print("✔ Insert response:", response.data, flush=True)

    except Exception as e:
        print("🔥 ERROR:", str(e), flush=True)


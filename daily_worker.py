import os
import ee
import requests
from supabase import create_client
from gee_growth import run_growth_analysis_by_plot
from datetime import date

# ---------------- ENV ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
FASTAPI_URL = os.getenv("FASTAPI_PLOTS_URL")
WORKER_TOKEN = os.getenv("WORKER_TOKEN")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("🚀 Initializing GEE...")
service_account_info = json.loads(os.environ["EE_SERVICE_ACCOUNT_JSON"])

credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"],
    key_data=json.dumps(service_account_info)
)

ee.Initialize(credentials, project=service_account_info["project_id"])


# ---------------- MAIN ----------------
def run():
    print("🛰 Fetching plots from API...")
    res = requests.get(
        FASTAPI_URL,
        headers={"x-worker-token": WORKER_TOKEN}
    )

    plots = res.json()["plots"]
    print(f"📍 Found {len(plots)} plots")

    for plot in plots:
        plot_name = plot["plot_name"]
        print(f"\n🌱 Processing {plot_name}")

        # ---------------- Get plot in DB ----------------
        db_plot = supabase.table("plots") \
            .select("id") \
            .eq("plot_name", plot_name) \
            .execute()

        if not db_plot.data:
            print("❌ Plot not found in Supabase:", plot_name)
            continue

        plot_id = db_plot.data[0]["id"]

        # ---------------- GEE Compute ----------------
        try:
            result = run_growth_analysis_by_plot(
                plot_data=plot,
                start_date="2025-01-01",
                end_date=str(date.today())
            )
        except Exception as e:
            print("❌ GEE failed:", e)
            continue

        analysis_date = result["analysis_date"]

        # ---------------- Skip if exists ----------------
        cached = supabase.table("analysis_results") \
            .select("id") \
            .eq("plot_id", plot_id) \
            .eq("analysis_type", "growth") \
            .eq("analysis_date", analysis_date) \
            .execute()

        if cached.data:
            print("⏭ Already cached for", analysis_date)
            continue

        # ---------------- Satellite image row ----------------
        sat = supabase.table("satellite_images") \
            .insert({
                "plot_id": plot_id,
                "satellite": result["sensor"],
                "satellite_date": analysis_date
            }) \
            .execute()

        sat_id = sat.data[0]["id"]

        # ---------------- Store analysis ----------------
        supabase.table("analysis_results") \
            .insert({
                "plot_id": plot_id,
                "satellite_image_id": sat_id,
                "analysis_type": "growth",
                "analysis_date": analysis_date,
                "sensor_used": result["sensor"],
                "tile_url": result["tile_url"],
                "response_json": result["response_json"]
            }) \
            .execute()

        print("✅ Stored growth for", plot_name, analysis_date)

# ---------------- RUN ----------------
if __name__ == "__main__":
    run()

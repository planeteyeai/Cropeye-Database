import os
import json
import ee
import requests
from datetime import date
from supabase import create_client
from gee_growth import run_growth_analysis_by_plot

# ======================================================
# ENV VARIABLES
# ======================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
FASTAPI_URL = os.getenv("FASTAPI_PLOTS_URL")   # must return LIST of plot names
WORKER_TOKEN = os.getenv("WORKER_TOKEN")
EE_JSON = os.getenv("EE_SERVICE_ACCOUNT_JSON")

if not all([
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    FASTAPI_URL,
    WORKER_TOKEN,
    EE_JSON
]):
    raise RuntimeError("❌ Missing one or more required environment variables")

# ======================================================
# SUPABASE CLIENT
# ======================================================
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ======================================================
# INITIALIZE GOOGLE EARTH ENGINE (SERVICE ACCOUNT)
# ======================================================
print("🚀 Initializing Google Earth Engine...")

service_account_info = json.loads(EE_JSON)

credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"],
    key_data=json.dumps(service_account_info)
)

ee.Initialize(credentials, project=service_account_info["project_id"])

print("✅ GEE initialized successfully")

# ======================================================
# MAIN WORKER
# ======================================================
def run():
    print("🛰 Fetching plots from API...")

    res = requests.get(
        FASTAPI_URL,
        headers={"x-worker-token": WORKER_TOKEN},
        timeout=30
    )

    if res.status_code != 200:
        raise RuntimeError(f"❌ Failed to fetch plots: {res.status_code}")

    plots = res.json()

    if not isinstance(plots, list):
        raise RuntimeError("❌ API response is not a list")

    print(f"📍 Found {len(plots)} plots")

    for plot_name in plots:
        print(f"\n🌱 Processing plot: {plot_name}")

        # --------------------------------------------------
        # FETCH PLOT FROM SUPABASE (WITH GEOMETRY)
        # --------------------------------------------------
        db_plot = (
            supabase.table("plots")
            .select("id, geometry")
            .eq("plot_name", plot_name)
            .execute()
        )

        if not db_plot.data:
            print("❌ Plot not found in Supabase:", plot_name)
            continue

        plot_record = db_plot.data[0]
        plot_id = plot_record["id"]

        # --------------------------------------------------
        # RUN GEE ANALYSIS
        # --------------------------------------------------
        try:
            result = run_growth_analysis_by_plot(
                plot_data=plot_record,
                start_date="2025-01-01",
                end_date=str(date.today())
            )
        except Exception as e:
            print("❌ GEE failed:", e)
            continue

        analysis_date = result["analysis_date"]

        # --------------------------------------------------
        # SKIP IF ALREADY STORED
        # --------------------------------------------------
        cached = (
            supabase.table("analysis_results")
            .select("id")
            .eq("plot_id", plot_id)
            .eq("analysis_type", "growth")
            .eq("analysis_date", analysis_date)
            .execute()
        )

        if cached.data:
            print("⏭ Already cached for", analysis_date)
            continue

        # --------------------------------------------------
        # INSERT SATELLITE IMAGE
        # --------------------------------------------------
        sat = (
            supabase.table("satellite_images")
            .insert({
                "plot_id": plot_id,
                "satellite": result["sensor"],
                "satellite_date": analysis_date
            })
            .execute()
        )

        sat_id = sat.data[0]["id"]

        # --------------------------------------------------
        # INSERT ANALYSIS RESULT
        # --------------------------------------------------
        supabase.table("analysis_results").insert({
            "plot_id": plot_id,
            "satellite_image_id": sat_id,
            "analysis_type": "growth",
            "analysis_date": analysis_date,
            "sensor_used": result["sensor"],
            "tile_url": result["tile_url"],
            "response_json": result["response_json"]
        }).execute()

        print("✅ Stored growth analysis for", plot_name, analysis_date)

# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    run()

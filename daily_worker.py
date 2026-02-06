import os
import json
import ee
import requests
from datetime import date
from supabase import create_client
from gee_growth import run_growth_analysis_by_plot

# ================== CONFIG ==================
GEOMETRY_COLUMN = os.getenv("PLOT_GEOMETRY_COLUMN", "geom")  
# 👆 change to geom / polygon / geojson if needed

# ================== ENV VALIDATION ==================
REQUIRED_ENV_VARS = [
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "FASTAPI_PLOTS_URL",
    "WORKER_TOKEN",
    "EE_SERVICE_ACCOUNT_JSON",
]

missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing:
    raise RuntimeError(f"❌ Missing environment variables: {missing}")

# ================== ENV ==================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
FASTAPI_URL = os.getenv("FASTAPI_PLOTS_URL")
WORKER_TOKEN = os.getenv("WORKER_TOKEN")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ================== GEE INIT ==================
print("🚀 Initializing Google Earth Engine...")

service_account_info = json.loads(os.environ["EE_SERVICE_ACCOUNT_JSON"])

credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"],
    key_data=json.dumps(service_account_info),
)

ee.Initialize(credentials, project=service_account_info["project_id"])
print("✅ GEE initialized successfully")

# ================== MAIN ==================
def run():
    print("🛰 Fetching plots from API...")

    res = requests.get(
        FASTAPI_URL,
        headers={"x-worker-token": WORKER_TOKEN},
        timeout=30,
    )

    if res.status_code != 200:
        raise RuntimeError(res.text)

    plots = res.json()
    if not isinstance(plots, list):
        raise RuntimeError("❌ API response must be List[str]")

    print(f"📍 Found {len(plots)} plots")

    for plot_name in plots:
        print(f"\n🌱 Processing plot: {plot_name}")

        # ---------- Fetch plot ----------
        db_plot = (
            supabase
            .table("plots")
            .select(f"id,{GEOMETRY_COLUMN}")
            .eq("plot_name", plot_name)
            .execute()
        )

        if not db_plot.data:
            print("❌ Plot not found:", plot_name)
            continue

        plot = db_plot.data[0]
        plot_id = plot["id"]
        geometry = plot.get(GEOMETRY_COLUMN)

        if not geometry:
            print(f"❌ Geometry missing ({GEOMETRY_COLUMN}) for:", plot_name)
            continue

        # ---------- Run GEE ----------
        try:
            result = run_growth_analysis_by_plot(
                plot_data={
                    "plot_name": plot_name,
                    "geometry": geometry,
                },
                start_date="2025-01-01",
                end_date=str(date.today()),
            )
        except Exception as e:
            print("❌ GEE failed:", e)
            continue

        analysis_date = result["analysis_date"]

        # ---------- Cache check ----------
        cached = (
            supabase
            .table("analysis_results")
            .select("id")
            .eq("plot_id", plot_id)
            .eq("analysis_type", "growth")
            .eq("analysis_date", analysis_date)
            .execute()
        )

        if cached.data:
            print("⏭ Already exists:", analysis_date)
            continue

        # ---------- Satellite image ----------
        sat = supabase.table("satellite_images").insert({
            "plot_id": plot_id,
            "satellite": result["sensor"],
            "satellite_date": analysis_date,
        }).execute()

        sat_id = sat.data[0]["id"]

        # ---------- Store result ----------
        supabase.table("analysis_results").insert({
            "plot_id": plot_id,
            "satellite_image_id": sat_id,
            "analysis_type": "growth",
            "analysis_date": analysis_date,
            "sensor_used": result["sensor"],
            "tile_url": result["tile_url"],
            "response_json": result["response_json"],
        }).execute()

        print("✅ Stored growth:", plot_name, analysis_date)

# ================== RUN ==================
if __name__ == "__main__":
    run()

import os
import json
import datetime
import ee
from supabase import create_client

# =========================
# ENV VALIDATION
# =========================
REQUIRED_ENV = [
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "EE_SERVICE_ACCOUNT_JSON",
]

for k in REQUIRED_ENV:
    if not os.getenv(k):
        raise RuntimeError(f"❌ Missing env var: {k}")

TODAY = datetime.date.today().isoformat()

# =========================
# SUPABASE
# =========================
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)

# =========================
# GEE INIT
# =========================
print("🚀 Initializing Google Earth Engine...")
creds = json.loads(os.environ["EE_SERVICE_ACCOUNT_JSON"])
ee.Initialize(
    ee.ServiceAccountCredentials(
        creds["client_email"],
        key_data=json.dumps(creds)
    )
)
print("✅ GEE initialized successfully")

# =========================
# MAIN WORKER
# =========================
def run():
    print("🛰 Fetching plots from Supabase...")

    plots = (
        supabase
        .table("plots")
        .select("id, plot_name, geojson")
        .execute()
        .data
    )

    print(f"📍 Found {len(plots)} plots")

    for plot in plots:
        plot_id = plot["id"]
        plot_name = plot["plot_name"]
        geojson = plot["geojson"]

        print(f"\n🌱 Processing plot: {plot_name}")

        if not geojson:
            print("⚠️ Missing geojson, skipping")
            continue

        # 🚫 Skip if already processed today
        existing = (
            supabase
            .table("satellite_images")
            .select("id")
            .eq("plot_id", plot_id)
            .eq("satellite_date", TODAY)
            .execute()
            .data
        )

        if existing:
            print("⏩ Already processed today, skipping")
            continue

        try:
            ee_geom = ee.Geometry(geojson)

            # 📐 Area
            area_ha = ee_geom.area(maxError=1).getInfo() / 10_000
            print(f"📐 Area: {area_ha:.2f} ha")

            # 🛰 Sentinel-2
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR")
                .filterBounds(ee_geom)
                .filterDate("2024-01-01", TODAY)
                .sort("CLOUDY_PIXEL_PERCENTAGE")
            )

            img = collection.first()
            if img is None:
                print("⚠️ No satellite image found")
                continue

            # 🌿 NDVI
            ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
            mean_ndvi = (
                ndvi
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=ee_geom,
                    scale=10,
                    maxPixels=1e9
                )
                .get("NDVI")
                .getInfo()
            )

            print(f"🌿 NDVI: {mean_ndvi}")

            # 📡 Save satellite image log
            supabase.table("satellite_images").insert({
                "plot_id": plot_id,
                "satellite": "sentinel-2",
                "satellite_date": TODAY
            }).execute()

            # 📊 Save analysis
            supabase.table("analysis_results").insert({
                "plot_id": plot_id,
                "analysis_type": "growth",
                "sensor_used": "Sentinel-2",
                "analysis_date": TODAY,
                "response_json": {
                    "ndvi": mean_ndvi,
                    "area_hectares": area_ha
                }
            }).execute()

            print("✅ Stored successfully")

        except Exception as e:
            print(f"❌ Skipped due to error: {e}")

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    run()

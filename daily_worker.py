import os
import json
import time
import requests
import ee
from supabase import create_client

# =========================
# ENV VALIDATION
# =========================
REQUIRED_ENV = [
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "EE_SERVICE_ACCOUNT_JSON",
    "FASTAPI_PLOTS_URL",
    "WORKER_TOKEN"
]

missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
if missing:
    print("❌ Missing environment variables:")
    for k in missing:
        print(f"   - {k}")
    raise RuntimeError("❌ Environment validation failed")

# =========================
# INIT SUPABASE
# =========================
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)

# =========================
# INIT GEE
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
# FETCH + NORMALIZE PLOTS
# =========================
def fetch_plots():
    url = os.environ["FASTAPI_PLOTS_URL"]
    token = os.environ["WORKER_TOKEN"]

    for attempt in range(3):
        try:
            res = requests.get(
                url,
                headers={"x-worker-token": token},
                timeout=20
            )
            res.raise_for_status()
            data = res.json()

            # Case 1: { "plots": [...] }
            if isinstance(data, dict) and "plots" in data:
                data = data["plots"]

            normalized = []

            for item in data:
                # Case 2: "999_11"
                if isinstance(item, str):
                    normalized.append({
                        "plot_name": item,
                        "geometry": None
                    })

                # Case 3: full object
                elif isinstance(item, dict):
                    normalized.append(item)

            return normalized

        except requests.exceptions.ReadTimeout:
            print(f"⏳ Timeout fetching plots (attempt {attempt+1}/3)")
            time.sleep(2)

        except Exception as e:
            print("❌ Failed fetching plots:", e)
            break

    return []

# =========================
# MAIN WORKER
# =========================
def run():
    print("🛰 Fetching plots from API...")
    plots = fetch_plots()

    print(f"📍 Found {len(plots)} plots")

    for plot in plots:
        plot_name = plot.get("plot_name")
        geometry = plot.get("geometry")

        print(f"\n🌱 Processing plot: {plot_name}")

        if not geometry:
            print("⚠️ No geometry provided by API, skipping GEE")
            continue

        try:
            ee_geom = ee.Geometry(geometry)

            area_m2 = ee_geom.area(maxError=1).getInfo()
            area_ha = area_m2 / 10_000
            print(f"📐 Area: {area_ha:.2f} ha")

            img = (
                ee.ImageCollection("COPERNICUS/S2_SR")
                .filterBounds(ee_geom)
                .filterDate("2024-01-01", "2024-12-31")
                .sort("CLOUDY_PIXEL_PERCENTAGE")
                .first()
            )

            if img is None:
                print("⚠️ No imagery found")
                continue

            ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")

            mean_ndvi = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_geom,
                scale=10,
                maxPixels=1e9
            ).get("NDVI").getInfo()

            print(f"🌿 NDVI: {mean_ndvi}")

            supabase.table("plot_metrics").insert({
                "plot_name": plot_name,
                "area_ha": area_ha,
                "ndvi": mean_ndvi
            }).execute()

        except ee.EEException as e:
            print(f"❌ GEE error: {e}")

        except Exception as e:
            print(f"❌ Unexpected error: {e}")

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    run()

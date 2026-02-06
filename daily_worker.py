import os
import json
import ee
from supabase import create_client
from postgrest.exceptions import APIError

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
        raise RuntimeError(f"❌ Missing env: {k}")

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
# FIND GEOMETRY COLUMN
# =========================
def detect_geometry_column():
    candidates = [
        "geometry",
        "geom",
        "geojson",
        "geometry_json",
        "boundary",
        "polygon",
    ]

    for col in candidates:
        try:
            supabase.table("plots").select(col).limit(1).execute()
            print(f"✅ Using geometry column: {col}")
            return col
        except APIError:
            continue

    raise RuntimeError("❌ No geometry column found in plots table")

# =========================
# MAIN WORKER
# =========================
def run():
    print("🛰 Fetching plots directly from Supabase...")

    geom_col = detect_geometry_column()

    plots = (
        supabase
        .table("plots")
        .select(f"id, plot_name, {geom_col}")
        .execute()
        .data
    )

    print(f"📍 Found {len(plots)} plots")

    for plot in plots:
        plot_id = plot["id"]
        plot_name = plot.get("plot_name", "UNKNOWN")
        geometry = plot.get(geom_col)

        print(f"\n🌱 Processing plot: {plot_name}")

        if not geometry:
            print("⚠️ Missing geometry, skipping")
            continue

        try:
            ee_geom = ee.Geometry(geometry)

            area_ha = (
                ee_geom
                .area(maxError=1)
                .divide(10_000)
                .getInfo()
            )

            print(f"📐 Area: {area_ha:.2f} ha")

            img = (
                ee.ImageCollection("COPERNICUS/S2_SR")
                .filterBounds(ee_geom)
                .filterDate("2024-01-01", "2024-12-31")
                .sort("CLOUDY_PIXEL_PERCENTAGE")
                .first()
            )

            if img is None:
                print("⚠️ No imagery, skipping")
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
                "plot_id": plot_id,
                "plot_name": plot_name,
                "area_ha": area_ha,
                "ndvi": mean_ndvi
            }).execute()

            print("✅ Stored")

        except ee.EEException as e:
            print(f"❌ GEE error (skipped): {e}")

        except Exception as e:
            print(f"❌ Unexpected error (skipped): {e}")

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    run()

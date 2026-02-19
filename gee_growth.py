import ee
from datetime import datetime
import json
import os

# ======================================================
# Earth Engine Initialization (SAFE)
# ======================================================

if "EE_SERVICE_ACCOUNT_JSON" not in os.environ:
    raise RuntimeError("EE_SERVICE_ACCOUNT_JSON env variable not set")

service_account_info = json.loads(os.environ["EE_SERVICE_ACCOUNT_JSON"])

credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"],
    key_data=json.dumps(service_account_info)
)

ee.Initialize(credentials, project=service_account_info["project_id"])

# ======================================================
# Growth Analysis
# ======================================================

def run_growth_analysis_by_plot(plot_name, plot_data, start_date, end_date):

    try:
        geometry = plot_data.get("geometry")
        properties = plot_data.get("properties") or {}
        area_acres = properties.get("area_acres")

        if not geometry:
            print("⚠ Missing geometry", flush=True)
            return None

        polygon = ee.Geometry(geometry)

        results = []

        # =====================================================
        # SENTINEL-2 (NDVI)
        # =====================================================

        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(polygon)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            .map(lambda img: img.addBands(
                img.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ))
            .sort("system:time_start", False)
        )

        if s2_collection.size().getInfo() > 0:

            image = s2_collection.first()

            latest_date = ee.Date(
                image.get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()

            ndvi = image.select("NDVI")

            tile_url = ndvi.getMapId(
                {"min": 0, "max": 1, "palette": ["red", "yellow", "green"]}
            )["tile_fetcher"].url_format

            geojson_s2 = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": geometry.info(),
                    "properties": {
                        "plot_name": plot_name,
                        "area_acres": area_acres,
                        "start_date": start_date,
                        "end_date": end_date,
                        "tile_url": tile_url,
                        "data_source": "Sentinel-2 NDVI",
                        "latest_image_date": latest_date,
                        "last_updated": datetime.utcnow().isoformat()
                    }
                }]
            }

            results.append(geojson_s2)

        else:
            print("⚠ No Sentinel-2 data", flush=True)

        # =====================================================
        # SENTINEL-1 (VH Backscatter)
        # =====================================================

        s1_collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(polygon)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
            .select("VH")
            .sort("system:time_start", False)
        )

        if s1_collection.size().getInfo() > 0:

            image = s1_collection.first()

            latest_date = ee.Date(
                image.get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()

            vh = image.select("VH")

            tile_url = vh.getMapId(
                {"min": -25, "max": 0}
            )["tile_fetcher"].url_format

            geojson_s1 = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": geometry.info(),
                    "properties": {
                        "plot_name": plot_name,
                        "area_acres": area_acres,
                        "start_date": start_date,
                        "end_date": end_date,
                        "tile_url": tile_url,
                        "data_source": "Sentinel-1 VH",
                        "latest_image_date": latest_date,
                        "last_updated": datetime.utcnow().isoformat()
                    }
                }]
            }

            results.append(geojson_s1)

        else:
            print("⚠ No Sentinel-1 data", flush=True)

        if not results:
            print("⚠ No results returned", flush=True)
            return None

        return results

    except Exception as e:
        print(f"❌ Growth analysis failed: {e}", flush=True)
        return None

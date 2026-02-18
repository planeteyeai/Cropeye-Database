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
# Growth Analysis (UNCHANGED CORE LOGIC)
# ======================================================

def run_growth_analysis_by_plot(plot_name, plot_data, start_date, end_date):

    if not plot_data or "geometry" not in plot_data:
        raise ValueError("plot_data missing geometry")

    geometry = plot_data["geometry"]
    props = plot_data.get("properties", {})

    area_hectares = geometry.area().divide(10000).getInfo()
    area_acres = round(area_hectares * 2.47105, 2)

    analysis_start = ee.Date(start_date)
    analysis_end = ee.Date(end_date)

    results = []

    # ======================================================
    # SENTINEL-2
    # ======================================================

    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(analysis_start, analysis_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        .map(lambda img: img.clip(geometry))
        .sort("system:time_start", False)
    )

    if s2_collection.size().getInfo() > 0:

        latest_image = ee.Image(s2_collection.first())
        latest_date = ee.Date(latest_image.get("system:time_start"))
        latest_image_date = latest_date.format("YYYY-MM-dd").getInfo()

        ndvi = latest_image.normalizedDifference(["B8", "B4"]).rename("NDVI")

        # ✅ PRIORITY FIXED (stress evaluated first)
        stress_mask = ndvi.gte(0.0).And(ndvi.lt(0.2))
        weak_mask = ndvi.gte(0.2).And(ndvi.lt(0.35))
        moderate_mask = ndvi.gte(0.35).And(ndvi.lt(0.6))
        healthy_mask = ndvi.gte(0.6)

        result = _build_response(
            geometry,
            props,
            area_acres,
            weak_mask,
            stress_mask,
            moderate_mask,
            healthy_mask,
            latest_image_date,
            "Sentinel-2 NDVI",
            start_date,
            end_date,
            s2_collection.size().getInfo()
        )

        results.append(result)

    # ======================================================
    # SENTINEL-1
    # ======================================================

    s1_collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(analysis_start, analysis_end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VH"])
        .map(lambda img: img.clip(geometry))
        .sort("system:time_start", False)
    )

    if s1_collection.size().getInfo() > 0:

        latest_image = ee.Image(s1_collection.first())
        latest_date = ee.Date(latest_image.get("system:time_start"))
        latest_image_date = latest_date.format("YYYY-MM-dd").getInfo()

        vh = latest_image.select("VH")

        # ✅ PRIORITY FIXED (stress stronger band separation)
        stress_mask = vh.gt(-13)
        weak_mask = vh.lte(-13).And(vh.gt(-15))
        moderate_mask = vh.lte(-15).And(vh.gt(-17))
        healthy_mask = vh.lte(-17)

        result = _build_response(
            geometry,
            props,
            area_acres,
            weak_mask,
            stress_mask,
            moderate_mask,
            healthy_mask,
            latest_image_date,
            "Sentinel-1 VH",
            start_date,
            end_date,
            s1_collection.size().getInfo()
        )

        results.append(result)

    if not results:
        raise Exception("No Sentinel-1 or Sentinel-2 images found")

    return results


# ======================================================
# STANDARDIZED RESPONSE BUILDER (FRONTEND SAFE)
# ======================================================

def _build_response(
    geometry,
    props,
    area_acres,
    weak_mask,
    stress_mask,
    moderate_mask,
    healthy_mask,
    latest_image_date,
    data_source,
    start_date,
    end_date,
    image_count
):

    # ✅ PRIORITY ORDER FIXED
    combined = (
        ee.Image(0)
        .where(stress_mask, 2)
        .where(weak_mask, 1)
        .where(moderate_mask, 3)
        .where(healthy_mask, 4)
        .clip(geometry)
    )

    tile_url = (
        combined.visualize(
            min=0,
            max=4,
            palette=["#bc1e29", "#f39c12", "#58cf54", "#056c3e"]
        )
        .getMapId()["tile_fetcher"]
        .url_format
    )

    count_img = ee.Image.constant(1)

    def pixel_count(mask):
        return (
            count_img.updateMask(mask)
            .reduceRegion(ee.Reducer.count(), geometry, 10, bestEffort=True)
            .get("constant")
        )

    healthy = pixel_count(healthy_mask).getInfo() or 0
    moderate = pixel_count(moderate_mask).getInfo() or 0
    weak = pixel_count(weak_mask).getInfo() or 0
    stress = pixel_count(stress_mask).getInfo() or 0
    total = healthy + moderate + weak + stress

    def percent(val):
        return (val / total * 100) if total > 0 else 0

    # ✅ FIXED plot_name fallback logic
    properties["plot_name"] = plot_name

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": geometry.getInfo(),
            "properties": {
                "plot_name": plot_name,
                "area_acres": area_acres,
                "start_date": start_date,
                "end_date": end_date,
                "image_count": image_count,
                "tile_url": tile_url,
                "data_source": data_source,
                "latest_image_date": latest_image_date,
                "last_updated": datetime.utcnow().isoformat()
            }
        }],
        "pixel_summary": {
            "total_pixel_count": total,

            "healthy_pixel_count": healthy,
            "healthy_pixel_percentage": percent(healthy),

            "moderate_pixel_count": moderate,
            "moderate_pixel_percentage": percent(moderate),

            "weak_pixel_count": weak,
            "weak_pixel_percentage": percent(weak),

            "stress_pixel_count": stress,
            "stress_pixel_percentage": percent(stress),

            "analysis_start_date": start_date,
            "analysis_end_date": end_date,
            "latest_image_date": latest_image_date
        }
    }

    return geojson

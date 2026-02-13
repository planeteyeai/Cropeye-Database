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
# Growth Analysis (UNCHANGED LOGIC – MULTI SENSOR RETURN)
# ======================================================

def run_growth_analysis_by_plot(plot_data, start_date, end_date):

    if not plot_data or "geometry" not in plot_data:
        raise ValueError("plot_data missing geometry")

    geometry = plot_data["geometry"]
    props = plot_data.get("properties", {})

    area_hectares = geometry.area().divide(10000).getInfo()

    analysis_start = ee.Date(start_date)
    analysis_end = ee.Date(end_date)

    results = []

    # ======================================================
    # SENTINEL-2 (UNCHANGED)
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

        latest_s2_image = ee.Image(s2_collection.first())
        latest_s2_date = ee.Date(latest_s2_image.get("system:time_start"))
        latest_image_date = latest_s2_date.format("YYYY-MM-dd").getInfo()

        ndvi = latest_s2_image.normalizedDifference(["B8", "B4"]).rename("NDVI")

        weak_mask = ndvi.gte(0.2).And(ndvi.lt(0.4))
        stress_mask = ndvi.gte(0.0).And(ndvi.lt(0.2))
        moderate_mask = ndvi.gte(0.4).And(ndvi.lt(0.6))
        healthy_mask = ndvi.gte(0.6)

        data_source = "Sentinel-2 NDVI"
        sensor = "Sentinel-2"

        result = _build_response(
            geometry,
            props,
            area_hectares,
            weak_mask,
            stress_mask,
            moderate_mask,
            healthy_mask,
            latest_image_date,
            data_source,
            sensor,
            start_date,
            end_date
        )

        results.append(result)

    # ======================================================
    # SENTINEL-1 (UNCHANGED)
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

        latest_s1_image = ee.Image(s1_collection.first())
        latest_s1_date = ee.Date(latest_s1_image.get("system:time_start"))
        latest_image_date = latest_s1_date.format("YYYY-MM-dd").getInfo()

        vh = latest_s1_image.select("VH")

        weak_mask = vh.gte(-11)
        stress_mask = vh.lt(-11).And(vh.gt(-13))
        moderate_mask = vh.lte(-13).And(vh.gt(-15))
        healthy_mask = vh.lte(-15)

        data_source = "Sentinel-1 VH"
        sensor = "Sentinel-1"

        result = _build_response(
            geometry,
            props,
            area_hectares,
            weak_mask,
            stress_mask,
            moderate_mask,
            healthy_mask,
            latest_image_date,
            data_source,
            sensor,
            start_date,
            end_date
        )

        results.append(result)

    if not results:
        raise Exception("No Sentinel-1 or Sentinel-2 images found")

    return results


# ======================================================
# SHARED RESPONSE BUILDER (NO LOGIC CHANGED)
# ======================================================

def _build_response(
    geometry,
    props,
    area_hectares,
    weak_mask,
    stress_mask,
    moderate_mask,
    healthy_mask,
    latest_image_date,
    data_source,
    sensor,
    start_date,
    end_date
):

    combined_class = (
        ee.Image(0)
        .where(weak_mask, 1)
        .where(stress_mask, 2)
        .where(moderate_mask, 3)
        .where(healthy_mask, 4)
        .clip(geometry)
    )

    combined_smooth = combined_class.focal_mean(radius=10, units="meters")

    tile_url = (
        combined_smooth.visualize(
            min=0,
            max=4,
            palette=["#bc1e29", "#58cf54", "#28ae31", "#056c3e"]
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

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": geometry.getInfo(),
            "properties": {
                "plot_name": props.get("plot_name"),
                "area_hectares": round(area_hectares, 2),
                "data_source": data_source,
                "latest_image_date": latest_image_date,
                "last_updated": datetime.utcnow().isoformat()
            }
        }],
        "pixel_summary": {
            "total_pixel_count": total,
            "healthy_pixel_count": healthy,
            "moderate_pixel_count": moderate,
            "weak_pixel_count": weak,
            "stress_pixel_count": stress,
            "analysis_start_date": start_date,
            "analysis_end_date": end_date,
            "latest_image_date": latest_image_date
        }
    }

    return {
        "analysis_date": latest_image_date,
        "sensor": sensor,
        "tile_url": tile_url,
        "response_json": geojson
    }

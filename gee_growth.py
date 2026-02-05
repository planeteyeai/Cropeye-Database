import ee
from datetime import datetime, timedelta, date
import json
service_account_info = json.loads(os.environ["EE_SERVICE_ACCOUNT_JSON"])

credentials = ee.ServiceAccountCredentials(
    service_account_info["client_email"],
    key_data=json.dumps(service_account_info)
)

ee.Initialize(credentials, project=service_account_info["project_id"])
    
def run_growth_analysis_by_plot(plot_data, start_date, end_date):
    geometry = plot_data["geometry"]
    area_hectares = geometry.area().divide(10000).getInfo()

    analysis_start = ee.Date(start_date)
    analysis_end = ee.Date(end_date)

    # -------------------- SENTINEL-2 --------------------
    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(geometry)
        .filterDate(analysis_start, analysis_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        .map(lambda img: img.clip(geometry))
        .sort("system:time_start", False)
    )

    s2_count = s2_collection.size().getInfo()
    latest_s2_image = None
    latest_s2_date = None

    if s2_count > 0:
        latest_s2_image = ee.Image(s2_collection.first())
        latest_s2_date = ee.Date(latest_s2_image.get("system:time_start"))

    # -------------------- SENTINEL-1 --------------------
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

    s1_count = s1_collection.size().getInfo()
    latest_s1_image = None
    previous_s1_image = None
    latest_s1_date = None

    if s1_count >= 2:
        latest_s1_image = ee.Image(s1_collection.toList(2).get(0))
        previous_s1_image = ee.Image(s1_collection.toList(2).get(1))
        latest_s1_date = ee.Date(latest_s1_image.get("system:time_start"))
    elif s1_count == 1:
        latest_s1_image = ee.Image(s1_collection.first())
        latest_s1_date = ee.Date(latest_s1_image.get("system:time_start"))

    # -------------------- DECISION LOGIC --------------------
    use_s2 = False
    use_s1 = False

    if latest_s2_date and latest_s1_date:
        use_s2 = latest_s2_date.millis().getInfo() >= latest_s1_date.millis().getInfo()
        use_s1 = not use_s2
    elif latest_s2_date:
        use_s2 = True
    elif latest_s1_date:
        use_s1 = True
    else:
        raise Exception("No Sentinel-1 or Sentinel-2 images found")

    # -------------------- ANALYSIS --------------------
    if use_s2:
        analysis = latest_s2_image.normalizedDifference(["B8", "B4"]).rename("NDVI").clip(geometry)
        ndvi = analysis.select("NDVI")

        weak_mask = ndvi.gte(0.2).And(ndvi.lt(0.4))
        stress_mask = ndvi.gte(0.0).And(ndvi.lt(0.2))
        moderate_mask = ndvi.gte(0.4).And(ndvi.lt(0.6))
        healthy_mask = ndvi.gte(0.6)

        latest_image_date = latest_s2_date.format("YYYY-MM-dd").getInfo()
        data_source = "Sentinel-2 NDVI"
        sensor = "Sentinel-2"

    else:
        analysis = latest_s1_image.select("VH").clip(geometry)
        vh = analysis.select("VH")

        weak_mask = vh.gte(-11)
        stress_mask = vh.lt(-11).And(vh.gt(-13))
        moderate_mask = vh.lte(-13).And(vh.gt(-15))
        healthy_mask = vh.lte(-15)

        latest_image_date = latest_s1_date.format("YYYY-MM-dd").getInfo()
        data_source = "Sentinel-1 VH"
        sensor = "Sentinel-1"

    # -------------------- VISUALIZATION --------------------
    combined_class = (
        ee.Image(0)
        .where(weak_mask, 1)
        .where(stress_mask, 2)
        .where(moderate_mask, 3)
        .where(healthy_mask, 4)
        .clip(geometry)
    )

    combined_smooth = combined_class.focal_mean(radius=10, units="meters")

    vis_params = {
        "min": 0,
        "max": 4,
        "palette": ["#bc1e29", "#58cf54", "#28ae31", "#056c3e"]
    }

    tile_url = combined_smooth.visualize(**vis_params) \
        .clip(geometry) \
        .getMapId()["tile_fetcher"].url_format

    # -------------------- PIXEL COUNTS --------------------
    count_image = ee.Image.constant(1)

    def get_pixel_count(mask):
        return count_image.updateMask(mask).reduceRegion(
            ee.Reducer.count(), geometry, 10, bestEffort=True
        ).get("constant")

    healthy_count = get_pixel_count(healthy_mask).getInfo() or 0
    moderate_count = get_pixel_count(moderate_mask).getInfo() or 0
    weak_count = get_pixel_count(weak_mask).getInfo() or 0
    stress_count = get_pixel_count(stress_mask).getInfo() or 0
    total_pixel_count = get_pixel_count(count_image).getInfo() or 0

    # -------------------- RESPONSE JSON --------------------
    response_json = {
        "area_hectares": round(area_hectares, 2),
        "image_date": latest_image_date,
        "data_source": data_source,
        "pixel_summary": {
            "total_pixel_count": total_pixel_count,
            "healthy": healthy_count,
            "moderate": moderate_count,
            "weak": weak_count,
            "stress": stress_count,
            "healthy_pct": (healthy_count / total_pixel_count) * 100 if total_pixel_count else 0,
            "moderate_pct": (moderate_count / total_pixel_count) * 100 if total_pixel_count else 0,
            "weak_pct": (weak_count / total_pixel_count) * 100 if total_pixel_count else 0,
            "stress_pct": (stress_count / total_pixel_count) * 100 if total_pixel_count else 0,
        }
    }

    return {
        "analysis_date": latest_image_date,
        "sensor": sensor,
        "tile_url": tile_url,
        "response_json": response_json
    }

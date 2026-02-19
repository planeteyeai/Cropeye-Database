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

def run_growth_analysis_by_plot(plot_data, start_date, end_date):

    geometry = plot_data["geometry"]
    properties = plot_data.get("properties", {})
    plot_name = properties.get("plot_name", "Unknown")
    area_acres = properties.get("area_acres", 0)

    polygon = ee.Geometry(geometry)

    # ==============================
    # Sentinel-2 NDVI
    # ==============================

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(polygon)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
    )

    image_count = s2.size().getInfo()

    if image_count == 0:
        return []

    latest_image = s2.sort("system:time_start", False).first()
    latest_date = datetime.utcfromtimestamp(
        latest_image.get("system:time_start").getInfo() / 1000
    ).strftime("%Y-%m-%d")

    ndvi = latest_image.select("NDVI")

    # ==============================
    # Pixel classification
    # ==============================

    healthy = ndvi.gte(0.6)
    moderate = ndvi.gte(0.4).And(ndvi.lt(0.6))
    weak = ndvi.gte(0.2).And(ndvi.lt(0.4))
    stress = ndvi.lt(0.2)

    scale = 10

    total_pixels = ndvi.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=polygon,
        scale=scale,
        maxPixels=1e13,
    ).get("NDVI")

    healthy_pixels = healthy.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=polygon,
        scale=scale,
        maxPixels=1e13,
    ).get("NDVI")

    moderate_pixels = moderate.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=polygon,
        scale=scale,
        maxPixels=1e13,
    ).get("NDVI")

    weak_pixels = weak.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=polygon,
        scale=scale,
        maxPixels=1e13,
    ).get("NDVI")

    stress_pixels = stress.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=polygon,
        scale=scale,
        maxPixels=1e13,
    ).get("NDVI")

    total = total_pixels.getInfo() or 0
    healthy_count = healthy_pixels.getInfo() or 0
    moderate_count = moderate_pixels.getInfo() or 0
    weak_count = weak_pixels.getInfo() or 0
    stress_count = stress_pixels.getInfo() or 0

    # ==============================
    # Pixel coordinates extraction
    # ==============================

    def extract_coordinates(mask_image):
        vectors = mask_image.selfMask().reduceToVectors(
            geometry=polygon,
            scale=scale,
            geometryType="centroid",
            maxPixels=1e13,
        )
        features = vectors.getInfo()["features"]
        return [
            f["geometry"]["coordinates"]
            for f in features
        ]

    healthy_coords = extract_coordinates(healthy)
    moderate_coords = extract_coordinates(moderate)
    weak_coords = extract_coordinates(weak)
    stress_coords = extract_coordinates(stress)

    # ==============================
    # Tile URL
    # ==============================

    vis = {"min": 0, "max": 1, "palette": ["red", "yellow", "green"]}

    map_id = ndvi.getMapId(vis)
    tile_url = map_id["tile_fetcher"].url_format

    # ==============================
    # Final Response (MATCHES ADMIN)
    # ==============================

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "plot_name": plot_name,
                    "area_acres": area_acres,
                    "start_date": start_date,
                    "end_date": end_date,
                    "image_count": image_count,
                    "tile_url": tile_url,
                    "data_source": "Sentinel-2 NDVI",
                    "latest_image_date": latest_date,
                    "last_updated": datetime.utcnow().isoformat(),
                },
            }
        ],
        "pixel_summary": {
            "total_pixel_count": total,
            "healthy_pixel_count": healthy_count,
            "healthy_pixel_percentage": (healthy_count / total * 100)
            if total
            else 0,
            "healthy_pixel_coordinates": healthy_coords,
            "moderate_pixel_count": moderate_count,
            "moderate_pixel_percentage": (moderate_count / total * 100)
            if total
            else 0,
            "moderate_pixel_coordinates": moderate_coords,
            "weak_pixel_count": weak_count,
            "weak_pixel_percentage": (weak_count / total * 100)
            if total
            else 0,
            "weak_pixel_coordinates": weak_coords,
            "stress_pixel_count": stress_count,
            "stress_pixel_percentage": (stress_count / total * 100)
            if total
            else 0,
            "stress_pixel_coordinates": stress_coords,
            "analysis_start_date": start_date,
            "analysis_end_date": end_date,
            "latest_image_date": latest_date,
        },
    }

    return [geojson]

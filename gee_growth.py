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

        # Convert geometry once (fix JSON serialization issue)
        geometry_geojson = polygon.getInfo()

        results = []

        # =====================================================
        # FUNCTION TO BUILD PIXEL SUMMARY
        # =====================================================

        def build_pixel_summary(image_band, latest_date):

            classified = (
                image_band.expression(
                    """
                    b < 0.2 ? 0 :
                    b < 0.4 ? 1 :
                    b < 0.6 ? 2 :
                    3
                    """,
                    {"b": image_band}
                ).rename("class")
            )

            pixel_data = classified.addBands(ee.Image.pixelLonLat())

            samples = pixel_data.sample(
                region=polygon,
                scale=10,
                geometries=True
            ).getInfo()

            total_pixel_count = 0
            healthy_pixels = []
            moderate_pixels = []
            weak_pixels = []
            stress_pixels = []

            for feature in samples["features"]:
                total_pixel_count += 1
                value = feature["properties"]["class"]
                coords = feature["geometry"]["coordinates"]

                if value == 3:
                    healthy_pixels.append(coords)
                elif value == 2:
                    moderate_pixels.append(coords)
                elif value == 1:
                    weak_pixels.append(coords)
                elif value == 0:
                    stress_pixels.append(coords)

            def percentage(count):
                return (count / total_pixel_count * 100) if total_pixel_count else 0

            return {
                "total_pixel_count": total_pixel_count,

                "healthy_pixel_count": len(healthy_pixels),
                "healthy_pixel_percentage": percentage(len(healthy_pixels)),
                "healthy_pixel_coordinates": healthy_pixels,

                "moderate_pixel_count": len(moderate_pixels),
                "moderate_pixel_percentage": percentage(len(moderate_pixels)),
                "moderate_pixel_coordinates": moderate_pixels,

                "weak_pixel_count": len(weak_pixels),
                "weak_pixel_percentage": percentage(len(weak_pixels)),
                "weak_pixel_coordinates": weak_pixels,

                "stress_pixel_count": len(stress_pixels),
                "stress_pixel_percentage": percentage(len(stress_pixels)),
                "stress_pixel_coordinates": stress_pixels,

                "analysis_start_date": start_date,
                "analysis_end_date": end_date,
                "latest_image_date": latest_date
            }

        # =====================================================
        # SENTINEL-2 NDVI
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

            pixel_summary = build_pixel_summary(ndvi, latest_date)

            geojson_s2 = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": geometry_geojson,
                    "properties": {
                        "plot_name": plot_name,
                        "area_acres": area_acres,
                        "start_date": start_date,
                        "end_date": end_date,
                        "image_count": s2_collection.size().getInfo(),
                        "tile_url": tile_url,
                        "data_source": "Sentinel-2 NDVI",
                        "latest_image_date": latest_date,
                        "last_updated": datetime.utcnow().isoformat()
                    }
                }],
                "pixel_summary": pixel_summary
            }

            results.append(geojson_s2)

        else:
            print("⚠ No Sentinel-2 data", flush=True)

        # =====================================================
        # SENTINEL-1 VH
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

            pixel_summary = build_pixel_summary(vh, latest_date)

            geojson_s1 = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": geometry_geojson,
                    "properties": {
                        "plot_name": plot_name,
                        "area_acres": area_acres,
                        "start_date": start_date,
                        "end_date": end_date,
                        "image_count": s1_collection.size().getInfo(),
                        "tile_url": tile_url,
                        "data_source": "Sentinel-1 VH",
                        "latest_image_date": latest_date,
                        "last_updated": datetime.utcnow().isoformat()
                    }
                }],
                "pixel_summary": pixel_summary
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

# ==========================================================
# WATER UPTAKE
# ==========================================================

def run_water_uptake_analysis_by_plot(plot_name, plot_data, start_date, end_date):

    try:
        geometry = plot_data.get("geometry")
        geom_type = plot_data.get("geom_type")
        original_coords = plot_data.get("original_coords")

        if not geometry:
            return None

        polygon = ee.Geometry(geometry)

        analysis_start = ee.Date(start_date)
        analysis_end = ee.Date(end_date)

        # Sentinel-2 NDMI
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(polygon)
            .filterDate(analysis_start, analysis_end)
            .map(lambda img: img.addBands(
                img.normalizedDifference(['B8A', 'B11']).rename('NDMI')
            ))
            .sort("system:time_start", False)
        )

        # Sentinel-1
        s1_collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(polygon)
            .filterDate(analysis_start, analysis_end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VH"])
            .sort("system:time_start", False)
        )

        s2_count = s2_collection.size().getInfo()
        s1_count = s1_collection.size().getInfo()

        if s2_count == 0 and s1_count == 0:
            return None

        # Use Sentinel-1 logic for classification (same as your sample)
        image_list = s1_collection.toList(2)
        latest_image = ee.Image(image_list.get(0))
        previous_image = ee.Image(image_list.get(1)) if s1_count > 1 else latest_image

        delta_vh = latest_image.subtract(previous_image).rename("deltaVH").clip(polygon)

        deficient = delta_vh.lt(-3.13)
        less = delta_vh.gte(-3.13).And(delta_vh.lte(-0.1))
        adequate = delta_vh.gt(-0.1).And(delta_vh.lt(4))
        excellent = delta_vh.gte(4).And(delta_vh.lt(6))
        excess = delta_vh.gte(6)

        combined = (
            ee.Image(0)
            .where(deficient, 1)
            .where(less, 2)
            .where(adequate, 3)
            .where(excellent, 4)
            .where(excess, 5)
            .clip(polygon)
        )

        tile_url = combined.getMapId()["tile_fetcher"].url_format

        def get_mask_data(mask):
            count = (
                ee.Image.constant(1)
                .updateMask(mask)
                .reduceRegion(ee.Reducer.count(), polygon, 10, bestEffort=True)
                .get("constant")
                .getInfo() or 0
            )

            samples = (
                mask.selfMask()
                .addBands(ee.Image.pixelLonLat())
                .sample(region=polygon, scale=10, geometries=True)
                .getInfo()
            )

            coords = [f["geometry"]["coordinates"] for f in samples.get("features", [])]
            return count, coords

        total_pixels = (
            ee.Image.constant(1)
            .reduceRegion(ee.Reducer.count(), polygon, 10, bestEffort=True)
            .get("constant")
            .getInfo()
        )

        deficient_count, deficient_coords = get_mask_data(combined.eq(1))
        less_count, less_coords = get_mask_data(combined.eq(2))
        adequate_count, adequate_coords = get_mask_data(combined.eq(3))
        excellent_count, excellent_coords = get_mask_data(combined.eq(4))
        excess_count, excess_coords = get_mask_data(combined.eq(5))

        def percent(x):
            return (x / total_pixels * 100) if total_pixels else 0

        feature = {
            "type": "Feature",
            "geometry": {"type": geom_type, "coordinates": original_coords},
            "properties": {
                "plot_name": plot_name,
                "tile_url": tile_url,
                "sensor": "s1",
                "image_count_in_range": s1_count,
                "analysis_dates": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "latest_image_date": ee.Date(
                        latest_image.get("system:time_start")
                    ).format("YYYY-MM-dd").getInfo(),
                    "previous_image_date": ee.Date(
                        previous_image.get("system:time_start")
                    ).format("YYYY-MM-dd").getInfo()
                },
                "last_updated": datetime.utcnow().isoformat(),
            },
        }

        result = {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": {
                "total_pixel_count": total_pixels,

                "deficient_pixel_count": deficient_count,
                "deficient_pixel_percentage": percent(deficient_count),
                "deficient_pixel_coordinates": deficient_coords,

                "less_pixel_count": less_count,
                "less_pixel_percentage": percent(less_count),
                "less_pixel_coordinates": less_coords,

                "adequat_pixel_count": adequate_count,
                "adequat_pixel_percentage": percent(adequate_count),
                "adequat_pixel_coordinates": adequate_coords,

                "excellent_pixel_count": excellent_count,
                "excellent_pixel_percentage": percent(excellent_count),
                "excellent_pixel_coordinates": excellent_coords,

                "excess_pixel_count": excess_count,
                "excess_pixel_percentage": percent(excess_count),
                "excess_pixel_coordinates": excess_coords,

                "analysis_start_date": start_date,
                "analysis_end_date": end_date,
            }
        }

        return [result]

    except Exception as e:
        print("❌ Water uptake failed:", e, flush=True)
        return None


# ==========================================================
# SOIL MOISTURE (FULL PIXEL STRUCTURE MATCHING YOUR SAMPLE)
# ==========================================================

def run_soil_moisture_analysis_by_plot(plot_name, plot_data, start_date, end_date):

    try:
        geometry = plot_data.get("geometry")
        geom_type = plot_data.get("geom_type")
        original_coords = plot_data.get("original_coords")

        if not geometry:
            return None

        polygon = ee.Geometry(geometry)

        s1_collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(polygon)
            .filterDate(start_date, end_date)
        )

        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(polygon)
            .filterDate(start_date, end_date)
        )

        s1_size = s1_collection.size().getInfo()
        s2_size = s2_collection.size().getInfo()

        if s1_size == 0 and s2_size == 0:
            return None

        image = (
            s1_collection
            .select("VH")
            .median()
            .clip(polygon)
        )

        less = image.lt(-15)
        adequate = image.gte(-15).And(image.lt(-5))
        excellent = image.gte(-5).And(image.lt(2))
        excess = image.gte(2)

        def extract(mask):
            count = (
                ee.Image.constant(1)
                .updateMask(mask)
                .reduceRegion(ee.Reducer.count(), polygon, 10, bestEffort=True)
                .get("constant")
                .getInfo() or 0
            )

            samples = (
                mask.selfMask()
                .addBands(ee.Image.pixelLonLat())
                .sample(region=polygon, scale=10, geometries=True)
                .getInfo()
            )

            coords = [f["geometry"]["coordinates"] for f in samples.get("features", [])]
            return count, coords

        total_pixels = (
            ee.Image.constant(1)
            .reduceRegion(ee.Reducer.count(), polygon, 10, bestEffort=True)
            .get("constant")
            .getInfo()
        )

        less_count, less_coords = extract(less)
        adequate_count, adequate_coords = extract(adequate)
        excellent_count, excellent_coords = extract(excellent)
        excess_count, excess_coords = extract(excess)

        def percent(x):
            return round((x / total_pixels * 100), 2) if total_pixels else 0

        feature = {
            "type": "Feature",
            "geometry": {"type": geom_type, "coordinates": original_coords},
            "properties": {
                "plot_name": plot_name,
                "start_date": start_date,
                "end_date": end_date,
                "sensor_used": "Sentinel-1",
                "latest_image_date": None,
                "image_count": s1_size,
                "tile_url": image.getMapId()["tile_fetcher"].url_format,
                "last_updated": datetime.utcnow().isoformat(),
            }
        }

        result = {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": {
                "total_pixel_count": total_pixels,
                "start_date": start_date,
                "end_date": end_date,
                "latest_image_date": None,
                "sensor_used": "Sentinel-1",
                "s1_available": s1_size > 0,
                "s2_available": s2_size > 0,
                "s1_latest_date": None,
                "s2_latest_date": None,

                "less_pixel_count": less_count,
                "less_pixel_percentage": percent(less_count),
                "less_pixel_coordinates": less_coords,

                "adequate_pixel_count": adequate_count,
                "adequate_pixel_percentage": percent(adequate_count),
                "adequate_pixel_coordinates": adequate_coords,

                "excellent_pixel_count": excellent_count,
                "excellent_pixel_percentage": percent(excellent_count),
                "excellent_pixel_coordinates": excellent_coords,

                "excess_pixel_count": excess_count,
                "excess_pixel_percentage": percent(excess_count),
                "excess_pixel_coordinates": excess_coords,
            }
        }

        return [result]

    except Exception as e:
        print("❌ Soil moisture failed:", e, flush=True)
        return None


# ==========================================================
# PEST DETECTION (FULL MATCH)
# ==========================================================

def run_pest_detection_analysis_by_plot(plot_name, plot_data, start_date, end_date):

    try:
        geometry = plot_data.get("geometry")
        geom_type = plot_data.get("geom_type")
        original_coords = plot_data.get("original_coords")

        if not geometry:
            return None

        polygon = ee.Geometry(geometry)

        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(polygon)
            .filterDate(start_date, end_date)
        )

        if s1.size().getInfo() == 0:
            return None

        image = (
            s1
            .select("VH")
            .median()
            .clip(polygon)
        )


        chewing_mask = image.lt(-18)

        def extract(mask):
            count = (
                ee.Image.constant(1)
                .updateMask(mask)
                .reduceRegion(ee.Reducer.count(), polygon, 10, bestEffort=True)
                .get("constant")
                .getInfo() or 0
            )

            samples = (
                mask.selfMask()
                .addBands(ee.Image.pixelLonLat())
                .sample(region=polygon, scale=10, geometries=True)
                .getInfo()
            )

            coords = [f["geometry"]["coordinates"] for f in samples.get("features", [])]
            return count, coords

        total_pixels = (
            ee.Image.constant(1)
            .reduceRegion(ee.Reducer.count(), polygon, 10, bestEffort=True)
            .get("constant")
            .getInfo()
        )

        chewing_count, chewing_coords = extract(chewing_mask)

        def percent(x):
            return (x / total_pixels * 100) if total_pixels else 0

        feature = {
            "type": "Feature",
            "geometry": {"type": geom_type, "coordinates": original_coords},
            "properties": {
                "plot_name": plot_name,
                "start_date": start_date,
                "end_date": end_date,
                "image_count": s1.size().getInfo(),
                "image_dates": [],
                "tile_url": image.getMapId()["tile_fetcher"].url_format,
                "last_updated": datetime.utcnow().isoformat(),
            }
        }

        result = {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": {
                "total_pixel_count": total_pixels,
                "healthy_pixel_count": total_pixels - chewing_count,
                "chewing_affected_pixel_count": chewing_count,
                "chewing_affected_pixel_percentage": percent(chewing_count),
                "chewing_affected_pixel_coordinates": chewing_coords,

                "fungi_affected_pixel_count": 0,
                "fungi_affected_pixel_percentage": 0,
                "fungi_affected_pixel_coordinates": [],

                "sucking_affected_pixel_count": 0,
                "sucking_affected_pixel_percentage": 0,
                "sucking_affected_pixel_coordinates": [],

                "wilt_affected_pixel_count": 0,
                "wilt_affected_pixel_percentage": 0,
                "wilt_affected_pixel_coordinates": [],

                "SoilBorn_pixel_count": 0,
                "SoilBorn_affected_pixel_percentage": 0,
                "SoilBorn_affected_pixel_coordinates": [],

                "baseline_start_date": None,
                "baseline_end_date": None,
                "analysis_start_date": start_date,
                "analysis_end_date": end_date,
            }
        }

        return [result]

    except Exception as e:
        print("❌ Pest detection failed:", e, flush=True)
        return None

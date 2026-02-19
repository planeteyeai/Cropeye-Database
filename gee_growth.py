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

def run_water_uptake_analysis_by_plot(plot_name, plot_data, start_date, end_date):

    try:
        geometry = plot_data.get("geometry")
        geom_type = plot_data.get("geom_type")
        original_coords = plot_data.get("original_coords")

        if not geometry:
            print("⚠ Missing geometry", flush=True)
            return None

        polygon = ee.Geometry(geometry)

        analysis_start = ee.Date(start_date)
        analysis_end = ee.Date(end_date)

        # ---------------- Sentinel-2 NDMI ----------------
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(polygon)
            .filterDate(analysis_start, analysis_end)
            .map(lambda img: img.clip(polygon))
            .map(lambda img: img.addBands(
                img.normalizedDifference(['B8A', 'B11']).rename('NDMI')
            ))
            .select(['NDMI'])
            .sort("system:time_start", False)
        )

        s2_count = s2_collection.size().getInfo()
        latest_s2_date = None
        latest_s2_image = None

        if s2_count > 0:
            latest_s2_image = ee.Image(s2_collection.first())
            latest_s2_date = ee.Date(latest_s2_image.get("system:time_start"))

        # ---------------- Sentinel-1 VH ----------------
        s1_collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(polygon)
            .filterDate(analysis_start, analysis_end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VH"])
            .map(lambda img: img.clip(polygon))
            .sort("system:time_start", False)
        )

        s1_count = s1_collection.size().getInfo()
        latest_image = None
        previous_image = None
        latest_s1_date = None

        if s1_count >= 2:
            latest_image = ee.Image(s1_collection.toList(2).get(0))
            previous_image = ee.Image(s1_collection.toList(2).get(1))
            latest_s1_date = ee.Date(latest_image.get("system:time_start"))

        # ---------------- Decide dataset ----------------
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
            print("⚠ No Sentinel data", flush=True)
            return None

        # ---------------- Classification ----------------
        if use_s2:
            sensor = "s2"
            image = s2_collection.median().clip(polygon)

            deficient = image.lt(-0.21)
            less = image.gte(-0.21).And(image.lt(-0.031))
            adequat = image.gte(-0.031).And(image.lt(0.142))
            excellent = image.gte(0.14).And(image.lt(0.22))
            excess = image.gte(0.22)

            latest_image_date = latest_s2_date.format("YYYY-MM-dd").getInfo()
            previous_image_date = None

        else:
            sensor = "s1"
            delta_vh = latest_image.subtract(previous_image).rename("deltaVH").clip(polygon)

            excess = delta_vh.gte(6)
            excellent = delta_vh.gte(4.0).And(delta_vh.lt(6))
            adequat = delta_vh.gt(-0.1).And(delta_vh.lt(4))
            less = delta_vh.gte(-3.13).And(delta_vh.lte(-0.1))
            deficient = delta_vh.lt(-3.13)

            latest_image_date = latest_s1_date.format("YYYY-MM-dd").getInfo()
            previous_image_date = ee.Date(
                previous_image.get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()

        # ---------------- Combined Class ----------------
        combined_class = (
            ee.Image(0)
            .where(deficient, 1)
            .where(less, 2)
            .where(adequat, 3)
            .where(excellent, 4)
            .where(excess, 5)
            .clip(polygon)
        )

        smoothed = combined_class.focal_mean(radius=7, units="meters")

        tile_url = smoothed.visualize(
            min=0,
            max=5,
            palette=["#EBFF34", "#CC8213AF", "#1348E88E", "#2E199ABD", "#0602178F"]
        ).getMapId()["tile_fetcher"].url_format

        # ---------------- Pixel Counts ----------------
        count_image = ee.Image.constant(1)

        def get_pixel_count(mask):
            return (
                count_image.updateMask(mask)
                .reduceRegion(ee.Reducer.count(), polygon, 10, bestEffort=True)
                .get("constant")
                .getInfo() or 0
            )

        def mask_to_coords(mask):
            samples = (
                mask.selfMask()
                .addBands(ee.Image.pixelLonLat())
                .sample(region=polygon, scale=10, geometries=True, tileScale=4)
                .getInfo()
            )
            coords = [f["geometry"]["coordinates"] for f in samples.get("features", [])]
            return [list(x) for x in {tuple(c) for c in coords}]

        total_pixel_count = get_pixel_count(count_image)

        deficient_count = get_pixel_count(combined_class.eq(1))
        less_count = get_pixel_count(combined_class.eq(2))
        adequat_count = get_pixel_count(combined_class.eq(3))
        excellent_count = get_pixel_count(combined_class.eq(4))
        excess_count = get_pixel_count(combined_class.eq(5))

        # ---------------- Build Response ----------------
        feature = {
            "type": "Feature",
            "geometry": {
                "type": geom_type,
                "coordinates": original_coords,
            },
            "properties": {
                "plot_name": plot_name,
                "tile_url": tile_url,
                "sensor": sensor,
                "image_count_in_range": s2_count if use_s2 else s1_count,
                "analysis_dates": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "latest_image_date": latest_image_date,
                    "previous_image_date": previous_image_date
                },
                "last_updated": datetime.utcnow().isoformat(),
            },
        }

        result = {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": {
                "total_pixel_count": total_pixel_count,
                "deficient_pixel_count": deficient_count,
                "less_pixel_count": less_count,
                "adequat_pixel_count": adequat_count,
                "excellent_pixel_count": excellent_count,
                "excess_pixel_count": excess_count,
                "analysis_start_date": start_date,
                "analysis_end_date": end_date,
            }
        }

        return [result]   # IMPORTANT: return as list for cron

    except Exception as e:
        print(f"❌ Water uptake analysis failed: {e}", flush=True)
        return None

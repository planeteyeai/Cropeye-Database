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

        results = []

        # ======================= SENTINEL-2 NDMI =======================
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(polygon)
            .filterDate(analysis_start, analysis_end)
            .map(lambda img: img.addBands(
                img.normalizedDifference(['B8A', 'B11']).rename('NDMI')
            ))
            .sort("system:time_start", False)
        )

        s2_count = s2_collection.size().getInfo()

        if s2_count > 0:

            ndmi_image = s2_collection.median().clip(polygon)

            deficient = ndmi_image.lt(-0.21)
            less = ndmi_image.gte(-0.21).And(ndmi_image.lt(-0.031))
            adequate = ndmi_image.gte(-0.031).And(ndmi_image.lt(0.142))
            excellent = ndmi_image.gte(0.14).And(ndmi_image.lt(0.22))
            excess = ndmi_image.gte(0.22)

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

            latest_s2_image = ee.Image(s2_collection.first())
            latest_s2_date = ee.Date(
                latest_s2_image.get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()

            results.append(build_water_result(
                plot_name, geom_type, original_coords,
                combined, polygon, tile_url,
                "s2", s2_count,
                start_date, end_date,
                latest_s2_date, None
            ))

        # ======================= SENTINEL-1 =======================
        s1_collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(polygon)
            .filterDate(analysis_start, analysis_end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VH"])
            .sort("system:time_start", False)
        )

        s1_count = s1_collection.size().getInfo()

        if s1_count >= 2:

            image_list = s1_collection.toList(2)
            latest_image = ee.Image(image_list.get(0))
            previous_image = ee.Image(image_list.get(1))

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

            latest_date = ee.Date(
                latest_image.get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()

            previous_date = ee.Date(
                previous_image.get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()

            results.append(build_water_result(
                plot_name, geom_type, original_coords,
                combined, polygon, tile_url,
                "s1", s1_count,
                start_date, end_date,
                latest_date, previous_date
            ))

        return results if results else None

    except Exception as e:
        print("❌ Water uptake failed:", e, flush=True)
        return None

def build_water_result(
    plot_name, geom_type, original_coords,
    combined, polygon, tile_url,
    sensor, image_count,
    start_date, end_date,
    latest_image_date, previous_image_date
):

    count_image = ee.Image.constant(1)

    def get_mask_data(mask):
        count = (
            count_image.updateMask(mask)
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
        count_image.reduceRegion(ee.Reducer.count(), polygon, 10, bestEffort=True)
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
            "sensor": sensor,
            "image_count_in_range": image_count,
            "analysis_dates": {
                "start_date": start_date,
                "end_date": end_date,
                "latest_image_date": latest_image_date,
                "previous_image_date": previous_image_date
            },
            "last_updated": datetime.utcnow().isoformat(),
        },
    }

    return {
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

# ==========================================================
# SOIL MOISTURE (FULL PIXEL STRUCTURE MATCHING YOUR SAMPLE)
# ==========================================================

def run_soil_moisture_analysis_by_plot(plot_name, plot_data, start_date, end_date):

    try:
        geometry = plot_data["geometry"]
        geom_type = plot_data["geom_type"]
        original_coords = plot_data["original_coords"]

        # ==================== SENTINEL-1 ====================
        s1_collection = (
            filter_s1(
                ee.ImageCollection("COPERNICUS/S1_GRD"),
                start_date, end_date, geometry
            )
            .map(addIndices)
        )

        s1_size = s1_collection.size().getInfo()
        s1_latest_date = None

        if s1_size > 0:
            s1_sorted = s1_collection.sort("system:time_start", False)
            s1_latest_img = ee.Image(s1_sorted.first())
            s1_latest_date = ee.Date(
                s1_latest_img.get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()

        # ==================== SENTINEL-2 ====================
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            .map(lambda img: img.clip(geometry))
        )

        s2_size = s2_collection.size().getInfo()
        s2_latest_date = None

        if s2_size > 0:
            s2_sorted = s2_collection.sort("system:time_start", False)
            s2_latest_img = ee.Image(s2_sorted.first())
            s2_latest_date = ee.Date(
                s2_latest_img.get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()

        # ==================== SENSOR SELECTION ====================
        if s1_size == 0 and s2_size == 0:
            return None

        if s1_latest_date and s2_latest_date:
            if s2_latest_date >= s1_latest_date:
                use_sentinel2 = True
                sensor_used = "Sentinel-2"
                latest_date = s2_latest_date
            else:
                use_sentinel2 = False
                sensor_used = "Sentinel-1"
                latest_date = s1_latest_date
        elif s2_latest_date:
            use_sentinel2 = True
            sensor_used = "Sentinel-2"
            latest_date = s2_latest_date
        else:
            use_sentinel2 = False
            sensor_used = "Sentinel-1"
            latest_date = s1_latest_date

        # ==================== CLASSIFICATION ====================
        if use_sentinel2:

            s2_composite = s2_collection.median().clip(geometry)
            ndwi = s2_composite.normalizedDifference(["B3", "B8"]).rename("NDWI")

            classified = (
                ndwi.where(ndwi.lte(-0.4), 1)
                .where(ndwi.gt(-0.4).And(ndwi.lte(-0.3)), 2)
                .where(ndwi.gt(-0.3).And(ndwi.lte(0)), 3)
                .where(ndwi.gt(0).And(ndwi.lte(0.2)), 4)
                .where(ndwi.gt(0.2), 5)
            ).clip(geometry)

            image_count = s2_size

        else:

            vv_composite = safe_median(
                s1_collection.select(["VV"])
            ).clip(geometry)

            vv_band = vv_composite.select("VV")

            classified = (
                vv_band.where(vv_band.gt(-6), 5)
                .where(vv_band.gt(-8).And(vv_band.lte(-6)), 4)
                .where(vv_band.gt(-10).And(vv_band.lte(-8)), 3)
                .where(vv_band.gt(-12).And(vv_band.lte(-10)), 2)
                .where(vv_band.lte(-12), 1)
            ).clip(geometry)

            image_count = s1_size

        # ==================== VISUALIZATION ====================
        labels = {
            1: "less",
            2: "adequate",
            3: "excellent",
            4: "excess",
            5: "shallow water",
        }

        palette = [
            "#2FC0D3",
            "#4365D4",
            "#473CDF",
            "#2116BF",
            "#000475",
        ]

        vis_params = {"min": 1, "max": 5, "palette": palette}

        smoothed = classified.focal_mean(radius=20, units="meters")
        visual = smoothed.visualize(**vis_params).clip(geometry)
        tile_url = visual.getMapId()["tile_fetcher"].url_format

        # ==================== PIXEL COUNTING ====================
        count_image = ee.Image.constant(1)

        total_pixel_count = (
            count_image.reduceRegion(
                ee.Reducer.count(), geometry, 10, bestEffort=True
            ).get("constant").getInfo()
        )

        def get_pixel_count(mask):
            return (
                count_image.updateMask(mask)
                .reduceRegion(
                    ee.Reducer.count(), geometry, 10, bestEffort=True
                ).get("constant").getInfo() or 0
            )

        def mask_to_coords(mask):
            points = (
                mask.selfMask()
                .addBands(ee.Image.pixelLonLat())
                .sample(region=geometry, scale=10, geometries=True, tileScale=4)
                .getInfo()
            )
            coords = [
                f["geometry"]["coordinates"]
                for f in points.get("features", [])
            ]
            return [list(x) for x in {tuple(c) for c in coords}]

        pixel_summary = {
            "total_pixel_count": total_pixel_count,
            "start_date": start_date,
            "end_date": end_date,
            "latest_image_date": latest_date,
            "sensor_used": sensor_used,
            "s1_available": s1_size > 0,
            "s2_available": s2_size > 0,
            "s1_latest_date": s1_latest_date,
            "s2_latest_date": s2_latest_date,
        }

        for class_id in range(1, 6):

            label = labels[class_id]
            mask = classified.eq(class_id)

            count = get_pixel_count(mask)
            percent = (
                (count / total_pixel_count) * 100
                if total_pixel_count else 0
            )

            coords = mask_to_coords(mask)

            key_base = label.lower().replace(" ", "_")

            pixel_summary[f"{key_base}_pixel_count"] = count
            pixel_summary[f"{key_base}_pixel_percentage"] = round(percent, 2)
            pixel_summary[f"{key_base}_pixel_coordinates"] = coords

        # ==================== FEATURE ====================
        feature = {
            "type": "Feature",
            "geometry": {
                "type": geom_type,
                "coordinates": original_coords,
            },
            "properties": {
                "plot_name": plot_name,
                "start_date": start_date,
                "end_date": end_date,
                "sensor_used": sensor_used,
                "latest_image_date": latest_date,
                "image_count": image_count,
                "analysis_dates": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "latest_image_date": latest_date,
                },
                "tile_url": tile_url,
                "last_updated": datetime.utcnow().isoformat(),
            },
        }

        # ==================== FINAL RESPONSE ====================
        return {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": pixel_summary,
        }

    except Exception as e:
        print("❌ Soil moisture failed:", e)
        return None
# ==========================================================
# PEST DETECTION (FIXED)
# ==========================================================

def run_pest_detection_analysis_by_plot(
    plot_name: str,
    plot_data: dict,
    start_date: str,
    end_date: str,
):
    try:
        geometry = plot_data["geometry"]
        geom_type = plot_data["geom_type"]
        original_coords = plot_data["original_coords"]

        props = plot_data.get("properties", {})
        crop_type = (
            props.get("crop_type_name")
            or props.get("crop_type")
            or props.get("crop")
            or ""
        )
        crop_type = str(crop_type).lower().strip()

        if not crop_type:
            raise ValueError("Crop type missing in plot metadata")

        analysis_start = ee.Date(start_date)
        analysis_end = ee.Date(end_date)

        baseline_year = analysis_end.get("year").subtract(1)

        # ========================================================
        # GRAPES LOGIC
        # ========================================================
        if "grape" in crop_type:

            baseline_start = ee.Date.fromYMD(baseline_year, 4, 1)
            baseline_end = ee.Date.fromYMD(baseline_year, 6, 30)

            def db_to_linear(img):
                return ee.Image(10).pow(img.divide(10)).add(1e-6)

            s1 = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
                .select(["VV", "VH"])
            )

            image_count = int(s1.size().getInfo())
            if image_count == 0:
                return None

            cur = s1.median().clip(geometry)

            base = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(geometry)
                .filterDate(baseline_start, baseline_end)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
                .select(["VV", "VH"])
                .median()
                .clip(geometry)
            )

            vv_cur = db_to_linear(cur.select("VV"))
            vh_cur = db_to_linear(cur.select("VH"))
            vv_base = db_to_linear(base.select("VV"))
            vh_base = db_to_linear(base.select("VH"))

            rvi_cur = vh_cur.multiply(4).divide(vv_cur.add(vh_cur))
            rvi_base = vh_base.multiply(4).divide(vv_base.add(vh_base))

            vhvv_cur = vh_cur.divide(vv_cur)
            vhvv_base = vh_base.divide(vv_base)

            fungi_mask = (
                rvi_cur.subtract(rvi_base).lt(-0.55)
                .And(vhvv_cur.subtract(vhvv_base).lt(-0.15))
                .And(vh_cur.subtract(vh_base).lt(0))
            )

            sucking_mask = (
                rvi_base.subtract(rvi_cur).gt(0.20)
                .And(vhvv_base.subtract(vhvv_cur).gt(0.10))
                .And(vh_cur.subtract(vh_base).lt(-0.01))
            )

            chewing_mask = (
                vh_base.subtract(vh_cur).gt(0.05)
                .And(rvi_base.subtract(rvi_cur).gt(0.06))
                .And(vhvv_base.subtract(vhvv_cur).gt(0.10))
            )

            wilt_mask = ee.Image(0)
            soilborne_mask = chewing_mask
            tile_image = cur.select("VV")

        # ========================================================
        # SUGARCANE LOGIC
        # ========================================================
        elif "sugar" in crop_type:

            baseline_start = ee.Date.fromYMD(baseline_year, 6, 1)
            baseline_end = ee.Date.fromYMD(baseline_year, 10, 30)

            s1 = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .select(["VV", "VH"])
            )

            image_count = int(s1.size().getInfo())
            if image_count == 0:
                return None

            img = s1.median().clip(geometry)

            vv = img.select("VV")
            vh = img.select("VH")
            ratio = vv.divide(vh.add(1e-6))

            chewing_mask = ratio.lte(0.01)
            fungi_mask = chewing_mask
            sucking_mask = ratio.gte(1)
            wilt_mask = ratio.gte(1)
            soilborne_mask = chewing_mask

            tile_image = img.select("VV")

        else:
            raise ValueError(f"Pest detection not supported for crop type: {crop_type}")

        tile_url = tile_image.getMapId()["tile_fetcher"].url_format

        one = ee.Image.constant(1)

        def count(mask):
            value = one.updateMask(mask).reduceRegion(
                ee.Reducer.count(), geometry, 10, bestEffort=True
            ).get("constant")
            return int(value.getInfo() or 0)

        total_pixel_count = int(
            one.reduceRegion(
                ee.Reducer.count(), geometry, 10, bestEffort=True
            ).get("constant").getInfo()
        )

        chewing_pixel_count = count(chewing_mask)
        fungi_pixel_count = count(fungi_mask)
        sucking_pixel_count = count(sucking_mask)
        wilt_pixel_count = count(wilt_mask)
        soilborne_pixel_count = count(soilborne_mask)

        healthy_pixel_count = total_pixel_count - (
            chewing_pixel_count
            + fungi_pixel_count
            + sucking_pixel_count
            + wilt_pixel_count
            + soilborne_pixel_count
        )

        def percent(x):
            return (x / total_pixel_count) * 100 if total_pixel_count else 0

        def mask_to_coords(mask):
            points = (
                mask.selfMask()
                .addBands(ee.Image.pixelLonLat())
                .sample(region=geometry, scale=10, geometries=True, tileScale=4)
                .getInfo()
            )
            coords = [f["geometry"]["coordinates"] for f in points.get("features", [])]
            return [list(x) for x in {tuple(c) for c in coords}]

        chewing_coords = mask_to_coords(chewing_mask)
        fungi_coords = mask_to_coords(fungi_mask)
        sucking_coords = mask_to_coords(sucking_mask)
        wilt_coords = mask_to_coords(wilt_mask)
        soilborne_coords = mask_to_coords(soilborne_mask)

        # ✅ ADDED THIS BLOCK TO FIX YOUR ERROR
        analysis_dates = {
            "baseline_start_date": baseline_start.format("YYYY-MM-dd").getInfo(),
            "baseline_end_date": baseline_end.format("YYYY-MM-dd").getInfo(),
            "analysis_start_date": analysis_start.format("YYYY-MM-dd").getInfo(),
            "analysis_end_date": analysis_end.format("YYYY-MM-dd").getInfo(),
        }

        feature = {
            "type": "Feature",
            "geometry": {
                "type": geom_type,
                "coordinates": original_coords,
            },
            "properties": {
                "plot_name": plot_name,
                "crop_type": crop_type,
                "start_date": start_date,
                "end_date": end_date,
                "image_count": image_count,
                "image_dates": [],
                "analysis_dates": analysis_dates,  # ✅ FIX
                "tile_url": tile_url,
                "last_updated": datetime.utcnow().isoformat(),
            },
        }

        return {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": {
                "total_pixel_count": total_pixel_count,
                "healthy_pixel_count": healthy_pixel_count,

                "chewing_affected_pixel_count": chewing_pixel_count,
                "chewing_affected_pixel_percentage": percent(chewing_pixel_count),
                "chewing_affected_pixel_coordinates": chewing_coords,

                "fungi_affected_pixel_count": fungi_pixel_count,
                "fungi_affected_pixel_percentage": percent(fungi_pixel_count),
                "fungi_affected_pixel_coordinates": fungi_coords,

                "sucking_affected_pixel_count": sucking_pixel_count,
                "sucking_affected_pixel_percentage": percent(sucking_pixel_count),
                "sucking_affected_pixel_coordinates": sucking_coords,

                "wilt_affected_pixel_count": wilt_pixel_count,
                "wilt_affected_pixel_percentage": percent(wilt_pixel_count),
                "wilt_affected_pixel_coordinates": wilt_coords,

                "SoilBorn_pixel_count": soilborne_pixel_count,
                "SoilBorn_affected_pixel_percentage": percent(soilborne_pixel_count),
                "SoilBorn_affected_pixel_coordinates": soilborne_coords,

                **analysis_dates,  # ✅ SAME DATES ALSO HERE
            },
        }

    except Exception as e:
        print("❌ Pest detection failed:", e, flush=True)
        return None

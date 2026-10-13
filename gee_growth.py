import ee
from datetime import datetime
import json
import os

from s3_tiles import pick_preferred_sensor_result, resolve_tile_url
from ee_utils import ee_get_info, sleep_between_windows

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

def build_feature(plot_name, sensor, latest_date):
    return {
        "plot_name": plot_name,
        "sensor": sensor,
        "latest_image_date": latest_date
    }


def run_growth_analysis_by_plot(
    plot_name,
    plot_data,
    start_date,
    end_date,
    plot_id=None,
    prefer_s2=True,
):

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

            sensor_used = "Sentinel-2 NDVI"  # FIX

            image = s2_collection.first()

            latest_date = ee.Date(
                image.get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()

            ndvi = image.select("NDVI")

            vis = ndvi.visualize(
                min=0, max=1, palette=["red", "yellow", "green"]
            )
            tile_url = resolve_tile_url(
                vis, polygon, plot_id, "growth", latest_date
            )

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
                        "sensor_used": sensor_used,
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
        # SENTINEL-1 VH (fallback when prefer_s2 and no S2)
        # =====================================================

        run_s1 = (not prefer_s2) or (len(results) == 0)

        if run_s1:
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

                sensor_used = "Sentinel-1 VH"  # FIX

                image = s1_collection.first()

                latest_date = ee.Date(
                    image.get("system:time_start")
                ).format("YYYY-MM-dd").getInfo()

                vh = image.select("VH")

                vis = vh.visualize(min=-25, max=0)
                tile_url = resolve_tile_url(
                    vis, polygon, plot_id, "growth", latest_date
                )

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
                            "sensor_used": sensor_used,
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

        return pick_preferred_sensor_result(results, prefer_s2=prefer_s2)

    except Exception as e:
        print(f"❌ Growth analysis failed: {e}", flush=True)
        return None

# ==========================================================
# WATER UPTAKE
# ==========================================================

def run_water_uptake_analysis_by_plot(
    plot_name,
    plot_data,
    start_date,
    end_date,
    plot_id=None,
    prefer_s2=True,
):

    try:
        geometry = plot_data.get("geometry")
        geom_type = plot_data.get("geom_type")
        original_coords = plot_data.get("original_coords")

        if not geometry or not geom_type or not original_coords:
            print("⚠ Missing geometry", flush=True)
            return None

        if not isinstance(geometry, ee.Geometry):
            geometry = ee.Geometry(geometry)

        analysis_start = ee.Date(start_date)
        analysis_end = ee.Date(end_date)

        results = []

        # =========================================================
        # COMMON HELPERS
        # =========================================================
        def get_pixel_count(mask):
            try:
                value = ee.Image.constant(1).updateMask(mask).reduceRegion(
                    ee.Reducer.count(), geometry, 10, bestEffort=True
                ).get("constant")
                return int(value.getInfo() or 0)
            except:
                return 0

        def mask_to_coords(mask):
            try:
                sampled = mask.selfMask().addBands(
                    ee.Image.pixelLonLat()
                ).sample(
                    region=geometry,
                    scale=10,
                    geometries=True,
                    tileScale=4,
                    numPixels=10000
                ).getInfo()

                coords = [f["geometry"]["coordinates"] for f in sampled.get("features", [])]
                return [list(x) for x in {tuple(c) for c in coords}]
            except:
                return []

        # =========================================================
        # SENTINEL-2 (NDWI)
        # =========================================================
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geometry)
            .filterDate(analysis_start, analysis_end)
            .map(lambda img:
                img.normalizedDifference(['B3', 'B8'])
                   .rename('NDWI')
                   .clip(geometry)
                   .copyProperties(img, ["system:time_start"])
            )
            .sort("system:time_start", False)
        )

        if s2_collection.size().getInfo() > 0:

            image = ee.Image(s2_collection.first())
            latest_date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()

            ndwi = s2_collection.median().clip(geometry)

            deficient = ndwi.lt(-0.21)
            less = ndwi.gte(-0.21).And(ndwi.lt(-0.031))
            adequat = ndwi.gte(-0.031).And(ndwi.lt(0.142))
            excellent = ndwi.gte(0.14).And(ndwi.lt(0.22))
            excess = ndwi.gte(0.22)

            combined = (
                ee.Image.constant(0)
                .where(deficient, 1)
                .where(less, 2)
                .where(adequat, 3)
                .where(excellent, 4)
                .where(excess, 5)
                .clip(geometry)
            )

            tile_vis = combined.focal_mean(radius=7, units="meters").visualize(
                min=0, max=5,
                palette=["#EBFF34", "#CC8213AF", "#1348E88E", "#2E199ABD", "#0602178F"]
            )
            tile_url = resolve_tile_url(
                tile_vis, geometry, plot_id, "water_uptake", latest_date
            )

            total = get_pixel_count(ee.Image.constant(1))

            result = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {"type": geom_type, "coordinates": original_coords},
                    "properties": {
                        "plot_name": plot_name,
                        "tile_url": tile_url,
                        "sensor_used": "Sentinel-2",
                        "image_count_in_range": s2_collection.size().getInfo(),
                        "latest_image_date": latest_date,
                        "analysis_dates": {
                            "start_date": start_date,
                            "end_date": end_date,
                            "latest_image_date": latest_date,
                            "previous_image_date": None,
                        },
                        "last_updated": datetime.utcnow().isoformat(),
                    },
                }],
                "pixel_summary": {
                    "total_pixel_count": total,

                    "deficient_pixel_count": get_pixel_count(combined.eq(1)),
                    "deficient_pixel_percentage": 0,
                    "deficient_pixel_coordinates": mask_to_coords(combined.eq(1)),

                    "less_pixel_count": get_pixel_count(combined.eq(2)),
                    "less_pixel_percentage": 0,
                    "less_pixel_coordinates": mask_to_coords(combined.eq(2)),

                    "adequat_pixel_count": get_pixel_count(combined.eq(3)),
                    "adequat_pixel_percentage": 0,
                    "adequat_pixel_coordinates": mask_to_coords(combined.eq(3)),

                    "excellent_pixel_count": get_pixel_count(combined.eq(4)),
                    "excellent_pixel_percentage": 0,
                    "excellent_pixel_coordinates": mask_to_coords(combined.eq(4)),

                    "excess_pixel_count": get_pixel_count(combined.eq(5)),
                    "excess_pixel_percentage": 0,
                    "excess_pixel_coordinates": mask_to_coords(combined.eq(5)),

                    "analysis_start_date": start_date,
                    "analysis_end_date": end_date,
                },
            }

            results.append(result)

        # =========================================================
        # SENTINEL-1 (VH) — only if no S2 when prefer_s2
        # =========================================================
        run_s1 = (not prefer_s2) or (len(results) == 0)

        if run_s1:
            s1_collection = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(geometry)
                .filterDate(analysis_start, analysis_end)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                .select("VH")
                .sort("system:time_start", False)
            )

            if s1_collection.size().getInfo() >= 2:

                images = s1_collection.toList(2)
                latest = ee.Image(images.get(0))
                prev = ee.Image(images.get(1))

                latest_date = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd").getInfo()
                prev_date = ee.Date(prev.get("system:time_start")).format("YYYY-MM-dd").getInfo()

                delta = latest.subtract(prev).clip(geometry)

                deficient = delta.lt(-3.13)
                less = delta.gte(-3.13).And(delta.lte(-0.1))
                adequat = delta.gt(-0.1).And(delta.lt(4))
                excellent = delta.gte(4).And(delta.lt(6))
                excess = delta.gte(6)

                combined = (
                    ee.Image.constant(0)
                    .where(deficient, 1)
                    .where(less, 2)
                    .where(adequat, 3)
                    .where(excellent, 4)
                    .where(excess, 5)
                    .clip(geometry)
                )

                tile_vis = combined.focal_mean(radius=7, units="meters").visualize(
                    min=0, max=5,
                    palette=["#EBFF34", "#CC8213AF", "#1348E88E", "#2E199ABD", "#0602178F"]
                )
                tile_url = resolve_tile_url(
                    tile_vis, geometry, plot_id, "water_uptake", latest_date
                )

                total = get_pixel_count(ee.Image.constant(1))

                result = {
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "geometry": {"type": geom_type, "coordinates": original_coords},
                        "properties": {
                            "plot_name": plot_name,
                            "tile_url": tile_url,
                            "sensor_used": "Sentinel-1",
                            "image_count_in_range": s1_collection.size().getInfo(),
                            "latest_image_date": latest_date,
                            "analysis_dates": {
                                "start_date": start_date,
                                "end_date": end_date,
                                "latest_image_date": latest_date,
                                "previous_image_date": prev_date,
                            },
                            "last_updated": datetime.utcnow().isoformat(),
                        },
                    }],
                    "pixel_summary": {
                        "total_pixel_count": total,
                        "analysis_start_date": start_date,
                        "analysis_end_date": end_date,
                    },
                }

                results.append(result)

        return pick_preferred_sensor_result(results, prefer_s2=prefer_s2) if results else None

    except Exception as e:
        print("❌ Water uptake failed:", e, flush=True)
        return None
# ==========================================================
# SOIL MOISTURE (FULL PIXEL STRUCTURE MATCHING YOUR SAMPLE)
# ==========================================================

def run_soil_moisture_analysis_by_plot(
    plot_name,
    plot_data,
    start_date,
    end_date,
    plot_id=None,
    prefer_s2=True,
):

    try:
        geometry = ee.Geometry(plot_data["geometry"])
        geom_type = plot_data["geom_type"]
        original_coords = plot_data["original_coords"]

        analysis_start = ee.Date(start_date)
        analysis_end = ee.Date(end_date)

        results = []

        # =====================================================
        # SENTINEL-2 (NDWI BASED) — preferred
        # =====================================================
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(geometry)
            .filterDate(analysis_start, analysis_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            .map(lambda img: img.clip(geometry))
        )

        s2_size = int(s2_collection.size().getInfo() or 0)

        if s2_size > 0:

            s2_composite = s2_collection.median().clip(geometry)

            ndwi = s2_composite.normalizedDifference(["B3", "B8"]).rename("NDWI")

            classified = (
                ee.Image.constant(0)
                .where(ndwi.lte(-0.4), 1)
                .where(ndwi.gt(-0.4).And(ndwi.lte(-0.3)), 2)
                .where(ndwi.gt(-0.3).And(ndwi.lte(0)), 3)
                .where(ndwi.gt(0).And(ndwi.lte(0.2)), 4)
                .where(ndwi.gt(0.2), 5)
                .rename("soil_class")
                .clip(geometry)
            )

            sensor_used = "Sentinel-2"

            latest_date = ee.Date(
                s2_collection.sort("system:time_start", False)
                .first()
                .get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()

            results.append(
                _build_soil_moisture_response(
                    plot_name, geom_type, original_coords,
                    classified, geometry,
                    start_date, end_date,
                    sensor_used, latest_date, s2_size,
                    plot_id=plot_id,
                )
            )

        # =====================================================
        # SENTINEL-1 (VV) — fallback when no S2
        # =====================================================
        run_s1 = (not prefer_s2) or (len(results) == 0)

        if run_s1:
            s1_collection = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(geometry)
                .filterDate(analysis_start, analysis_end)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .select(["VV"])
                .map(lambda img: img.clip(geometry))
            )

            s1_size = int(s1_collection.size().getInfo() or 0)

            if s1_size > 0:

                vv_composite = s1_collection.median().clip(geometry)
                vv_band = vv_composite.select("VV")

                classified = (
                    ee.Image.constant(0)
                    .where(vv_band.lte(-12), 1)
                    .where(vv_band.gt(-12).And(vv_band.lte(-10)), 2)
                    .where(vv_band.gt(-10).And(vv_band.lte(-8)), 3)
                    .where(vv_band.gt(-8).And(vv_band.lte(-6)), 4)
                    .where(vv_band.gt(-6), 5)
                    .rename("soil_class")
                    .clip(geometry)
                )

                sensor_used = "Sentinel-1"

                latest_date = ee.Date(
                    s1_collection.sort("system:time_start", False)
                    .first()
                    .get("system:time_start")
                ).format("YYYY-MM-dd").getInfo()

                results.append(
                    _build_soil_moisture_response(
                        plot_name, geom_type, original_coords,
                        classified, geometry,
                        start_date, end_date,
                        sensor_used, latest_date, s1_size,
                        plot_id=plot_id,
                    )
                )

        return pick_preferred_sensor_result(results, prefer_s2=prefer_s2) if results else None

    except Exception as e:
        print("❌ Soil moisture failed:", e, flush=True)
        return None

def _build_soil_moisture_response(
    plot_name, geom_type, original_coords,
    classified, geometry,
    start_date, end_date,
    sensor_used, latest_date, image_count,
    plot_id=None,
):

    palette = ["#2FC0D3", "#4365D4", "#473CDF", "#2116BF", "#000475"]
    vis_params = {"min": 1, "max": 5, "palette": palette}

    smoothed = classified.focal_mean(radius=20, units="meters")
    tile_vis = smoothed.visualize(**vis_params).clip(geometry)
    tile_url = resolve_tile_url(
        tile_vis, geometry, plot_id, "soil_moisture", latest_date
    )

    count_image = ee.Image.constant(1)

    def get_pixel_count(mask):
        return int(
            count_image.updateMask(mask)
            .reduceRegion(ee.Reducer.count(), geometry, 10, bestEffort=True)
            .get("constant")
            .getInfo() or 0
        )

    def mask_to_coords(mask):
        sampled = (
            mask.selfMask()
            .addBands(ee.Image.pixelLonLat())
            .sample(region=geometry, scale=10, geometries=True, tileScale=4)
            .getInfo()
        )
        coords = [f["geometry"]["coordinates"] for f in sampled.get("features", [])]
        return [list(x) for x in {tuple(c) for c in coords}]

    total_pixel_count = get_pixel_count(count_image)

    labels = {
        1: "less",
        2: "adequate",
        3: "excellent",
        4: "excess",
        5: "shallow_water",
    }

    pixel_summary = {
        "total_pixel_count": total_pixel_count,
        "start_date": start_date,
        "end_date": end_date,
        "latest_image_date": latest_date,
        "sensor_used": sensor_used,
    }

    for class_id in range(1, 6):
        mask = classified.eq(class_id)
        count = get_pixel_count(mask)
        coords = mask_to_coords(mask)

        pct = (count / total_pixel_count) * 100 if total_pixel_count else 0
        key = labels[class_id]

        pixel_summary[f"{key}_pixel_count"] = count
        pixel_summary[f"{key}_pixel_percentage"] = round(pct, 2)
        pixel_summary[f"{key}_pixel_coordinates"] = coords

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
            "tile_url": tile_url,
            "last_updated": datetime.utcnow().isoformat(),
        },
    }

    return {
        "type": "FeatureCollection",
        "features": [feature],
        "pixel_summary": pixel_summary,
    }
# ==========================================================
# PEST DETECTION (FIXED)
# ==========================================================

def run_pest_detection_analysis_by_plot(
    plot_name: str,
    plot_data: dict,
    start_date: str,
    end_date: str,
    plot_id=None,
    prefer_s2=True,
):
    try:

        # ==============================
        # SAFE INPUT VALIDATION
        # ==============================

        if not isinstance(plot_data, dict):
            print(f"❌ Invalid plot_data for {plot_name}")
            return None

        geometry = plot_data.get("geometry")
        geom_type = plot_data.get("geom_type")
        original_coords = plot_data.get("original_coords")

        if geometry is None or geom_type is None or original_coords is None:
            print(f"❌ Missing geometry for {plot_name}")
            return None

        # ✅ FIX: geometry already EE object sometimes
        if not isinstance(geometry, ee.Geometry):
            geometry = ee.Geometry(geometry)

        props = plot_data.get("properties") or {}

        crop_type = (
            props.get("crop_type_name")
            or props.get("crop_type")
            or props.get("crop")
            or ""
        )

        crop_type = str(crop_type).lower().strip()

        # Missing crop type: still run generic S1 pest so rebin dates are not empty
        if not crop_type:
            print(
                f"⚠ Crop type missing for {plot_name} → using generic pest detection",
                flush=True,
            )
            crop_type = "generic"

        # ==============================
        # DATE CREATION
        # ==============================

        if not start_date or not end_date:
            print(f"❌ Invalid date range for {plot_name}")
            return None

        analysis_start = ee.Date(start_date)
        analysis_end = ee.Date(end_date)

        end_dt = datetime.strptime(str(end_date)[:10], "%Y-%m-%d")
        baseline_year_py = end_dt.year - 1
        sensor_used = "Sentinel-1"
        one = ee.Image.constant(1)

        def count_masks(chewing_mask, fungi_mask, sucking_mask, wilt_mask):
            """One EE round-trip for all pest class pixel counts."""
            try:
                stacked = ee.Image.cat([
                    one.rename("total"),
                    one.updateMask(chewing_mask).rename("chewing"),
                    one.updateMask(fungi_mask).rename("fungi"),
                    one.updateMask(sucking_mask).rename("sucking"),
                    one.updateMask(wilt_mask).rename("wilt"),
                ])
                stats = stacked.reduceRegion(
                    ee.Reducer.count(),
                    geometry,
                    10,
                    bestEffort=True,
                )
                info = ee_get_info(stats, label=f"pest-counts:{plot_name}") or {}
                return {
                    "total": int(info.get("total") or 0),
                    "chewing": int(info.get("chewing") or 0),
                    "fungi": int(info.get("fungi") or 0),
                    "sucking": int(info.get("sucking") or 0),
                    "wilt": int(info.get("wilt") or 0),
                }
            except Exception as e:
                print(f"⚠ pest count failed for {plot_name}: {e}", flush=True)
                return {
                    "total": 0,
                    "chewing": 0,
                    "fungi": 0,
                    "sucking": 0,
                    "wilt": 0,
                }

        def build_pest_result(
            chewing_mask,
            fungi_mask,
            sucking_mask,
            wilt_mask,
            soilborne_mask,
            tile_image,
            analysis_image_date,
            baseline_start_str,
            baseline_end_str,
            grape_crop,
        ):
            try:
                if grape_crop:
                    tile_vis = tile_image.visualize(min=-25, max=0)
                else:
                    tile_vis = tile_image.visualize(min=0, max=1)
                tile_url = resolve_tile_url(
                    tile_vis, geometry, plot_id, "pest_detection", analysis_image_date
                )
            except Exception:
                tile_url = None

            counts = count_masks(
                chewing_mask, fungi_mask, sucking_mask, wilt_mask
            )
            # soilborne uses same mask as chewing in current models
            soilborne_pixel_count = counts["chewing"]

            total_pixel_count = counts["total"]
            chewing_pixel_count = counts["chewing"]
            fungi_pixel_count = counts["fungi"]
            sucking_pixel_count = counts["sucking"]
            wilt_pixel_count = counts["wilt"]

            healthy_pixel_count = max(
                total_pixel_count - (
                    chewing_pixel_count +
                    fungi_pixel_count +
                    sucking_pixel_count +
                    wilt_pixel_count +
                    soilborne_pixel_count
                ), 0
            )

            def percent(x):
                return (x / total_pixel_count) * 100 if total_pixel_count else 0

            analysis_dates = {
                "baseline_start_date": baseline_start_str,
                "baseline_end_date": baseline_end_str,
                "analysis_start_date": start_date,
                "analysis_end_date": end_date,
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
                    "sensor_used": sensor_used,
                    "image_count": 1,
                    "image_dates": [analysis_image_date],
                    "latest_image_date": analysis_image_date,
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
                    "fungi_affected_pixel_count": fungi_pixel_count,
                    "fungi_affected_pixel_percentage": percent(fungi_pixel_count),
                    "sucking_affected_pixel_count": sucking_pixel_count,
                    "sucking_affected_pixel_percentage": percent(sucking_pixel_count),
                    "wilt_affected_pixel_count": wilt_pixel_count,
                    "wilt_affected_pixel_percentage": percent(wilt_pixel_count),
                    "SoilBorn_pixel_count": soilborne_pixel_count,
                    "SoilBorn_affected_pixel_percentage": percent(soilborne_pixel_count),
                    "latest_image_date": analysis_image_date,
                    "image_dates": [analysis_image_date],
                    **analysis_dates,
                },
            }

        # ========================================================
        # GRAPES — one stored result per Sentinel-1 acquisition
        # ========================================================

        if "grape" in crop_type:

            baseline_start_str = f"{baseline_year_py}-04-01"
            baseline_end_str = f"{baseline_year_py}-06-30"
            baseline_start = ee.Date(baseline_start_str)
            baseline_end = ee.Date(baseline_end_str)

            def db_to_linear(img):
                return ee.Image(10).pow(img.divide(10)).add(1e-6)

            # Prefer DESCENDING, but fall back to all orbits so some plots
            # (ASCENDING-only coverage) still get rebin dates.
            s1_base = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .select(["VV", "VH"])
            )
            s1 = s1_base.filter(
                ee.Filter.eq("orbitProperties_pass", "DESCENDING")
            ).sort("system:time_start")

            image_count = int(ee_get_info(s1.size(), label=f"s1-size:{plot_name}") or 0)
            if image_count == 0:
                print(
                    f"⚠ No DESCENDING S1 for {plot_name} → trying all orbits",
                    flush=True,
                )
                s1 = s1_base.sort("system:time_start")
                image_count = int(
                    ee_get_info(s1.size(), label=f"s1-size-all:{plot_name}") or 0
                )

            if image_count == 0:
                print(f"⚠ No Sentinel-1 images for {plot_name}")
                return None

            base_col = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(geometry)
                .filterDate(baseline_start, baseline_end)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .select(["VV", "VH"])
            )
            base_desc = base_col.filter(
                ee.Filter.eq("orbitProperties_pass", "DESCENDING")
            )
            base_desc_n = int(
                ee_get_info(base_desc.size(), label=f"s1-base:{plot_name}") or 0
            )
            base = (
                (base_desc if base_desc_n > 0 else base_col)
                .median()
                .clip(geometry)
            )

            vv_base = db_to_linear(base.select("VV"))
            vh_base = db_to_linear(base.select("VH"))
            rvi_base = vh_base.multiply(4).divide(vv_base.add(vh_base))
            vhvv_base = vh_base.divide(vv_base)

            results = []
            image_list = s1.toList(image_count)

            for i in range(image_count):
                if i > 0:
                    sleep_between_windows(1.0)

                cur = ee.Image(image_list.get(i)).clip(geometry)
                analysis_image_date = ee_get_info(
                    ee.Date(cur.get("system:time_start")).format("YYYY-MM-dd"),
                    label=f"s1-date:{plot_name}",
                )

                vv_cur = db_to_linear(cur.select("VV"))
                vh_cur = db_to_linear(cur.select("VH"))
                rvi_cur = vh_cur.multiply(4).divide(vv_cur.add(vh_cur))
                vhvv_cur = vh_cur.divide(vv_cur)

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

                results.append(
                    build_pest_result(
                        chewing_mask,
                        fungi_mask,
                        sucking_mask,
                        ee.Image.constant(0),
                        chewing_mask,
                        cur.select("VV"),
                        analysis_image_date,
                        baseline_start_str,
                        baseline_end_str,
                        grape_crop=True,
                    )
                )

            return results

        # ========================================================
        # SUGARCANE / GENERIC — one result per Sentinel-1 acquisition
        # Unsupported crops fall through here so rebin is not empty.
        # ========================================================

        else:
            if "sugar" not in crop_type and crop_type != "generic":
                print(
                    f"⚠ Crop '{crop_type}' has no dedicated pest model "
                    f"for {plot_name} → using generic S1 pest",
                    flush=True,
                )

            baseline_start_str = f"{baseline_year_py}-06-01"
            baseline_end_str = f"{baseline_year_py}-10-30"
            baseline_start = ee.Date(baseline_start_str)
            baseline_end = ee.Date(baseline_end_str)

            s1 = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains(
                    "transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains(
                    "transmitterReceiverPolarisation", "VH"))
                .sort("system:time_start")
            )

            image_count = int(
                ee_get_info(s1.size(), label=f"s1-size:{plot_name}") or 0
            )

            if image_count == 0:
                print(f"⚠ No Sentinel-1 images for {plot_name}")
                return None

            results = []
            image_list = s1.toList(image_count)

            for i in range(image_count):
                if i > 0:
                    sleep_between_windows(1.0)

                img = ee.Image(image_list.get(i)).clip(geometry)
                analysis_image_date = ee_get_info(
                    ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"),
                    label=f"s1-date:{plot_name}",
                )

                vv = img.select("VV")
                vh = img.select("VH")
                ratio = vv.divide(vh.add(1e-6)).rename("VV_VH")
                sar_composite = vv.addBands(vh).addBands(ratio)
                sar_mean = sar_composite.reduce(ee.Reducer.mean()).rename("sar_fc")
                stretched = sar_mean.unitScale(-25, 5).clamp(0, 1)

                chewing_mask = stretched.lte(0.01)
                fungi_mask = stretched.lte(0.03)
                sucking_mask = stretched.gte(0.85)
                wilt_mask = stretched.gte(0.9)

                results.append(
                    build_pest_result(
                        chewing_mask,
                        fungi_mask,
                        sucking_mask,
                        wilt_mask,
                        chewing_mask,
                        stretched,
                        analysis_image_date,
                        baseline_start_str,
                        baseline_end_str,
                        grape_crop=False,
                    )
                )

            return results

    except Exception as e:
        print(f"❌ Pest detection failed for {plot_name}: {e}", flush=True)
        return None

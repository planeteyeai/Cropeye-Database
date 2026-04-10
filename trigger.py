from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import threading
import time
from datetime import date, timedelta

from fastapi.middleware.cors import CORSMiddleware

from worker import task_queue, worker, daily_scheduler, plot_sync_service, plot_dict, run_query
from gee_growth import (
    run_growth_analysis_by_plot,
    run_water_uptake_analysis_by_plot,
    run_soil_moisture_analysis_by_plot,
    run_pest_detection_analysis_by_plot,
)
from psycopg2.extras import Json

# =====================================================
# FASTAPI
# =====================================================

@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=daily_scheduler, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)

# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# HELPERS
# =====================================================

def extract_result(geojson_list):
    """Pull the first valid GeoJSON result from a list or dict."""
    if isinstance(geojson_list, dict):
        geojson_list = [geojson_list]
    for item in geojson_list or []:
        if item and item.get("features"):
            return item
    return None

def run_fresh_analysis(plot_name, plot_data):
    """Run all 4 analyses fresh and return results dict."""
    end = date.today()
    start = (end - timedelta(days=30)).isoformat()
    end = end.isoformat()

    results = {}
    for name, fn in [
        ("growth", run_growth_analysis_by_plot),
        ("water", run_water_uptake_analysis_by_plot),
        ("soil", run_soil_moisture_analysis_by_plot),
        ("pest", run_pest_detection_analysis_by_plot),
    ]:
        try:
            raw = fn(plot_name, plot_data, start, end)
            results[name] = extract_result(raw)
        except Exception as e:
            print(f"⚠ {name} analysis failed: {e}", flush=True)
            results[name] = None

    return results

# =====================================================
# TRIGGER NEW PLOT
# =====================================================

@app.post("/trigger-new-plot")
async def trigger_new(request: Request):

    global plot_dict

    try:
        body = await request.json()
    except Exception:
        return {"status": "error", "message": "Invalid JSON"}

    plot_name = body.get("plot_name")

    if not plot_name:
        return {"status": "error", "message": "plot_name required"}

    print(f"🚀 Trigger received: {plot_name}", flush=True)

    # Force fresh fetch from Django
    plot_dict.clear()
    plot_dict.update(plot_sync_service.get_plots_dict(force_refresh=True))

    if plot_name not in plot_dict:
        return {"status": "error", "message": "Plot not found"}

    plot_data = plot_dict[plot_name]
    geom = plot_data.get("geometry")

    if not geom:
        return {"status": "error", "message": "No geometry"}

    try:
        geom_geojson = geom.getInfo()
        area_ha = float(geom.area().divide(10000).getInfo())
    except Exception as e:
        return {"status": "error", "message": f"Invalid geometry: {str(e)}"}

    props = plot_data.get("properties", {})

    # Upsert plot into DB
    run_query(
        """
        INSERT INTO plots 
        (plot_name, geojson, area_hectares, django_plot_id, plantation_date, crop_type)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (plot_name)
        DO UPDATE SET
            geojson = EXCLUDED.geojson,
            area_hectares = EXCLUDED.area_hectares,
            django_plot_id = EXCLUDED.django_plot_id,
            plantation_date = EXCLUDED.plantation_date,
            crop_type = EXCLUDED.crop_type
        """,
        (
            plot_name,
            Json(geom_geojson),
            area_ha,
            props.get("django_id"),
            props.get("plantation_date"),
            props.get("crop_type_name"),
        )
    )

    # Run fresh analysis synchronously so we return new data
    print(f"🔎 Running fresh analysis for: {plot_name}", flush=True)
    analysis_results = run_fresh_analysis(plot_name, plot_data)

    # Persist fresh results to DB
    plot_row = run_query(
        "SELECT id FROM plots WHERE plot_name=%s",
        (plot_name,),
        fetchone=True
    )
    plot_id = plot_row["id"] if plot_row else None

    if plot_id:
        end_str = date.today().isoformat()
        for analysis_type, geojson in analysis_results.items():
            if not geojson:
                continue
            try:
                feat_props = geojson["features"][0]["properties"]
                analysis_date = (
                    feat_props.get("analysis_image_date")
                    or feat_props.get("latest_image_date")
                    or end_str
                )
                sensor = feat_props.get("sensor_used", "unknown")
                tile_url = feat_props.get("tile_url")
                final_type = f"{analysis_type}_{sensor.lower()}"

                run_query(
                    """
                    INSERT INTO analysis_results
                    (plot_id, analysis_type, analysis_date, response_json, tile_url, sensor_used)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (plot_id, analysis_type, analysis_date)
                    DO UPDATE SET
                        response_json = EXCLUDED.response_json,
                        tile_url = EXCLUDED.tile_url,
                        sensor_used = EXCLUDED.sensor_used
                    """,
                    (plot_id, final_type, analysis_date, Json(geojson), tile_url, sensor)
                )
            except Exception as e:
                print(f"⚠ Store failed for {analysis_type}: {e}", flush=True)

    print(f"✅ Fresh analysis done for: {plot_name}", flush=True)

    return {
        "status": "success",
        "plot": plot_name,
        "results": analysis_results,
    }

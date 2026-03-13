from datetime import date, timedelta
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import ee
from flask import Flask, request, jsonify
from psycopg2.extras import RealDictCursor, Json

from gee_growth import (
    run_growth_analysis_by_plot,
    run_water_uptake_analysis_by_plot,
    run_soil_moisture_analysis_by_plot,
    run_pest_detection_analysis_by_plot
)

from shared_services import PlotSyncService, run_plot_sync
from db import get_connection
from Admin import run_monthly_backfill_for_plot


# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)

# =====================================================
# CONFIG
# =====================================================

CHECK_INTERVAL = 20
MAX_PARALLEL_ANALYSIS = 4


# =====================================================
# INITIAL PLOT SYNC
# =====================================================

print("🔄 Running plot sync...", flush=True)

try:
    sync_result = run_plot_sync()
    print("✅ Sync result:", sync_result, flush=True)
except Exception as e:
    print("❌ Plot sync failed:", str(e), flush=True)

plot_service = PlotSyncService()
plots = plot_service.get_plots_dict(force_refresh=True)

known_plots = set(plots.keys())


# =====================================================
# DB HELPER
# =====================================================

def run_query(query, params=None, fetchone=False, fetchall=False):

    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:

        if params:
            new_params = []
            for p in params:
                if isinstance(p, dict):
                    new_params.append(Json(p))
                else:
                    new_params.append(p)
            params = tuple(new_params)

        cursor.execute(query, params)

        if fetchone:
            result = cursor.fetchone()
        elif fetchall:
            result = cursor.fetchall()
        else:
            result = None

        conn.commit()
        return result

    except Exception as e:
        conn.rollback()
        print("🔥 DB error:", e, flush=True)
        return None

    finally:
        cursor.close()
        conn.close()


# =====================================================
# STORE RESULTS
# =====================================================

def store_results(results, analysis_type, plot_id):

    if not results:
        return

    if isinstance(results, dict):
        results = [results]

    for geojson in results:

        features = geojson.get("features")
        if not features:
            continue

        props = features[0].get("properties", {})

        analysis_date = (
            props.get("analysis_image_date")
            or props.get("latest_image_date")
        )

        sensor_used = props.get("sensor", "Unknown")
        tile_url = props.get("tile_url")

        if not analysis_date:
            continue

        run_query(
            """
            INSERT INTO satellite_images
            (plot_id,satellite,satellite_date)
            VALUES (%s,%s,%s)
            ON CONFLICT DO NOTHING
            """,
            (plot_id, sensor_used, analysis_date)
        )

        run_query(
            """
            INSERT INTO analysis_results
            (plot_id,analysis_type,analysis_date,sensor_used,tile_url,response_json)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (plot_id,analysis_type,analysis_date) DO NOTHING
            """,
            (plot_id, analysis_type, analysis_date, sensor_used, tile_url, Json(geojson))
        )

        print(f"✅ Stored {analysis_type} {analysis_date}", flush=True)


# =====================================================
# TODAY ANALYSIS (PARALLEL)
# =====================================================

def run_today_analysis_for_plot(plot_name, plot_data, plot_id):

    print(f"🌅 Running TODAY analysis for {plot_name}", flush=True)

    end_date = date.today()
    start_date = (end_date - timedelta(days=30)).isoformat()
    end_date = end_date.isoformat()

    def run_growth():
        try:
            res = run_growth_analysis_by_plot(plot_name, plot_data, start_date, end_date)
            store_results(res, "growth", plot_id)
        except Exception as e:
            print(f"🔥 growth failed {plot_name}: {e}")

    def run_water():
        try:
            res = run_water_uptake_analysis_by_plot(plot_name, plot_data, start_date, end_date)
            store_results(res, "water_uptake", plot_id)
        except Exception as e:
            print(f"🔥 water uptake failed {plot_name}: {e}")

    def run_soil():
        try:
            res = run_soil_moisture_analysis_by_plot(plot_name, plot_data, start_date, end_date)
            store_results(res, "soil_moisture", plot_id)
        except Exception as e:
            print(f"🔥 soil moisture failed {plot_name}: {e}")

    def run_pest():
        try:
            res = run_pest_detection_analysis_by_plot(plot_name, plot_data, start_date, end_date)
            store_results(res, "pest_detection", plot_id)
        except Exception as e:
            print(f"🔥 pest detection failed {plot_name}: {e}")

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_ANALYSIS) as executor:

        executor.submit(run_growth)
        executor.submit(run_water)
        executor.submit(run_soil)
        executor.submit(run_pest)

    print(f"✅ TODAY analysis complete {plot_name}", flush=True)


# =====================================================
# NEW PLOT PIPELINE
# =====================================================

def process_new_plot(plot_name):

    print(f"🚀 Processing new plot {plot_name}", flush=True)

    plots = plot_service.get_plots_dict(force_refresh=True)
    plot_data = plots.get(plot_name)

    if not plot_data:
        print("⚠ Plot not found after sync")
        return

    row = run_query(
        "SELECT id FROM plots WHERE plot_name=%s",
        (plot_name,),
        fetchone=True
    )

    if not row:
        print("⚠ Plot not in DB yet")
        return

    plot_id = row["id"]

    run_today_analysis_for_plot(plot_name, plot_data, plot_id)

    run_monthly_backfill_for_plot(plot_name, plot_data)


# =====================================================
# MANUAL TRIGGER API
# =====================================================

@app.route("/trigger-new-plot", methods=["POST"])
def trigger_new_plot():

    data = request.json
    plot_name = data.get("plot_name")

    if not plot_name:
        return jsonify({"error": "plot_name missing"}), 400

    print(f"🚀 Manual trigger received for {plot_name}")

    threading.Thread(
        target=process_new_plot,
        args=(plot_name,)
    ).start()

    return jsonify({"status": "triggered"})


# =====================================================
# WORKER LOOP
# =====================================================

def worker_loop():

    global known_plots

    print("🚀 Worker loop started")

    while True:

        try:

            print("🔎 Checking for new plots...", flush=True)

            run_plot_sync()

            plots = plot_service.get_plots_dict(force_refresh=True)

            current_plots = set(plots.keys())

            new_plots = current_plots - known_plots

            if new_plots:

                print(f"🆕 New plots detected: {new_plots}", flush=True)

                for plot_name in new_plots:

                    threading.Thread(
                        target=process_new_plot,
                        args=(plot_name,)
                    ).start()

            known_plots = current_plots

            time.sleep(CHECK_INTERVAL)

        except Exception as e:

            print("🔥 Worker error:", e)

            time.sleep(30)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    threading.Thread(target=worker_loop).start()

    print("🌐 Worker API running on port 8000")

    app.run(host="0.0.0.0", port=8000)

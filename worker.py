from datetime import date, timedelta
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify
from psycopg2.extras import RealDictCursor, Json

from gee_growth import (
    run_growth_analysis_by_plot,
    run_water_uptake_analysis_by_plot,
    run_soil_moisture_analysis_by_plot,
    run_pest_detection_analysis_by_plot
)

from db import get_connection
from Admin import run_monthly_backfill_for_plot
from shared_services import run_plot_sync  # ✅ IMPORTANT

# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)

# =====================================================
# CONFIG
# =====================================================

MAX_PARALLEL_ANALYSIS = 4

known_plot_ids = set()

# =====================================================
# DB HELPER
# =====================================================

def run_query(query, params=None, fetchone=False, fetchall=False):

    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        if params:
            params = tuple(Json(p) if isinstance(p, dict) else p for p in params)

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
# INITIAL LOAD
# =====================================================

print("🔄 Running initial sync...", flush=True)

try:
    run_plot_sync()  # ✅ ensure DB is populated
except Exception as e:
    print("❌ Initial sync failed:", e)

print("🧠 Loading existing plots from DB...", flush=True)

rows = run_query("SELECT id FROM plots", fetchall=True) or []

for r in rows:
    known_plot_ids.add(r["id"])

print(f"✅ Loaded {len(known_plot_ids)} plots into memory", flush=True)

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
# TODAY ANALYSIS
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
# PROCESS NEW PLOT
# =====================================================

def process_new_plot(plot_name):

    print(f"🚀 Processing new plot {plot_name}", flush=True)

    row = run_query(
        "SELECT id, geojson FROM plots WHERE plot_name=%s",
        (plot_name,),
        fetchone=True
    )

    if not row:
        print("⚠ Plot not found in DB")
        return

    if not row["geojson"]:
        print(f"⚠ Plot {plot_name} has no geojson → skipping")
        return

    plot_id = row["id"]

    plot_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": row["geojson"],
                "properties": {}
            }
        ]
    }

    run_today_analysis_for_plot(plot_name, plot_data, plot_id)

    threading.Thread(
        target=run_monthly_backfill_for_plot,
        args=(plot_name, plot_data)
    ).start()

# =====================================================
# SMART REFRESH
# =====================================================

@app.route("/refresh-from-django", methods=["POST"])
def refresh_from_django():

    global known_plot_ids

    try:
        print("🔄 Smart refresh triggered...", flush=True)

        # ✅ CRITICAL: Sync first
        run_plot_sync()

        rows = run_query(
            "SELECT id, plot_name FROM plots",
            fetchall=True
        ) or []

        current_ids = set()
        plot_map = {}

        for r in rows:
            current_ids.add(r["id"])
            plot_map[r["id"]] = r["plot_name"]

        new_plots = current_ids - known_plot_ids

        print(f"🆕 New plots detected: {len(new_plots)}", flush=True)

        triggered = []

        for pid in new_plots:
            plot_name = plot_map[pid]

            print(f"🚀 Triggering {plot_name}", flush=True)

            threading.Thread(
                target=process_new_plot,
                args=(plot_name,)
            ).start()

            triggered.append(plot_name)

        known_plot_ids = current_ids

        return jsonify({
            "status": "success",
            "new_detected": len(new_plots),
            "triggered": triggered
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =====================================================
# AUTO REFRESH LOOP (FIXED STARTUP)
# =====================================================

def auto_refresh_loop():

    while True:
        try:
            print("⏱ Auto refresh calling...", flush=True)
            requests.post("http://127.0.0.1:8000/refresh-from-django", timeout=60)
        except Exception as e:
            print("⚠ Auto refresh failed:", e)

        time.sleep(5)

def start_background_jobs():
    time.sleep(2)  # ✅ wait for Flask to start
    auto_refresh_loop()

# =====================================================
# MANUAL TRIGGER
# =====================================================

@app.route("/trigger-new-plot", methods=["POST"])
def trigger_new_plot():

    data = request.json
    plot_name = data.get("plot_name")

    if not plot_name:
        return jsonify({"error": "plot_name missing"}), 400

    threading.Thread(
        target=process_new_plot,
        args=(plot_name,)
    ).start()

    return jsonify({"status": "triggered"})

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("🌐 Worker API running on port 8000")

    threading.Thread(target=start_background_jobs).start()

    app.run(host="0.0.0.0", port=8000)

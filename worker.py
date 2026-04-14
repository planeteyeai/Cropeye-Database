from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from datetime import date, timedelta
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue

from psycopg2.extras import RealDictCursor, Json

from gee_growth import (
    run_growth_analysis_by_plot,
    run_water_uptake_analysis_by_plot,
    run_soil_moisture_analysis_by_plot,
    run_pest_detection_analysis_by_plot
)

from db import get_connection
from Admin import run_monthly_backfill_for_plot
from shared_services import PlotSyncService

# =====================================================
# INIT
# =====================================================

plot_sync_service = PlotSyncService()
plot_dict = {}

# priority, timestamp, item
task_queue = PriorityQueue()

MAX_PARALLEL_ANALYSIS = 4
GLOBAL_LIMIT = 4

semaphore = threading.Semaphore(GLOBAL_LIMIT)

# =====================================================
# DB
# =====================================================

def run_query(query, params=None, fetchone=False, fetchall=False):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        if params:
            converted = []
            for p in params:
                if isinstance(p, dict):
                    converted.append(Json(p))
                else:
                    converted.append(p)
            params = tuple(converted)

        cursor.execute(query, params)

        result = None
        if fetchone:
            result = cursor.fetchone()
        elif fetchall:
            result = cursor.fetchall()

        conn.commit()
        return result

    except Exception as e:
        conn.rollback()
        print(f"🔥 DB error: {e}", flush=True)
        return None

    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass

# =====================================================
# STORE RESULTS
# =====================================================

def extract_metadata(geojson):
    try:
        props = geojson["features"][0]["properties"]
        return props.get("tile_url"), props.get("sensor_used")
    except Exception:
        return None, None


def store_results(results, analysis_type, plot_id):
    if not results:
        return

    if isinstance(results, dict):
        results = [results]

    for geojson in results:
        try:
            features = geojson.get("features", [])
            if not features:
                continue

            props = features[0].get("properties", {})

            analysis_date = (
                props.get("analysis_image_date")
                or props.get("latest_image_date")
                or props.get("analysis_dates", {}).get("latest_image_date")
            )

            if not analysis_date:
                print(f"⚠ Missing analysis date for {analysis_type}", flush=True)
                continue

            sensor = str(props.get("sensor_used") or "unknown").lower()
            final_type = f"{analysis_type}_{sensor}"

            tile_url, sensor_used = extract_metadata(geojson)

            run_query(
                """
                INSERT INTO analysis_results
                (
                    plot_id,
                    analysis_type,
                    analysis_date,
                    response_json,
                    tile_url,
                    sensor_used
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (plot_id, analysis_type, analysis_date)
                DO UPDATE SET
                    response_json = EXCLUDED.response_json,
                    tile_url = EXCLUDED.tile_url,
                    sensor_used = EXCLUDED.sensor_used
                """,
                (
                    plot_id,
                    final_type,
                    analysis_date,
                    Json(geojson),
                    tile_url,
                    sensor_used
                )
            )

            print(
                f"✅ Stored {final_type} for plot_id={plot_id} date={analysis_date}",
                flush=True
            )

        except Exception as e:
            print(f"🔥 store_results failed ({analysis_type}): {e}", flush=True)

# =====================================================
# ANALYSIS
# =====================================================

def safe_analysis(fn, analysis_name, plot_name, plot_data, start_date, end_date, plot_id):
    try:
        print(f"🔎 Running {analysis_name} for {plot_name}", flush=True)

        result = fn(
            plot_name,
            plot_data,
            start_date,
            end_date
        )

        if result:
            store_results(result, analysis_name, plot_id)
        else:
            print(f"⚠ No result for {analysis_name} on {plot_name}", flush=True)

    except Exception as e:
        print(f"🔥 {analysis_name} failed for {plot_name}: {e}", flush=True)


def run_today_analysis(plot_name, plot_data, plot_id):
    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    with semaphore:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_ANALYSIS) as executor:
            futures = []

            futures.append(
                executor.submit(
                    safe_analysis,
                    run_growth_analysis_by_plot,
                    "growth",
                    plot_name,
                    plot_data,
                    start_str,
                    end_str,
                    plot_id
                )
            )

            futures.append(
                executor.submit(
                    safe_analysis,
                    run_water_uptake_analysis_by_plot,
                    "water",
                    plot_name,
                    plot_data,
                    start_str,
                    end_str,
                    plot_id
                )
            )

            futures.append(
                executor.submit(
                    safe_analysis,
                    run_soil_moisture_analysis_by_plot,
                    "soil",
                    plot_name,
                    plot_data,
                    start_str,
                    end_str,
                    plot_id
                )
            )

            futures.append(
                executor.submit(
                    safe_analysis,
                    run_pest_detection_analysis_by_plot,
                    "pest",
                    plot_name,
                    plot_data,
                    start_str,
                    end_str,
                    plot_id
                )
            )

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"🔥 Future error: {e}", flush=True)

# =====================================================
# PROCESS SINGLE PLOT
# =====================================================

def process_plot(plot_name):
    global plot_dict

    try:
        print(f"🚀 Processing plot: {plot_name}", flush=True)

        if plot_name not in plot_dict:
            print(f"❌ Plot not found in memory: {plot_name}", flush=True)
            return

        plot_data = plot_dict[plot_name]
        geometry = plot_data.get("geometry")

        if geometry is None:
            print(f"❌ Geometry missing for {plot_name}", flush=True)
            return

        props = plot_data.get("properties", {})

        try:
            geometry_geojson = geometry.getInfo()
            area_hectares = float(geometry.area().divide(10000).getInfo())
        except Exception as e:
            print(f"🔥 Geometry conversion failed for {plot_name}: {e}", flush=True)
            return

        crop_type = (
            props.get("crop_type_name")
            or props.get("crop_type")
            or props.get("crop")
        )

        django_plot_id = (
            props.get("django_id")
            or props.get("plot_id")
            or props.get("id")
        )

        plantation_date = props.get("plantation_date")

        # Save / update plot immediately
        run_query(
            """
            INSERT INTO plots
            (
                plot_name,
                geojson,
                area_hectares,
                django_plot_id,
                plantation_date,
                crop_type,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (plot_name)
            DO UPDATE SET
                geojson = EXCLUDED.geojson,
                area_hectares = EXCLUDED.area_hectares,
                django_plot_id = EXCLUDED.django_plot_id,
                plantation_date = EXCLUDED.plantation_date,
                crop_type = EXCLUDED.crop_type,
                updated_at = NOW()
            """,
            (
                plot_name,
                Json(geometry_geojson),
                area_hectares,
                django_plot_id,
                plantation_date,
                crop_type
            )
        )

        row = run_query(
            """
            SELECT id
            FROM plots
            WHERE plot_name = %s
            """,
            (plot_name,),
            fetchone=True
        )

        if not row:
            print(f"❌ Could not fetch plot_id for {plot_name}", flush=True)
            return

        plot_id = row["id"]

        print(
            f"✅ Plot stored: {plot_name} | plot_id={plot_id} | crop={crop_type}",
            flush=True
        )

        run_today_analysis(plot_name, plot_data, plot_id)

    except Exception as e:
        print(f"🔥 process_plot failed for {plot_name}: {e}", flush=True)

# =====================================================
# WORKER
# =====================================================

def worker():
    while True:
        try:
            priority, timestamp, item = task_queue.get()

            try:
                if isinstance(item, str) and item.startswith("backfill::"):
                    plot_name = item.split("backfill::", 1)[1]

                    print(f"🧠 Running backfill for {plot_name}", flush=True)

                    if plot_name in plot_dict:
                        run_monthly_backfill_for_plot(
                            plot_name,
                            plot_dict[plot_name]
                        )
                    else:
                        print(f"⚠ Backfill skipped, plot missing: {plot_name}", flush=True)

                else:
                    print(
                        f"⚙️ Worker picked priority={priority} plot={item}",
                        flush=True
                    )
                    process_plot(item)

            except Exception as e:
                print(f"🔥 Worker inner error: {e}", flush=True)

            finally:
                task_queue.task_done()

        except Exception as e:
            print(f"🔥 Worker outer error: {e}", flush=True)
            time.sleep(1)

# =====================================================
# FASTAPI LIFESPAN
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting worker thread...", flush=True)

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    yield

    print("🛑 Shutting down...", flush=True)

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Worker server running"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "queue_size": task_queue.qsize()
    }

# =====================================================
# MANUAL TRIGGER
# =====================================================

@app.options("/trigger-new-plot")
async def trigger_options():
    return {"status": "ok"}


@app.post("/trigger-new-plot")
async def trigger_new_plot():
    global plot_dict

    print("🚀 Manual trigger called", flush=True)

    try:
        new_data = plot_sync_service.get_plots_dict(force_refresh=True)

        if not new_data:
            return {
                "status": "error",
                "message": "No plot data returned"
            }

        plot_dict.clear()
        plot_dict.update(new_data)

        queued = []
        skipped = []

        for plot_name, plot_data in new_data.items():
            geometry = plot_data.get("geometry")

            if geometry is None:
                skipped.append({
                    "plot": plot_name,
                    "reason": "missing_geometry"
                })
                continue

            exists = run_query(
                "SELECT id FROM plots WHERE plot_name = %s",
                (plot_name,),
                fetchone=True
            )

            if exists:
                skipped.append({
                    "plot": plot_name,
                    "reason": "already_exists"
                })
                continue

            # Highest priority for new plot
            task_queue.put((0, time.time(), plot_name))

            # Slightly lower priority for backfill
            task_queue.put((1, time.time(), f"backfill::{plot_name}"))

            queued.append(plot_name)

            print(f"🆕 Queued new plot: {plot_name}", flush=True)

        return {
            "status": "queued",
            "queued_count": len(queued),
            "queued_plots": queued,
            "skipped_count": len(skipped),
            "skipped": skipped,
            "queue_size": task_queue.qsize()
        }

    except Exception as e:
        print(f"🔥 trigger_new_plot failed: {e}", flush=True)

        return {
            "status": "error",
            "message": str(e)
        }

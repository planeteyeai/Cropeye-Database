from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import threading
import time

from fastapi.middleware.cors import CORSMiddleware

from worker import task_queue, worker, daily_scheduler, plot_sync_service, plot_dict, run_query
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
# TRIGGER NEW PLOT (FINAL)
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

    # ✅ FORCE REFRESH FROM DJANGO
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

    # ✅ STORE FULL DATA (MATCH YOUR TABLE)
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

    # ✅ PRIORITY QUEUE
    task_queue.put((0, time.time(), plot_name))

    print(f"🚨 PRIORITY TASK ADDED: {plot_name}", flush=True)

    return {"status": "queued", "plot": plot_name}

from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import threading
import time

from fastapi.middleware.cors import CORSMiddleware

# 🔥 IMPORT SHARED QUEUE + WORKER
from plot_worker import task_queue, worker, daily_scheduler

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
# API
# =====================================================

@app.post("/trigger-new-plot")
async def trigger_new(request: Request):

    body = await request.json()
    plot_name = body.get("plot_name")

    if not plot_name:
        return {"status": "error", "message": "plot_name required"}

    # 🔥 HIGHEST PRIORITY
    task_queue.put((0, time.time(), plot_name))

    print(f"🚨 PRIORITY TASK ADDED: {plot_name}", flush=True)

    return {"status": "queued with priority", "plot": plot_name}

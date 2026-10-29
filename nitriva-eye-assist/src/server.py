"""
Nitriva FastAPI Application Server.
Provides real-time WebSockets telemetry, MJPEG video streaming for Optics Sensor,
REST endpoints for calibration, NLP speech generation, and smart device control.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.tracker import tracker
from src.calibrator import calibrator, CALIBRATION_TARGETS
from src.nlp_engine import nlp_engine

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# State
active_devices = {
    "light": False,
    "fan": False,
    "ac": False,
    "nurse": False
}

sos_state = {
    "active": False,
    "timestamp": 0.0
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch camera tracking loop
    tracker.start_camera()
    print("[Nitriva Server] Eye tracking ready.")
    try:
        yield
    finally:
        tracker.stop_camera()
        print("[Nitriva Server] Camera loop stopped.")


app = FastAPI(title="Nitriva Assistive Eye-Tracking System", lifespan=lifespan)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Pydantic Schemas
class LoginRequest(BaseModel):
    key: str

class CalibratePointRequest(BaseModel):
    target_id: str
    samples_count: int = 10

class SpeakRequest(BaseModel):
    keywords: List[str]

class DeviceToggleRequest(BaseModel):
    device_id: str

class SimulateRequest(BaseModel):
    action: Optional[str] = None
    gaze_x: Optional[float] = None
    gaze_y: Optional[float] = None


@app.get("/healthz")
async def health():
    """Render readiness probe, including successful model initialization."""
    ready = tracker.detector is not None
    return JSONResponse(
        {"status": "ok" if ready else "unavailable", "model_ready": ready},
        status_code=200 if ready else 503,
    )


@app.get("/")
async def root():
    """Serve main Nitriva application."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_file)


@app.get("/video_feed")
def video_feed():
    """MJPEG streaming endpoint for Optics Sensor HUD."""
    return StreamingResponse(
        tracker.generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Real-time bi-directional telemetry connection supporting local camera and cloud browser webcam modes."""
    await websocket.accept()

    async def receive_loop():
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if "bytes" in msg and msg["bytes"]:
                    data = await run_in_threadpool(tracker.process_client_frame, msg["bytes"])
                    data["frame_processed"] = True
                    data["devices"] = active_devices
                    data["sos_active"] = sos_state["active"]
                    await websocket.send_json(data)
                elif "text" in msg and msg["text"]:
                    try:
                        parsed = json.loads(msg["text"])
                        if isinstance(parsed, dict) and parsed.get("type") == "frame":
                            data = await run_in_threadpool(tracker.process_client_frame, parsed.get("data", ""))
                            data["frame_processed"] = True
                            data["devices"] = active_devices
                            data["sos_active"] = sos_state["active"]
                            await websocket.send_json(data)
                    except Exception:
                        pass
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[WebSocket Rx] Error: {e}")

    async def broadcast_loop():
        try:
            while True:
                # If local hardware camera is active, broadcast continuous telemetry at ~30 FPS
                if not tracker.simulation_mode:
                    data = tracker.get_telemetry()
                    data["devices"] = active_devices
                    data["sos_active"] = sos_state["active"]
                    await websocket.send_json(data)
                await asyncio.sleep(0.033)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[WebSocket Tx] Error: {e}")

    rx_task = asyncio.create_task(receive_loop())
    tx_task = asyncio.create_task(broadcast_loop())

    done, pending = await asyncio.wait(
        [rx_task, tx_task],
        return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


@app.post("/api/process_frame")
async def api_process_frame(req: Request):
    """Fallback HTTP endpoint for processing a single client browser video frame."""
    body = await req.body()
    data = await run_in_threadpool(tracker.process_client_frame, body)
    data["devices"] = active_devices
    data["sos_active"] = sos_state["active"]
    return data


@app.post("/api/login")
async def api_login(req: LoginRequest):
    """Validate patient access key (Default demo: 1234)."""
    if req.key.strip() in ["1234", "demo", "nitriva", "admin"]:
        return {"status": "success", "message": "Access granted."}
    raise HTTPException(status_code=401, detail="Invalid Patient Access Key. Use 1234 for demo.")


@app.get("/api/calibrate/targets")
async def get_calibration_targets():
    """Retrieve 5 calibration anchor positions."""
    return {"targets": CALIBRATION_TARGETS}


@app.post("/api/calibrate/point")
async def calibrate_point(req: CalibratePointRequest):
    """Record current gaze observation for a calibration point."""
    for _ in range(req.samples_count):
        calibrator.add_sample(req.target_id, tracker.raw_gx, tracker.raw_gy)
    return {"status": "recorded", "target_id": req.target_id}


@app.post("/api/calibrate/finish")
async def calibrate_finish():
    """Finalize calibration model."""
    success = calibrator.compute_calibration()
    return {"status": "calibrated" if success else "failed", "is_calibrated": calibrator.is_calibrated}


@app.post("/api/speak")
async def api_speak(req: SpeakRequest):
    """Synthesize contextual sentence from selected keywords."""
    sentence = nlp_engine.generate_sentence(req.keywords)
    return {
        "keywords": req.keywords,
        "sentence": sentence
    }


@app.get("/api/devices")
async def get_devices():
    """Fetch smart appliance states."""
    return active_devices


@app.post("/api/devices/toggle")
async def toggle_device(req: DeviceToggleRequest):
    """Toggle smart appliance on/off."""
    dev = req.device_id.lower()
    if dev in active_devices:
        active_devices[dev] = not active_devices[dev]
        return {"device": dev, "state": active_devices[dev]}
    raise HTTPException(status_code=404, detail="Unknown device")


@app.post("/api/sos")
async def trigger_sos():
    """Trigger or clear Emergency SOS alarm."""
    sos_state["active"] = not sos_state["active"]
    return {"sos_active": sos_state["active"]}


@app.post("/api/simulate")
async def simulate_action(req: SimulateRequest):
    """Harness for testing wink, blink, and gaze coordinates."""
    if req.action:
        tracker.trigger_simulated_action(req.action)
    if req.gaze_x is not None and req.gaze_y is not None:
        tracker.simulated_gaze = (req.gaze_x, req.gaze_y)
        tracker.gaze_x = req.gaze_x
        tracker.gaze_y = req.gaze_y
    return {"status": "ok"}

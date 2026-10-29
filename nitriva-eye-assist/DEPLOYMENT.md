# Deploy Nitriva on Render

This deployment serves the web UI and FastAPI backend from one HTTPS origin.
Render builds the repository Dockerfile and runs `src.server:app`, not the older
Flask `app.py`. No separate frontend hosting or npm build is needed.

## Deploy with a Blueprint

1. Push these changes to a GitHub repository you can connect to Render.
2. In [Render](https://dashboard.render.com/), choose **New > Blueprint**.
3. Connect the repository and select the branch containing these changes.
4. Render reads `render.yaml`. Keep the `free` plan for an initial demo and
   create the service.
5. Wait for the Docker build and `/healthz` readiness check to succeed, then
   open the service's `https://...onrender.com` URL.
6. Log in with demo key `1234`, allow camera access, and calibrate while facing
   the webcam. Use one active browser session at a time.

If creating a Web Service manually, select **Docker**, use `./Dockerfile`, set
`NITRIVA_CAMERA_MODE=browser`, and set the health-check path to `/healthz`.
Keep the Docker command unchanged; it reads Render's `PORT` and uses one worker.
The image defaults to browser camera mode and limits BLAS/OpenMP worker threads.

## Cloud behavior

- The browser opens the visitor's camera through `getUserMedia` and sends JPEG
  frames over secure WebSocket `/ws/telemetry`. MediaPipe processes these frames
  on the server; video is transmitted to the host, not processed entirely locally.
- Browser mode opens no server webcam and runs no synthetic capture loop.
- The browser waits for an inference acknowledgement before sending the next
  frame, so a slow server drops frames instead of accumulating stale gaze input.
- CPU inference runs outside the HTTP event loop so health and API requests can
  still be handled during tracking.
- `/healthz` returns 200 only if the MediaPipe detector loaded; missing or invalid
  model assets produce 503. The model in `models/face_landmarker.task` is copied
  into the image and checked during the build.

## Free-plan expectations

Render's free service sleeps after 15 minutes without inbound HTTP traffic or
WebSocket messages. Waking it takes about a minute. The browser reconnects after
connection loss. See [Render's current free-service limits](https://render.com/docs/free).

The Free plan is for demonstrations. Model memory and tracking speed must be
checked on the live instance; these changes do not guarantee a particular FPS or
that the ML process fits every free-instance resource limit. If the service is
killed for memory, use a larger instance or move inference into the browser.

Calibration and device state are shared in this prototype, so use one active
patient/browser session per service. Calibration files on the free filesystem
are lost on sleep/restart/redeploy; recalibrate after waking. Multi-user use
requires per-session trackers and calibration storage. Device buttons and SOS
are demo state/UI: no external appliance or caregiver delivery is configured.

## Local verification

Use Python 3.11:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
$env:NITRIVA_CAMERA_MODE = "browser"
.venv\Scripts\python -m unittest test_nitriva -v
.venv\Scripts\python -m unittest discover -s tests -v
node tests/test_frame_streaming.cjs
```

To start the same cloud entry point locally:

```powershell
$env:NITRIVA_CAMERA_MODE = "browser"
.venv\Scripts\python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --workers 1
```

Then open `http://localhost:8000`. For a physical server-side webcam, set
`NITRIVA_CAMERA_MODE=auto` before startup; cloud images default to `browser`.

`requirements.txt` pins direct dependencies and installs one OpenCV distribution
(the one required by MediaPipe). The Docker image installs all Linux dependencies
from `requirements-linux.lock`. Regenerate that lock after changing requirements:

```text
uv pip compile requirements.txt --python-version 3.11 --python-platform x86_64-manylinux_2_28 --only-binary :all: --output-file requirements-linux.lock
```

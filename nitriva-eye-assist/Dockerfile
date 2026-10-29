FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    NITRIVA_CAMERA_MODE=browser \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app
COPY requirements-linux.lock .
RUN pip install --no-cache-dir -r requirements-linux.lock
COPY . .

# Use the model committed with the app; fail the build if it is missing.
RUN test -s models/face_landmarker.task

EXPOSE 8000
# One worker keeps this demo's in-memory tracker and calibration consistent.
CMD ["sh", "-c", "exec uvicorn src.server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers"]

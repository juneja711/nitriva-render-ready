from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import cv2
import os
import json
from optics_engine import NitrivaOpticsEngine
from sentence_generator import NitrivaSentenceSynthesizer
from tts_engine import NitrivaVoiceEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

engine = NitrivaOpticsEngine(camera_index=0)
synthesizer = NitrivaSentenceSynthesizer()
voice = NitrivaVoiceEngine()

# Global state
app_state = {
    "isLoggedIn": False,
    "route": "/dashboard",
    "devices": {"light": False, "ac": False, "tv": False},
    "selectedKeywords": [],
    "lastSentence": "Select keywords to speak...",
    "volume": 80
}

def generate_video_stream():
    """Generates continuous MJPEG stream of annotated OpenCV frames."""
    while True:
        frame, telemetry = engine.process_frame()
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/video_feed")
def video_feed():
    """Live OpenCV annotated computer-vision video feed."""
    return Response(generate_video_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/telemetry")
def get_telemetry():
    """Real-time AI telemetry: gaze coordinates, EAR, gesture, face detection, calibration."""
    telemetry = engine.get_telemetry()
    telemetry["app_state"] = app_state
    return jsonify(telemetry)

@app.route("/api/start_calibration", methods=["POST"])
def start_calibration():
    engine.start_calibration()
    voice.speak("Starting optics calibration. Look at the green target dots.")
    return jsonify({"status": "started", "calib_phase": engine.calib_phase})

@app.route("/api/skip_calibration", methods=["POST"])
def skip_calibration():
    engine.skip_calibration()
    voice.speak("Calibration skipped. Tracking ready.")
    return jsonify({"status": "skipped", "calib_phase": engine.calib_phase})

@app.route("/api/synthesize", methods=["POST"])
def synthesize():
    data = request.json or {}
    keywords = data.get("keywords", [])
    sentence = synthesizer.generate(keywords)
    app_state["lastSentence"] = sentence
    app_state["selectedKeywords"] = keywords
    voice.speak(sentence)
    return jsonify({"sentence": sentence})

@app.route("/api/speak", methods=["POST"])
def speak_route():
    data = request.json or {}
    text = data.get("text", "")
    voice.speak(text)
    return jsonify({"status": "ok"})

@app.route("/api/toggle_device", methods=["POST"])
def toggle_device():
    data = request.json or {}
    device = data.get("device", "")
    if device in app_state["devices"]:
        app_state["devices"][device] = not app_state["devices"][device]
        status = "on" if app_state["devices"][device] else "off"
        voice.speak(f"Smart {device.upper()} {status}")
    return jsonify(app_state["devices"])

@app.route("/assets/<path:path>")
def serve_assets(path):
    assets_dir = os.path.join(BASE_DIR, "assets")
    return send_from_directory(assets_dir, path)

@app.route("/favicon.svg")
def favicon():
    return send_from_directory(BASE_DIR, "favicon.svg")

@app.route("/face_landmarker.task")
def serve_model():
    return send_from_directory(BASE_DIR, "face_landmarker.task")

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_index(path):
    # Single-page application route fallback
    file_path = os.path.join(BASE_DIR, path)
    if path and os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(BASE_DIR, path)
    return send_from_directory(BASE_DIR, "index.html")

if __name__ == "__main__":
    print("=" * 60)
    print(" Nitriva Python AI/ML Server (Exact NayanVaani Engine)")
    print(" Running at: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

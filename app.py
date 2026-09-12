"""
app.py
Flask Backend — Face Recognition Attendance System
Routes:
  GET  /                  → Live camera feed page
  GET  /video_feed        → MJPEG stream (or demo image in DEMO_MODE)
  GET  /register          → Face registration page
  POST /api/register      → Register a new face (JSON: name + base64 image)
  GET  /dashboard         → Attendance analytics dashboard
  GET  /api/attendance    → JSON: attendance for a given date
  GET  /api/summary       → JSON: daily counts for chart
  GET  /api/export        → CSV download
  GET  /api/registered    → JSON: list of registered people
  POST /api/delete_person → Remove a registered person
  POST /api/mark_manual   → Manually mark a person's attendance (demo mode)
"""

import base64
import io
import logging
import os
import threading
import time
from datetime import date

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, send_file, url_for)
from PIL import Image

from attendance_manager import AttendanceManager
from face_engine import FaceEngine

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Demo Mode ─────────────────────────────────────────────────────────────────
# Set DEMO_MODE=true env var on cloud (Render). Disables webcam, enables manual marking.
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"

# ── App init ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
engine = FaceEngine()
attendance_mgr = AttendanceManager()

# ── Camera state (shared across threads) ─────────────────────────────────────
camera_lock = threading.Lock()
camera = None
latest_frame = None          # latest annotated frame for MJPEG
latest_results = []          # latest recognition results
frame_counter = 0
RECOGNITION_INTERVAL = 5    # run FR every N frames for performance


def open_camera():
    global camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    camera = cap


def camera_thread_fn():
    """Background thread: captures frames, runs recognition, marks attendance."""
    global latest_frame, latest_results, frame_counter, camera

    open_camera()
    cached_results = []

    while True:
        if camera is None or not camera.isOpened():
            time.sleep(0.5)
            continue

        ret, frame = camera.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_counter += 1

        # Run face recognition every RECOGNITION_INTERVAL frames
        if frame_counter % RECOGNITION_INTERVAL == 0 and engine._trained:
            annotated, results = engine.process_frame(frame.copy())
            cached_results = results

            # Auto-mark attendance for recognized faces
            for r in results:
                if r["name"] != "Unknown":
                    attendance_mgr.mark_attendance(r["name"])
        else:
            # Still draw cached boxes on current frame
            annotated = frame.copy()
            for r in cached_results:
                x, y, w, h = r["box"]
                name = r["name"]
                conf = r["confidence"]
                color = (0, 200, 100) if name != "Unknown" else (0, 60, 220)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                label = f"{name}  {conf:.1f}%"
                cv2.rectangle(annotated, (x, y - 24), (x + w, y), color, -1)
                cv2.putText(annotated, label, (x + 4, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Timestamp overlay
        ts = time.strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(annotated, ts, (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        with camera_lock:
            latest_frame = annotated
            latest_results = cached_results


# Start camera thread only when NOT in demo mode
if not DEMO_MODE:
    threading.Thread(target=camera_thread_fn, daemon=True).start()
    logger.info("Camera thread started (live mode).")
else:
    logger.info("DEMO_MODE enabled — camera thread skipped.")


# ── Demo frame generator ──────────────────────────────────────────────────────

def generate_demo_frame():
    """Yields a static 'Demo Mode' placeholder MJPEG frame."""
    while True:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (20, 20, 40)  # dark navy background

        # Draw icon area
        cv2.rectangle(frame, (220, 120), (420, 280), (40, 40, 80), -1)
        cv2.rectangle(frame, (220, 120), (420, 280), (80, 80, 160), 2)

        # Camera icon (simplified)
        cv2.circle(frame, (320, 195), 40, (100, 100, 200), 2)
        cv2.circle(frame, (320, 195), 15, (100, 100, 200), -1)
        cv2.rectangle(frame, (240, 150), (400, 250), (0, 0, 0), 0)

        # Text
        cv2.putText(frame, "DEMO MODE", (200, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 180, 255), 2)
        cv2.putText(frame, "Camera not available on cloud server.", (70, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 200), 1)
        cv2.putText(frame, "Use Register page to add faces via photo.", (60, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 200), 1)
        cv2.putText(frame, "Use Dashboard to mark attendance manually.", (55, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 200), 1)

        ts = time.strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (10, 468),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(1)  # static frame, refresh once per second


# ── MJPEG generator ───────────────────────────────────────────────────────────

def generate_mjpeg():
    while True:
        with camera_lock:
            frame = latest_frame

        if frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Initializing camera...", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            frame = placeholder

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(1 / 25)   # ~25 fps stream



# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/capture_frame")
def api_capture_frame():
    """Return the latest camera frame as base64 JPEG for registration page."""
    if DEMO_MODE:
        return jsonify({"success": False,
                        "message": "Camera not available in demo mode. Please upload a photo instead."}), 503

    import base64 as b64mod
    with camera_lock:
        frame = latest_frame

    if frame is None:
        return jsonify({"success": False, "message": "Camera not ready yet."}), 503

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    encoded = b64mod.b64encode(buf.tobytes()).decode("utf-8")
    return jsonify({"success": True, "image": f"data:image/jpeg;base64,{encoded}"})


@app.route("/")
def index():
    return render_template("index.html", today=date.today().isoformat(), demo_mode=DEMO_MODE)


@app.route("/video_feed")
def video_feed():
    if DEMO_MODE:
        return Response(generate_demo_frame(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/register")
def register_page():
    return render_template("register.html",
                           registered=engine.get_registered_names(),
                           demo_mode=DEMO_MODE)


@app.route("/api/register", methods=["POST"])
def api_register():
    """Receives JSON: { "name": str, "image": "<base64 JPEG>" }"""
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    image_b64 = data.get("image", "")

    if not name:
        return jsonify({"success": False, "message": "Name is required."}), 400
    if not image_b64:
        return jsonify({"success": False, "message": "Image is required."}), 400

    try:
        # Decode base64 → numpy BGR image
        img_bytes = base64.b64decode(image_b64.split(",")[-1])
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"success": False, "message": f"Image decode error: {e}"}), 400

    result = engine.register_face(name, frame)
    status = 200 if result["success"] else 422
    return jsonify(result), status


@app.route("/api/delete_person", methods=["POST"])
def api_delete_person():
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    result = engine.delete_person(name)
    return jsonify(result), (200 if result["success"] else 404)


@app.route("/api/mark_manual", methods=["POST"])
def api_mark_manual():
    """Manually mark a person as present (used in demo mode)."""
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "message": "Name is required."}), 400
    result = attendance_mgr.mark_attendance(name)
    return jsonify(result), 200


@app.route("/dashboard")
def dashboard():
    dates = attendance_mgr.get_available_dates()
    selected = request.args.get("date", date.today().isoformat())
    records = attendance_mgr.get_attendance(selected)
    registered = engine.get_registered_names()
    present_names = {r["name"] for r in records}
    absent = [n for n in registered if n not in present_names]
    return render_template("dashboard.html",
                           records=records,
                           absent=absent,
                           selected_date=selected,
                           dates=dates,
                           registered=registered,
                           demo_mode=DEMO_MODE)


@app.route("/api/attendance")
def api_attendance():
    target = request.args.get("date", date.today().isoformat())
    records = attendance_mgr.get_attendance(target)
    return jsonify({"date": target, "records": records, "count": len(records)})


@app.route("/api/summary")
def api_summary():
    return jsonify(attendance_mgr.get_summary_by_date())


@app.route("/api/registered")
def api_registered():
    return jsonify(engine.get_registered_names())


@app.route("/api/export")
def api_export():
    target = request.args.get("date", None)
    csv_str = attendance_mgr.export_csv(target)
    buf = io.BytesIO(csv_str.encode("utf-8"))
    filename = f"attendance_{target or 'all'}.csv"
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True, download_name=filename)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Face Recognition Attendance System")
    mode = "DEMO" if DEMO_MODE else "LIVE"
    print(f"  Mode: {mode}")
    print("  Open: http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

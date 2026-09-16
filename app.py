"""
app.py
Flask Backend — Face Recognition Attendance System
Features:
  - Client-side WebRTC / HTML5 Webcam AI recognition
  - Role-Based Access Control (RBAC): Admin login for management & export
  - Check-In / Check-Out punch logging & duration tracking
  - Defaulter Analysis (< 75% attendance)
  - Anti-spoofing / eye presence & multi-face verification
"""

import base64
import io
import logging
import os
import threading
import time
from datetime import date
from functools import wraps

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, send_file, url_for, session)
from PIL import Image

from attendance_manager import AttendanceManager
from face_engine import FaceEngine

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration & Credentials ──────────────────────────────────────────────
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

# ── App init ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "vedant-fc-attendance-secret-key-2026")

engine = FaceEngine()
attendance_mgr = AttendanceManager()


# ── RBAC Decorator ────────────────────────────────────────────────────────────

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("is_admin"):
            if request.path.startswith("/api/"):
                return jsonify({"success": False, "message": "Admin authorization required."}), 401
            return redirect(url_for("login_page", next=request.path))
        return f(*args, **kwargs)
    return decorated_function


@app.context_processor
def inject_admin_status():
    return {
        "is_admin": session.get("is_admin", False),
        "demo_mode": DEMO_MODE
    }


# ── Camera state (shared across threads for optional local feed) ──────────────
camera_lock = threading.Lock()
camera = None
latest_frame = None
latest_results = []
frame_counter = 0
RECOGNITION_INTERVAL = 5


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
    """Background thread for local host machine webcam capture."""
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

        if frame_counter % RECOGNITION_INTERVAL == 0 and engine._trained:
            annotated, results, meta = engine.process_frame(frame.copy())
            cached_results = results

            # Auto-mark attendance for recognized faces if single face
            if not meta.get("multiple_faces"):
                for r in results:
                    if r.get("status") == "known" and r["name"] != "Unknown":
                        attendance_mgr.mark_attendance(r["name"])
        else:
            annotated = frame.copy()
            for r in cached_results:
                x, y, w, h = r["box"]
                name = r["name"]
                conf = r["confidence"]
                color = (0, 200, 100) if r.get("status") == "known" else (0, 60, 220)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                label = f"{name} {conf:.0f}%"
                cv2.rectangle(annotated, (x, y - 24), (x + w, y), color, -1)
                cv2.putText(annotated, label, (x + 4, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        ts = time.strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(annotated, ts, (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        with camera_lock:
            latest_frame = annotated
            latest_results = cached_results


# Do not start server-side cv2.VideoCapture loop because the browser uses HTML5
# getUserMedia to capture frames directly. Keeping this disabled prevents Windows
# hardware camera lock conflicts between Python and Chrome.
logger.info("Server running in browser-webcam mode (hardware lock released for browser).")



def get_demo_image_bytes():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (10, 10, 15)
    cv2.rectangle(frame, (220, 120), (420, 280), (30, 30, 40), -1)
    cv2.circle(frame, (320, 195), 35, (180, 180, 180), 2)
    cv2.circle(frame, (320, 195), 12, (255, 255, 255), -1)
    cv2.putText(frame, "LIVE CAMERA FEED", (180, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


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
        time.sleep(1 / 25)


# ── Public Routes (Kiosk) ─────────────────────────────────────────────────────

@app.route("/")
def index():
    """Kiosk screen for students/employees to view live feed & mark attendance."""
    return render_template("index.html", today=date.today().isoformat())


@app.route("/video_feed")
def video_feed():
    if DEMO_MODE:
        return Response(get_demo_image_bytes(), mimetype="image/jpeg")
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    """
    Receives JSON: { "image": "<base64 JPEG>", "punch_type": "auto"|"in"|"out" }
    Runs face recognition, liveness check, multiple-face check, and logs check-in/out.
    """
    data = request.get_json(force=True)
    image_b64 = data.get("image", "")
    punch_type = data.get("punch_type", "auto")

    if not image_b64:
        return jsonify({"success": False, "message": "Image required."}), 400

    try:
        img_bytes = base64.b64decode(image_b64.split(",")[-1])
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"success": False, "message": f"Image decode error: {e}"}), 400

    annotated, results, meta = engine.process_frame(frame)

    marked_new = []
    # Only auto-mark if single face detected (prevents proxy / multi-person spoofing)
    if not meta.get("multiple_faces"):
        for r in results:
            if r.get("status") == "known" and r["name"] != "Unknown":
                res = attendance_mgr.mark_attendance(r["name"], punch_type=punch_type)
                if res.get("success"):
                    marked_new.append({
                        "name": r["name"],
                        "punch_type": res.get("punch_type"),
                        "time": res.get("time", ""),
                        "check_in": res.get("check_in", ""),
                        "check_out": res.get("check_out", ""),
                        "duration": res.get("duration", "--"),
                        "already_marked": res.get("already_marked", False),
                        "message": res.get("message", "")
                    })

    return jsonify({
        "success": True,
        "results": results,
        "meta": meta,
        "count": len(results),
        "marked": marked_new
    })


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


# ── Authentication Routes ─────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login_page():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["is_admin"] = True
            next_url = request.args.get("next") or url_for("dashboard")
            return redirect(next_url)
        else:
            error = "Invalid admin username or password. (Default: admin / admin123)"

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("is_admin", None)
    return redirect(url_for("index"))


# ── Protected Admin Routes ────────────────────────────────────────────────────

@app.route("/register")
@admin_required
def register_page():
    return render_template("register.html",
                           registered=engine.get_registered_names())


@app.route("/api/register", methods=["POST"])
@admin_required
def api_register():
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    image_b64 = data.get("image", "")

    if not name:
        return jsonify({"success": False, "message": "Name is required."}), 400
    if not image_b64:
        return jsonify({"success": False, "message": "Image is required."}), 400

    try:
        img_bytes = base64.b64decode(image_b64.split(",")[-1])
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"success": False, "message": f"Image decode error: {e}"}), 400

    result = engine.register_face(name, frame)
    status = 200 if result["success"] else 422
    return jsonify(result), status


@app.route("/api/delete_person", methods=["POST"])
@admin_required
def api_delete_person():
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    result = engine.delete_person(name)
    return jsonify(result), (200 if result["success"] else 404)


@app.route("/api/mark_manual", methods=["POST"])
@admin_required
def api_mark_manual():
    """Admin override to manually punch or update records."""
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    punch_type = data.get("punch_type", "auto")
    if not name:
        return jsonify({"success": False, "message": "Name is required."}), 400
    result = attendance_mgr.mark_attendance(name, punch_type=punch_type)
    return jsonify(result), 200


@app.route("/dashboard")
@admin_required
def dashboard():
    dates = attendance_mgr.get_available_dates()
    selected = request.args.get("date", date.today().isoformat())
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if start_date and end_date:
        records = attendance_mgr.get_attendance_range(start_date, end_date)
    else:
        records = attendance_mgr.get_attendance(selected)

    registered = engine.get_registered_names()
    present_names = {r["name"] for r in records}
    absent = [n for n in registered if n not in present_names]
    defaulters = attendance_mgr.get_defaulters(registered, threshold=75.0)

    return render_template("dashboard.html",
                           records=records,
                           absent=absent,
                           selected_date=selected,
                           start_date=start_date or "",
                           end_date=end_date or "",
                           dates=dates,
                           registered=registered,
                           defaulters=defaulters)


@app.route("/api/defaulters")
@admin_required
def api_defaulters():
    threshold = float(request.args.get("threshold", 75.0))
    registered = engine.get_registered_names()
    defaulters = attendance_mgr.get_defaulters(registered, threshold=threshold)
    return jsonify({
        "threshold": threshold,
        "records": defaulters,
        "count": len(defaulters)
    })


@app.route("/api/export")
@admin_required
def api_export():
    target = request.args.get("date", None)
    start_date = request.args.get("start_date", None)
    end_date = request.args.get("end_date", None)

    csv_str = attendance_mgr.export_csv(target_date=target, start_date=start_date, end_date=end_date)
    buf = io.BytesIO(csv_str.encode("utf-8"))
    filename = f"attendance_report_{target or 'all'}.csv"
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True, download_name=filename)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Vedant FC Attendance — Smart Recognition System")
    mode = "DEMO" if DEMO_MODE else "LIVE"
    print(f"  Mode: {mode}")
    print("  Admin Credentials: admin / admin123")
    print("  Open Kiosk: http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

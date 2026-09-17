# 🎯 Face Recognition Attendance System

A **real-time, browser-based Face Recognition Attendance System** built with Python, OpenCV, dlib, and Flask. Detects and identifies faces live from a webcam, automatically marks attendance, and presents analytics on a modern web dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-red?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎥 **Live Camera Feed** | Real-time MJPEG stream in browser with bounding boxes |
| 🧠 **Face Recognition** | dlib 128-d face encodings with configurable tolerance |
| 📋 **Auto Attendance** | Marks present once per person per day automatically |
| 📊 **Analytics Dashboard** | Pie chart, 30-day bar chart, present/absent tables |
| 👤 **Face Registration** | Submit new faces via webcam; teacher approval activates recognition |
| 📁 **CSV Export** | Download attendance for any date or all records |
| 🔒 **Duplicate Prevention** | SQLite UNIQUE constraint prevents double entries |
| 📱 **Responsive UI** | Modern dark theme, works on any screen size |

---

## 🏗️ Architecture

```
face-attendance-system/
├── app.py                    # Flask server + MJPEG stream + REST API
├── face_engine.py            # Face detection, encoding, recognition
├── attendance_manager.py     # SQLite attendance CRUD + CSV export
├── templates/
│   ├── index.html            # Live feed + real-time attendance list
│   ├── register.html         # Webcam-based face enrollment
│   └── dashboard.html        # Analytics dashboard
├── static/
│   ├── css/style.css         # Modern dark theme
│   └── js/main.js            # Shared JS utilities
├── data/
│   ├── known_faces/          # Saved face images per person
│   ├── encodings/            # Pickled face encodings
│   └── attendance.db         # SQLite database (auto-created)
└── requirements.txt
```

---

## ⚙️ Installation

### Prerequisites
- **Python 3.10+**
- **CMake** (required by dlib on Windows)
- **Visual C++ Build Tools** (Windows only)

### Step 1 — Install CMake & Build Tools (Windows)
```bash
# Option A: Install via winget
winget install Kitware.CMake

# Option B: Download from https://cmake.org/download/
```
Also install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the **"Desktop development with C++"** workload.

### Step 2 — Clone & set up environment
```bash
git clone https://github.com/yourusername/face-attendance-system.git
cd face-attendance-system

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

> **Note**: Installing `dlib` can take 5–10 minutes as it compiles from source.

### Step 3 — Run
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 🚀 Usage

### 1️⃣ Register Faces
1. Navigate to **Register** (`/register`) from the live attendance page
2. Enter the person's name
3. Click **Capture Photo** 3 times (move slightly between captures)
4. Click **Register** — a pending request is sent to the teacher
5. A teacher signs in to the dashboard and approves the request; only then is face recognition activated

### 2️⃣ Take Attendance
1. Go to **Live** (`/`) — the camera starts automatically
2. Recognized faces are highlighted with name + confidence %
3. Attendance is marked automatically (once per person per day)
4. The right panel updates live every 3 seconds

### 3️⃣ View & Export
- Open **Dashboard** (`/dashboard`) for analytics and charts
- Use the date picker to view historical records
- Click **Export CSV** to download attendance data

---

## 🔌 REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/video_feed` | MJPEG camera stream |
| `POST` | `/api/register` | Register a face `{name, image}` |
| `GET` | `/api/attendance?date=YYYY-MM-DD` | Fetch attendance for a date |
| `GET` | `/api/summary` | Daily attendance counts (30 days) |
| `GET` | `/api/registered` | List all registered names |
| `POST` | `/api/delete_person` | Remove a person `{name}` |
| `GET` | `/api/export?date=YYYY-MM-DD` | Download CSV |

---

## 🛠️ Configuration

Edit `face_engine.py` to tune recognition:

```python
TOLERANCE = 0.5   # 0.4 = stricter, 0.6 = more lenient
RECOGNITION_INTERVAL = 5  # Run FR every N frames (lower = more CPU)
```

---

## 🧰 Tech Stack

- **Face Recognition**: [face_recognition](https://github.com/ageitgey/face_recognition) (dlib HOG + 128-d embeddings)
- **Backend**: Flask 3.0, Python 3.10+
- **Database**: SQLite (via `sqlite3`)
- **Video**: OpenCV, MJPEG streaming
- **Frontend**: HTML5, Chart.js 4, Font Awesome 6
- **Data**: Pickle for encoding persistence, Pandas for CSV

---

## 📄 License

MIT © 2024 — Free to use for personal and educational projects.

"""
face_engine.py
Core Face Recognition Engine — OpenCV LBPH
Uses:
  - Haar Cascade (built-in OpenCV) for face detection
  - LBPH (Local Binary Pattern Histograms) for recognition
  - No TensorFlow, no dlib, no CMake — pure pip install!

Why LBPH?
  - Works out-of-the-box with opencv-contrib-python
  - Fast inference, ideal for real-time webcam streams
  - Robust to slight lighting variations
  - Production-ready for controlled environments (classrooms, offices)
"""

import os
import pickle
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

ENCODINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "encodings", "model.pkl")
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "data", "known_faces")

# Haar cascade XML — bundled locally since OpenCV 5 no longer ships these
CASCADE_PATH = os.path.join(os.path.dirname(__file__), "data", "haarcascade_frontalface_default.xml")

# LBPH confidence threshold — LOWER is more similar in LBPH
# Faces with confidence > THRESHOLD are labelled "Unknown"
CONFIDENCE_THRESHOLD = 70


class FaceEngine:
    def __init__(self):
        # Map: label_id (int) → name (str)
        self.id_to_name: dict[int, str] = {}
        self.name_to_id: dict[str, int] = {}
        self.next_id: int = 0

        # Training samples stored in memory for incremental re-training
        self.train_faces: list[np.ndarray] = []   # grayscale face images
        self.train_labels: list[int] = []          # corresponding int labels

        # OpenCV face recognizer and detector
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        self.detector = cv2.CascadeClassifier(CASCADE_PATH)
        self._trained = False

        self._load_model()

    # ─────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────

    def _load_model(self):
        """Load saved LBPH model and label map from disk."""
        if not os.path.exists(ENCODINGS_PATH):
            return
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                data = pickle.load(f)
            self.id_to_name   = data["id_to_name"]
            self.name_to_id   = data["name_to_id"]
            self.next_id      = data["next_id"]
            self.train_faces  = data["train_faces"]
            self.train_labels = data["train_labels"]

            if self.train_faces:
                self.recognizer.train(self.train_faces, np.array(self.train_labels))
                self._trained = True
            logger.info(f"Loaded model with {len(self.id_to_name)} people.")
        except Exception as e:
            logger.warning(f"Could not load model: {e}")

    def _save_model(self):
        """Persist everything to disk."""
        os.makedirs(os.path.dirname(ENCODINGS_PATH), exist_ok=True)
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump({
                "id_to_name":   self.id_to_name,
                "name_to_id":   self.name_to_id,
                "next_id":      self.next_id,
                "train_faces":  self.train_faces,
                "train_labels": self.train_labels,
            }, f)

    def _retrain(self):
        """Re-train the LBPH model on all stored samples."""
        if self.train_faces:
            self.recognizer.train(self.train_faces, np.array(self.train_labels))
            self._trained = True

    # ─────────────────────────────────────────────
    # Face extraction helper
    # ─────────────────────────────────────────────

    def _extract_face(self, bgr_image: np.ndarray) -> np.ndarray | None:
        """
        Detect the largest face in image.
        Returns 100×100 grayscale face ROI, or None if no face found.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        # Equalize histogram to improve detection in varying lighting
        gray = cv2.equalizeHist(gray)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return None

        # Use largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (100, 100))
        return face_resized

    # ─────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────

    def register_face(self, name: str, image_bgr: np.ndarray) -> dict:
        """Add a face sample and re-train the model."""
        face = self._extract_face(image_bgr)
        if face is None:
            return {
                "success": False,
                "message": "No face detected. Please use a clear, well-lit photo facing the camera."
            }

        name = name.strip().title()

        # Assign label ID to new person
        if name not in self.name_to_id:
            self.name_to_id[name] = self.next_id
            self.id_to_name[self.next_id] = name
            self.next_id += 1

        label = self.name_to_id[name]
        self.train_faces.append(face)
        self.train_labels.append(label)

        # Re-train incrementally
        self._retrain()

        # Save reference image
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        count = len(os.listdir(person_dir))
        cv2.imwrite(os.path.join(person_dir, f"{count + 1}.jpg"), image_bgr)

        self._save_model()
        samples = self.train_labels.count(label)
        return {
            "success": True,
            "message": f"'{name}' registered! ({samples} sample(s) total)"
        }

    def delete_person(self, name: str) -> dict:
        """Remove all samples for a person and re-train."""
        name = name.strip().title()
        if name not in self.name_to_id:
            return {"success": False, "message": f"'{name}' not found."}

        label = self.name_to_id[name]
        # Filter out this person's samples
        kept = [(f, l) for f, l in zip(self.train_faces, self.train_labels) if l != label]
        removed = len(self.train_faces) - len(kept)
        self.train_faces  = [x[0] for x in kept]
        self.train_labels = [x[1] for x in kept]

        del self.name_to_id[name]
        del self.id_to_name[label]

        # Re-create recognizer and re-train
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        self._trained = False
        self._retrain()
        self._save_model()

        return {"success": True, "message": f"'{name}' removed ({removed} sample(s) deleted)."}

    def get_registered_names(self) -> list[str]:
        return sorted(self.name_to_id.keys())

    # ─────────────────────────────────────────────
    # Recognition
    # ─────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Detect and recognize all faces in a BGR frame.
        Returns (annotated_frame, results).
        Result: {"name": str, "confidence": float, "box": (x,y,w,h)}
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        faces = self.detector.detectMultiScale(
            gray_eq,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        results = []
        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))

            name = "Unknown"
            display_conf = 0.0

            if self._trained:
                label_id, lbph_dist = self.recognizer.predict(face_roi)
                # LBPH dist: 0 = perfect, 100+ = poor match
                # Convert to a 0–100 confidence score (higher = better)
                confidence = max(0.0, 100.0 - lbph_dist)
                display_conf = round(confidence, 1)

                if lbph_dist < CONFIDENCE_THRESHOLD:
                    name = self.id_to_name.get(label_id, "Unknown")

            results.append({
                "name": name,
                "confidence": display_conf,
                "box": (x, y, w, h)
            })

            # ── Draw annotation ──
            color = (0, 200, 100) if name != "Unknown" else (0, 60, 220)
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)

            label = f"{name}  {display_conf:.1f}%" if name != "Unknown" else "Unknown"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame_bgr, (x, y - th - 10), (x + tw + 8, y), color, -1)
            cv2.putText(frame_bgr, label, (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return frame_bgr, results

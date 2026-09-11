"""
face_engine.py
Core Face Recognition Engine
- Loads/saves face encodings from disk
- Processes video frames to detect and identify faces
- Returns recognized names with confidence scores
"""

import face_recognition
import cv2
import numpy as np
import pickle
import os
import logging

logger = logging.getLogger(__name__)

ENCODINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "encodings", "encodings.pkl")
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "data", "known_faces")

# Tolerance: lower = stricter matching (0.4 strict, 0.6 default)
TOLERANCE = 0.5


class FaceEngine:
    def __init__(self):
        self.known_encodings = []   # list of 128-d face encoding arrays
        self.known_names = []       # corresponding names
        self.load_encodings()

    # ─────────────────────────────────────────────
    # Encoding persistence
    # ─────────────────────────────────────────────

    def load_encodings(self):
        """Load saved face encodings from disk."""
        if os.path.exists(ENCODINGS_PATH):
            with open(ENCODINGS_PATH, "rb") as f:
                data = pickle.load(f)
            self.known_encodings = data.get("encodings", [])
            self.known_names = data.get("names", [])
            logger.info(f"Loaded {len(self.known_names)} known faces.")
        else:
            logger.info("No encodings file found. Starting fresh.")

    def save_encodings(self):
        """Persist face encodings to disk."""
        os.makedirs(os.path.dirname(ENCODINGS_PATH), exist_ok=True)
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump({"encodings": self.known_encodings, "names": self.known_names}, f)
        logger.info(f"Saved {len(self.known_names)} encodings.")

    # ─────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────

    def register_face(self, name: str, image_bgr: np.ndarray) -> dict:
        """
        Encode a face from a BGR image and add it to the known set.
        Returns {"success": bool, "message": str}
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")

        if not boxes:
            return {"success": False, "message": "No face detected in the image. Please try again."}
        if len(boxes) > 1:
            return {"success": False, "message": "Multiple faces detected. Please use a photo with a single face."}

        encoding = face_recognition.face_encodings(rgb, boxes)[0]

        # Avoid duplicate registration of the same person
        if self.known_encodings:
            distances = face_recognition.face_distance(self.known_encodings, encoding)
            min_dist = np.min(distances)
            min_idx = np.argmin(distances)
            if min_dist < TOLERANCE:
                existing = self.known_names[min_idx]
                if existing.lower() == name.lower():
                    return {"success": False, "message": f"'{name}' is already registered."}

        self.known_encodings.append(encoding)
        self.known_names.append(name.strip().title())

        # Save face image for reference
        person_dir = os.path.join(KNOWN_FACES_DIR, name.strip().title())
        os.makedirs(person_dir, exist_ok=True)
        count = len(os.listdir(person_dir))
        img_path = os.path.join(person_dir, f"{count + 1}.jpg")
        cv2.imwrite(img_path, image_bgr)

        self.save_encodings()
        return {"success": True, "message": f"'{name.strip().title()}' registered successfully!"}

    def delete_person(self, name: str) -> dict:
        """Remove all encodings for a given person."""
        name = name.strip().title()
        indices = [i for i, n in enumerate(self.known_names) if n == name]
        if not indices:
            return {"success": False, "message": f"'{name}' not found."}
        for i in sorted(indices, reverse=True):
            self.known_encodings.pop(i)
            self.known_names.pop(i)
        self.save_encodings()
        return {"success": True, "message": f"'{name}' removed ({len(indices)} encoding(s) deleted)."}

    def get_registered_names(self) -> list:
        """Return sorted unique list of registered people."""
        return sorted(set(self.known_names))

    # ─────────────────────────────────────────────
    # Recognition
    # ─────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Detect and recognize faces in a BGR video frame.

        Returns:
            annotated_frame: BGR frame with bounding boxes + labels drawn
            results: list of dicts {"name": str, "confidence": float, "box": (top,right,bottom,left)}
        """
        # Resize frame for faster processing, then scale detections back
        small = cv2.resize(frame_bgr, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        boxes_small = face_recognition.face_locations(rgb_small, model="hog")
        encodings = face_recognition.face_encodings(rgb_small, boxes_small)

        results = []
        for enc, box_small in zip(encodings, boxes_small):
            top, right, bottom, left = [v * 2 for v in box_small]  # scale back up
            name = "Unknown"
            confidence = 0.0

            if self.known_encodings:
                distances = face_recognition.face_distance(self.known_encodings, enc)
                min_idx = int(np.argmin(distances))
                min_dist = float(distances[min_idx])
                confidence = round((1 - min_dist) * 100, 1)

                if min_dist <= TOLERANCE:
                    name = self.known_names[min_idx]

            results.append({
                "name": name,
                "confidence": confidence,
                "box": (top, right, bottom, left)
            })

            # Draw bounding box
            color = (0, 200, 100) if name != "Unknown" else (0, 60, 220)
            cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)

            # Label background
            label = f"{name}  {confidence:.1f}%" if name != "Unknown" else "Unknown"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame_bgr, (left, top - text_h - 10), (left + text_w + 8, top), color, -1)
            cv2.putText(frame_bgr, label, (left + 4, top - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return frame_bgr, results

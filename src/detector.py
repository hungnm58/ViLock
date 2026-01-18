"""Face detection module using MediaPipe for high accuracy."""

import cv2
import os
import urllib.request
from pathlib import Path

# Model path in assets
MODEL_PATH = Path(__file__).parent.parent / "assets" / "blaze_face_short_range.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


def download_model():
    """Download MediaPipe face detection model if not present."""
    if MODEL_PATH.exists():
        return str(MODEL_PATH)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face detection model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return str(MODEL_PATH)


class FaceDetector:
    """MediaPipe-based face detector with configurable confidence threshold.

    Falls back to Haar Cascade if MediaPipe unavailable.
    """

    def __init__(self, min_confidence: float = 0.5):
        """Initialize face detector.

        Args:
            min_confidence: Minimum detection confidence (0.0-1.0).
                           Lower = more sensitive, higher = fewer false positives.
        """
        self.min_confidence = min_confidence
        self._use_mediapipe = MEDIAPIPE_AVAILABLE

        if self._use_mediapipe:
            self._init_mediapipe()
        else:
            self._init_haar_cascade()

    def _init_mediapipe(self):
        """Initialize MediaPipe face detection (v0.10+ Tasks API)."""
        model_path = download_model()

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,  # Optimized for continuous frames
            min_detection_confidence=self.min_confidence
        )
        self.detector = vision.FaceDetector.create_from_options(options)
        self._frame_timestamp = 0

    def _init_haar_cascade(self):
        """Initialize Haar Cascade fallback."""
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect(self, frame, scale: float = 1.0) -> tuple[bool, list]:
        """Detect faces in frame.

        Args:
            frame: BGR image from OpenCV VideoCapture
            scale: Scale factor for faster processing (0.5 = half resolution)

        Returns:
            Tuple of (face_detected: bool, faces: list of detections)
        """
        # Downscale for faster processing if requested
        if scale < 1.0:
            small = cv2.resize(frame, None, fx=scale, fy=scale)
            if self._use_mediapipe:
                detected, faces = self._detect_mediapipe(small)
            else:
                detected, faces = self._detect_haar(small)
            # Scale face coordinates back to original size
            if detected:
                inv_scale = 1.0 / scale
                faces = [(int(x*inv_scale), int(y*inv_scale), int(w*inv_scale), int(h*inv_scale), c)
                         for (x, y, w, h, c) in faces]
            return detected, faces

        if self._use_mediapipe:
            return self._detect_mediapipe(frame)
        return self._detect_haar(frame)

    def _detect_mediapipe(self, frame) -> tuple[bool, list]:
        """Detect faces using MediaPipe Tasks API (VIDEO mode)."""
        # Convert BGR (OpenCV) to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image from numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run detection with timestamp (VIDEO mode requires monotonic timestamps)
        self._frame_timestamp += 33  # ~30fps interval in ms
        result = self.detector.detect_for_video(mp_image, self._frame_timestamp)

        if result.detections:
            faces = []
            h, w = frame.shape[:2]
            for det in result.detections:
                bbox = det.bounding_box
                # Coordinates are already absolute in Tasks API
                x = bbox.origin_x
                y = bbox.origin_y
                width = bbox.width
                height = bbox.height
                confidence = det.categories[0].score if det.categories else 0.5
                faces.append((x, y, width, height, confidence))
            return True, faces
        return False, []

    def _detect_haar(self, frame) -> tuple[bool, list]:
        """Detect faces using Haar Cascade (fallback)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        if len(faces) > 0:
            # Add confidence placeholder for Haar
            faces_with_conf = [(x, y, w, h, 1.0) for (x, y, w, h) in faces]
            return True, faces_with_conf
        return False, []

    def close(self):
        """Release detector resources."""
        if self._use_mediapipe and hasattr(self.detector, 'close'):
            self.detector.close()

    @property
    def backend(self) -> str:
        """Return current detection backend name."""
        return "MediaPipe" if self._use_mediapipe else "Haar Cascade"

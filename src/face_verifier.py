"""Face Verification - Verify face matches registered user."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Path to store registered face (inside project directory)
FACE_DATA_DIR = Path(__file__).parent.parent / "data" / "faces"
FACE_IMAGE_PATH = FACE_DATA_DIR / "registered_face.jpg"
FACE_ENCODING_PATH = FACE_DATA_DIR / "face_encoding.npy"

# Cache face cascade and encoding for faster verification
_face_cascade = None
_cached_encoding = None
_encoding_loaded = False


def ensure_data_dir():
    """Create data directory if not exists."""
    FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def register_face(frame: np.ndarray) -> bool:
    """Register a face from camera frame.

    Args:
        frame: BGR image from camera

    Returns:
        True if face registered successfully
    """
    try:
        ensure_data_dir()

        # Detect face using Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        if len(faces) == 0:
            print("No face detected for registration")
            return False

        if len(faces) > 1:
            print("Multiple faces detected - please show only one face")
            return False

        # Get the face region
        x, y, w, h = faces[0]

        # Add some padding
        pad = int(w * 0.2)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)

        face_img = frame[y:y+h, x:x+w]

        # Save face image
        cv2.imwrite(str(FACE_IMAGE_PATH), face_img)

        # Create face encoding using histogram
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (128, 128))
        encoding = face_resized.flatten().astype(np.float32)
        encoding = encoding / np.linalg.norm(encoding)  # Normalize
        np.save(FACE_ENCODING_PATH, encoding)

        print(f"Face registered successfully: {FACE_IMAGE_PATH}")
        return True

    except Exception as e:
        print(f"Face registration error: {e}")
        return False


def get_registered_encoding() -> Optional[np.ndarray]:
    """Load registered face encoding.

    Returns:
        Face encoding array or None
    """
    try:
        if FACE_ENCODING_PATH.exists():
            return np.load(FACE_ENCODING_PATH)
        return None
    except Exception:
        return None


def is_face_registered() -> bool:
    """Check if a face is registered."""
    return FACE_IMAGE_PATH.exists() and FACE_ENCODING_PATH.exists()


def _get_face_cascade():
    """Get cached face cascade."""
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    return _face_cascade


def _get_cached_encoding():
    """Get cached registered encoding."""
    global _cached_encoding, _encoding_loaded
    if not _encoding_loaded:
        _cached_encoding = get_registered_encoding()
        _encoding_loaded = True
    return _cached_encoding


def clear_encoding_cache():
    """Clear encoding cache (call after registering new face)."""
    global _cached_encoding, _encoding_loaded
    _cached_encoding = None
    _encoding_loaded = False


def verify_face(frame: np.ndarray, threshold: float = 0.6) -> Tuple[bool, float]:
    """Verify if face in frame matches registered face.

    Args:
        frame: BGR image from camera
        threshold: Similarity threshold (0-1, higher = stricter)

    Returns:
        Tuple of (is_match, similarity_score)
    """
    try:
        registered = _get_cached_encoding()
        if registered is None:
            # No registered face - allow anyone
            return (True, 1.0)

        # Detect face using cached cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = _get_face_cascade()
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces) == 0:
            return (False, 0.0)

        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Extract and encode
        face_gray = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_gray, (128, 128))
        encoding = face_resized.flatten().astype(np.float32)
        encoding = encoding / np.linalg.norm(encoding)

        # Compare using cosine similarity
        similarity = np.dot(registered, encoding)

        is_match = similarity >= threshold
        return (is_match, float(similarity))

    except Exception as e:
        print(f"Face verification error: {e}")
        return (False, 0.0)


def delete_registered_face() -> bool:
    """Delete registered face data.

    Returns:
        True if deleted successfully
    """
    try:
        if FACE_IMAGE_PATH.exists():
            FACE_IMAGE_PATH.unlink()
        if FACE_ENCODING_PATH.exists():
            FACE_ENCODING_PATH.unlink()
        return True
    except Exception:
        return False

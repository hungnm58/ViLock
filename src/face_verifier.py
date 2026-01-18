"""Face Verification using DeepFace for accurate recognition."""

import cv2
import numpy as np
import subprocess
import base64
import tempfile
import os
import threading
import queue
from pathlib import Path
from typing import Optional, Tuple

from cryptography.fernet import Fernet

# DeepFace import with fallback
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Warning: DeepFace not available, using fallback verification")

# Path to store registered face
FACE_DATA_DIR = Path(__file__).parent.parent / "data" / "faces"
FACE_IMAGE_PATH = FACE_DATA_DIR / "registered_face.enc"
UNLOCK_LOG_DIR = Path(__file__).parent.parent / "data" / "unlock_logs"
KEYCHAIN_SERVICE = "com.vilock.face"
KEYCHAIN_ACCOUNT = "encryption_key"

# Cache
_fernet = None
_registered_face_cache = None  # Cached decrypted face image
_arcface_model = None  # Cached ArcFace model

# Background queue for file I/O (reduces unlock latency)
_log_queue = queue.Queue()
_log_worker = None


def _log_worker_thread():
    """Background worker to handle file I/O asynchronously."""
    while True:
        try:
            task = _log_queue.get(timeout=1.0)
            if task is None:  # Shutdown signal
                break
            face_img, distance = task
            _write_log_file(face_img, distance)
            _log_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[LOG] Worker error: {e}", flush=True)


def _start_log_worker():
    """Start background worker if not running."""
    global _log_worker
    if _log_worker is None or not _log_worker.is_alive():
        _log_worker = threading.Thread(target=_log_worker_thread, daemon=True)
        _log_worker.start()


def preload_model():
    """Preload ArcFace model at app startup for faster verification."""
    global _arcface_model
    if DEEPFACE_AVAILABLE and _arcface_model is None:
        try:
            print("[INIT] Loading ArcFace model...", flush=True)
            _arcface_model = DeepFace.build_model('ArcFace')
            print("[INIT] ArcFace model loaded successfully", flush=True)
        except Exception as e:
            print(f"[INIT] Failed to preload ArcFace: {e}", flush=True)


def log_unlock_attempt(face_img: np.ndarray, distance: float, is_match: bool):
    """Queue face image for async logging (FAIL only).

    Args:
        face_img: Face image from camera
        distance: Verification distance
        is_match: Whether face matched registered user
    """
    # Only log failed attempts (potential intruders)
    if is_match:
        return

    # Start worker if needed and queue task
    _start_log_worker()
    _log_queue.put((face_img.copy(), distance))  # Copy to avoid race condition


def _write_log_file(face_img: np.ndarray, distance: float):
    """Actual file I/O - runs in background thread."""
    from datetime import datetime
    try:
        UNLOCK_LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp and distance
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]  # Include ms
        filename = f"{timestamp}_dist{distance:.3f}.jpg"
        filepath = UNLOCK_LOG_DIR / filename

        cv2.imwrite(str(filepath), face_img)

        # Keep only last 30 images to save space
        logs = sorted(UNLOCK_LOG_DIR.glob("*.jpg"), reverse=True)
        for old_log in logs[30:]:
            try:
                old_log.unlink()
            except Exception:
                pass

    except Exception as e:
        print(f"[LOG] Error: {e}", flush=True)


def _get_encryption_key() -> Optional[bytes]:
    """Get encryption key from Keychain, create if not exists."""
    try:
        result = subprocess.run(
            ['security', 'find-generic-password', '-s', KEYCHAIN_SERVICE,
             '-a', KEYCHAIN_ACCOUNT, '-w'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return base64.urlsafe_b64decode(result.stdout.strip())

        # Generate new key
        key = Fernet.generate_key()
        key_b64 = base64.urlsafe_b64encode(key).decode()
        subprocess.run(
            ['security', 'add-generic-password', '-s', KEYCHAIN_SERVICE,
             '-a', KEYCHAIN_ACCOUNT, '-w', key_b64, '-U'],
            capture_output=True
        )
        return key
    except Exception as e:
        print(f"Keychain error: {e}")
        return None


def _get_fernet() -> Optional[Fernet]:
    """Get cached Fernet cipher."""
    global _fernet
    if _fernet is None:
        key = _get_encryption_key()
        if key:
            _fernet = Fernet(key)
    return _fernet


def _encrypt_data(data: bytes) -> Optional[bytes]:
    """Encrypt data."""
    f = _get_fernet()
    return f.encrypt(data) if f else None


def _decrypt_data(data: bytes) -> Optional[bytes]:
    """Decrypt data."""
    f = _get_fernet()
    try:
        return f.decrypt(data) if f else None
    except Exception:
        return None


def ensure_data_dir():
    """Create data directory if not exists."""
    FACE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def check_face_quality(face_img: np.ndarray) -> Tuple[bool, str]:
    """Check if face image meets quality standards.

    Returns:
        (is_good, message) - quality assessment result
    """
    h, w = face_img.shape[:2]

    # Check minimum size
    if w < 100 or h < 100:
        return False, f"Face too small ({w}x{h}). Need at least 100x100px"

    # Check brightness
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 50:
        return False, f"Too dark (brightness={brightness:.0f}). Move to brighter area"
    if brightness > 220:
        return False, f"Too bright (brightness={brightness:.0f}). Avoid direct light"

    # Check blur using Laplacian variance (lowered threshold for webcam)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 30:
        return False, f"Image too blurry (sharpness={laplacian_var:.0f}). Hold still"

    # Check contrast
    contrast = np.std(gray)
    if contrast < 30:
        return False, f"Low contrast ({contrast:.0f}). Improve lighting"

    return True, f"Good quality: {w}x{h}, brightness={brightness:.0f}, sharpness={laplacian_var:.0f}"


def register_face(frame: np.ndarray) -> bool:
    """Register a face from camera frame.

    Args:
        frame: BGR image from camera

    Returns:
        True if face registered successfully
    """
    global _registered_face_cache
    try:
        ensure_data_dir()

        # Use DeepFace to detect and extract face
        if DEEPFACE_AVAILABLE:
            try:
                # Extract face using DeepFace
                print("[REGISTER] Extracting face with DeepFace...", flush=True)
                faces = DeepFace.extract_faces(
                    frame,
                    detector_backend='opencv',
                    enforce_detection=False  # Don't fail if detection uncertain
                )
                if not faces:
                    print("[REGISTER] No face detected")
                    return False

                # Filter faces with good confidence
                good_faces = [f for f in faces if f.get('confidence', 0) > 0.5]
                if not good_faces:
                    print(f"[REGISTER] Low confidence faces only: {[f.get('confidence', 0) for f in faces]}")
                    good_faces = faces  # Use what we have

                if len(good_faces) > 1:
                    print("[REGISTER] Multiple faces - using largest")
                    # Use largest face
                    good_faces = [max(good_faces, key=lambda f: f['facial_area']['w'] * f['facial_area']['h'])]

                # Get face region
                face_info = good_faces[0]
                facial_area = face_info['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                print(f"[REGISTER] Face found: {w}x{h} at ({x},{y})", flush=True)

                # Add padding
                pad = int(w * 0.3)
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(frame.shape[1] - x, w + 2 * pad)
                h = min(frame.shape[0] - y, h + 2 * pad)

                face_img = frame[y:y+h, x:x+w]

            except Exception as e:
                print(f"[REGISTER] DeepFace error: {e}", flush=True)
                return False
        else:
            # Fallback to Haar Cascade
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

            x, y, w, h = faces[0]
            pad = int(w * 0.3)
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(frame.shape[1] - x, w + 2 * pad)
            h = min(frame.shape[0] - y, h + 2 * pad)
            face_img = frame[y:y+h, x:x+w]

        # Check face quality
        is_good, quality_msg = check_face_quality(face_img)
        print(f"[REGISTER] Quality: {quality_msg}")
        if not is_good:
            return False

        # Encode and encrypt
        _, img_bytes = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        enc_img = _encrypt_data(img_bytes.tobytes())

        if not enc_img:
            print("Encryption failed")
            return False

        FACE_IMAGE_PATH.write_bytes(enc_img)
        _registered_face_cache = face_img.copy()  # Cache for quick access

        print(f"Face registered (DeepFace): {FACE_IMAGE_PATH}")
        return True

    except Exception as e:
        print(f"Face registration error: {e}")
        return False


def get_registered_face() -> Optional[np.ndarray]:
    """Load and decrypt registered face image.

    Returns:
        Face image as numpy array or None
    """
    global _registered_face_cache

    if _registered_face_cache is not None:
        return _registered_face_cache

    try:
        if FACE_IMAGE_PATH.exists():
            enc_data = FACE_IMAGE_PATH.read_bytes()
            dec_data = _decrypt_data(enc_data)
            if dec_data:
                img_array = np.frombuffer(dec_data, dtype=np.uint8)
                face_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                _registered_face_cache = face_img
                return face_img
        return None
    except Exception:
        return None


def is_face_registered() -> bool:
    """Check if a face is registered."""
    return FACE_IMAGE_PATH.exists()


def clear_encoding_cache():
    """Clear face cache (call after registering new face)."""
    global _registered_face_cache
    _registered_face_cache = None


def delete_registered_face() -> bool:
    """Delete registered face data and encryption key."""
    global _fernet, _registered_face_cache
    try:
        if FACE_IMAGE_PATH.exists():
            FACE_IMAGE_PATH.unlink()

        # Delete old files
        old_files = [
            FACE_DATA_DIR / "registered_face.jpg",
            FACE_DATA_DIR / "face_encoding.npy",
            FACE_DATA_DIR / "face_encoding.enc"
        ]
        for f in old_files:
            if f.exists():
                f.unlink()

        # Delete encryption key
        subprocess.run(
            ['security', 'delete-generic-password', '-s', KEYCHAIN_SERVICE,
             '-a', KEYCHAIN_ACCOUNT],
            capture_output=True
        )
        _fernet = None
        _registered_face_cache = None

        return True
    except Exception:
        return False


def verify_face_fast(frame: np.ndarray, face_box: Tuple[int, int, int, int, float], threshold: float = 0.4) -> Tuple[bool, float]:
    """Verify face using DeepFace.

    Args:
        frame: BGR image
        face_box: (x, y, w, h, confidence) from external detector
        threshold: Distance threshold (lower = stricter, default 0.4 for cosine)

    Returns:
        (is_match, distance) - distance closer to 0 means more similar
    """
    try:
        registered_face = get_registered_face()
        if registered_face is None:
            if is_face_registered():
                print("[VERIFY] Face file exists but can't load - DENY")
                return (False, 1.0)
            print("[VERIFY] No face registered - ALLOW")
            return (True, 0.0)

        if not DEEPFACE_AVAILABLE:
            return _verify_fallback(frame, face_box, registered_face)

        # Extract face region from frame
        x, y, w, h = int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3])

        # Add padding
        pad = int(w * 0.3)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)

        current_face = frame[y:y+h, x:x+w]

        # Use ArcFace with SSD detector for proper face alignment
        result = DeepFace.verify(
            img1_path=registered_face,
            img2_path=current_face,
            model_name='ArcFace',
            detector_backend='ssd',  # SSD for accurate face alignment
            enforce_detection=False,
            distance_metric='cosine'
        )

        distance = result['distance']

        # ArcFace cosine: 0=identical, ~0.68=default threshold
        # Use ULTRA strict threshold: 0.30 (only very similar faces unlock)
        is_match = distance < 0.30
        similarity = max(0, 1.0 - distance)

        print(f"[VERIFY] ArcFace: dist={distance:.3f}, match={is_match}, sim={similarity:.2f}", flush=True)

        # Log unlock attempt with face image
        log_unlock_attempt(current_face, distance, is_match)

        return (is_match, similarity)

    except Exception as e:
        print(f"[VERIFY] DeepFace error: {e}")
        return (False, 0.0)


def _verify_fallback(frame: np.ndarray, face_box: Tuple, registered_face: np.ndarray) -> Tuple[bool, float]:
    """Fallback verification using histogram comparison."""
    try:
        x, y, w, h = int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3])

        pad = int(w * 0.2)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)

        current_face = frame[y:y+h, x:x+w]

        # Resize both to same size
        size = (128, 128)
        reg_gray = cv2.cvtColor(cv2.resize(registered_face, size), cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(cv2.resize(current_face, size), cv2.COLOR_BGR2GRAY)

        # Compare using correlation
        reg_flat = reg_gray.flatten().astype(np.float32)
        cur_flat = cur_gray.flatten().astype(np.float32)

        reg_flat = reg_flat / (np.linalg.norm(reg_flat) + 1e-7)
        cur_flat = cur_flat / (np.linalg.norm(cur_flat) + 1e-7)

        similarity = float(np.dot(reg_flat, cur_flat))
        return (similarity >= 0.6, similarity)

    except Exception:
        return (False, 0.0)


# Legacy function for compatibility
def verify_face(frame: np.ndarray, threshold: float = 0.55) -> Tuple[bool, float]:
    """Verify face (legacy - detects face first)."""
    try:
        registered_face = get_registered_face()
        if registered_face is None:
            return (True, 1.0)

        if DEEPFACE_AVAILABLE:
            result = DeepFace.verify(
                img1_path=registered_face,
                img2_path=frame,
                model_name='ArcFace',
                enforce_detection=True,
                distance_metric='cosine'
            )
            is_match = result['distance'] < threshold
            return (is_match, 1.0 - min(result['distance'] / 0.68, 1.0))

        return (False, 0.0)
    except Exception as e:
        print(f"Verify error: {e}")
        return (False, 0.0)

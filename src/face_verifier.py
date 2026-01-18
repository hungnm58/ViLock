"""Face Verification - Verify face matches registered user with encryption."""

import cv2
import numpy as np
import subprocess
import base64
import io
from pathlib import Path
from typing import Optional, Tuple

from cryptography.fernet import Fernet

# Path to store registered face (inside project directory)
FACE_DATA_DIR = Path(__file__).parent.parent / "data" / "faces"
FACE_IMAGE_PATH = FACE_DATA_DIR / "registered_face.enc"  # Encrypted
FACE_ENCODING_PATH = FACE_DATA_DIR / "face_encoding.enc"  # Encrypted
KEYCHAIN_SERVICE = "com.vilock.face"
KEYCHAIN_ACCOUNT = "encryption_key"

# Cache face cascade and encoding for faster verification
_face_cascade = None
_cached_encoding = None
_encoding_loaded = False
_fernet = None


def _get_encryption_key() -> Optional[bytes]:
    """Get encryption key from Keychain, create if not exists."""
    try:
        # Try to get existing key
        result = subprocess.run(
            ['security', 'find-generic-password', '-s', KEYCHAIN_SERVICE,
             '-a', KEYCHAIN_ACCOUNT, '-w'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return base64.urlsafe_b64decode(result.stdout.strip())

        # Generate new key and store in Keychain
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
    """Encrypt data with Fernet."""
    f = _get_fernet()
    return f.encrypt(data) if f else None


def _decrypt_data(data: bytes) -> Optional[bytes]:
    """Decrypt data with Fernet."""
    f = _get_fernet()
    try:
        return f.decrypt(data) if f else None
    except Exception:
        return None


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

        # Encode face image to bytes
        _, img_bytes = cv2.imencode('.jpg', face_img)

        # Create face encoding
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (128, 128))
        encoding = face_resized.flatten().astype(np.float32)
        encoding = encoding / np.linalg.norm(encoding)

        # Encrypt and save
        enc_img = _encrypt_data(img_bytes.tobytes())
        enc_encoding = _encrypt_data(encoding.tobytes())

        if not enc_img or not enc_encoding:
            print("Encryption failed")
            return False

        FACE_IMAGE_PATH.write_bytes(enc_img)
        FACE_ENCODING_PATH.write_bytes(enc_encoding)

        print(f"Face registered (encrypted): {FACE_IMAGE_PATH}")
        return True

    except Exception as e:
        print(f"Face registration error: {e}")
        return False


def get_registered_encoding() -> Optional[np.ndarray]:
    """Load and decrypt registered face encoding.

    Returns:
        Face encoding array or None
    """
    try:
        if FACE_ENCODING_PATH.exists():
            enc_data = FACE_ENCODING_PATH.read_bytes()
            dec_data = _decrypt_data(enc_data)
            if dec_data:
                return np.frombuffer(dec_data, dtype=np.float32)
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
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        if len(faces) == 0:
            return (False, 0.0)

        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Add same padding as registration (20%)
        pad = int(w * 0.2)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)

        # Extract face from color image then convert to gray (same as registration)
        face_img = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
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
    """Delete registered face data and encryption key.

    Returns:
        True if deleted successfully
    """
    global _fernet
    try:
        # Delete encrypted files
        if FACE_IMAGE_PATH.exists():
            FACE_IMAGE_PATH.unlink()
        if FACE_ENCODING_PATH.exists():
            FACE_ENCODING_PATH.unlink()

        # Delete old unencrypted files if exist
        old_img = FACE_DATA_DIR / "registered_face.jpg"
        old_enc = FACE_DATA_DIR / "face_encoding.npy"
        if old_img.exists():
            old_img.unlink()
        if old_enc.exists():
            old_enc.unlink()

        # Delete encryption key from Keychain
        subprocess.run(
            ['security', 'delete-generic-password', '-s', KEYCHAIN_SERVICE,
             '-a', KEYCHAIN_ACCOUNT],
            capture_output=True
        )
        _fernet = None

        return True
    except Exception:
        return False


def verify_face_fast(frame: np.ndarray, face_box: Tuple[int, int, int, int], threshold: float = 0.6) -> Tuple[bool, float]:
    """Fast face verification using pre-detected face box (no face detection).

    Args:
        frame: BGR image
        face_box: (x, y, w, h) from external detector

    Returns:
        (is_match, similarity)
    """
    try:
        registered = _get_cached_encoding()
        if registered is None:
            # No registered face or decryption failed - deny access for security
            if is_face_registered():
                return (False, 0.0)  # Face file exists but can't decrypt - deny
            return (True, 1.0)  # No face registered - allow (first time setup)

        x, y, w, h = int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3])

        # Add 20% padding (same as registration)
        pad = int(w * 0.2)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(frame.shape[1] - x, w + 2 * pad)
        h = min(frame.shape[0] - y, h + 2 * pad)

        # Extract and encode
        face_img = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (128, 128))
        encoding = face_resized.flatten().astype(np.float32)
        norm = np.linalg.norm(encoding)
        if norm == 0:
            return (False, 0.0)
        encoding = encoding / norm

        similarity = float(np.dot(registered, encoding))
        return (similarity >= threshold, similarity)

    except Exception as e:
        print(f"Fast verify error: {e}")
        return (False, 0.0)



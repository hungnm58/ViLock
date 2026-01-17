#!/usr/bin/env python3
"""AutoLock - macOS auto screen lock when no face detected.

Main entry point for the menu bar application.
"""

import sys


def check_dependencies() -> bool:
    """Check if required dependencies are available.

    Returns:
        True if all dependencies available
    """
    missing = []

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        import rumps
    except ImportError:
        missing.append("rumps")

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False

    return True


def main():
    """Main entry point."""
    print("""
    ╔═══════════════════════════════════════╗
    ║     🔐 AutoLock v2.0 - macOS          ║
    ║   Auto lock when no face detected     ║
    ╚═══════════════════════════════════════╝
    """)

    if not check_dependencies():
        sys.exit(1)

    # Check for mediapipe (optional but recommended)
    try:
        import mediapipe
        print("✓ Using MediaPipe face detection (95%+ accuracy)")
    except ImportError:
        print("⚠ MediaPipe not found, using Haar Cascade (85-90% accuracy)")
        print("  For better accuracy: pip install mediapipe")

    print("\nApp running in menu bar...")
    print("Click the lock icon (🔓) to control.\n")

    from src.menu_bar import AutoLockApp, ensure_single_instance
    import signal

    # Kill any existing instances
    ensure_single_instance()

    app = AutoLockApp()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nQuitting...")
        app.quit_app(None)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app.run()


if __name__ == '__main__':
    main()

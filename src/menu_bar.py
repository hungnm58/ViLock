"""Menu bar application using rumps."""

import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path for imports when run directly
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import cv2

try:
    import rumps
    RUMPS_AVAILABLE = True
except ImportError:
    RUMPS_AVAILABLE = False

from detector import FaceDetector
from settings import load_settings, save_settings
from autostart import toggle_autostart, is_autostart_enabled
from bluetooth_unlocker import (
    is_screen_locked, unlock_screen, get_password_from_keychain,
    save_password_to_keychain, KEYCHAIN_ITEM
)
from face_verifier import (
    register_face, verify_face, is_face_registered, delete_registered_face,
    clear_encoding_cache
)
from notifier import notify_unlock, notify_lock, notify_failed_unlock


def safe_notification(title: str, subtitle: str, message: str):
    """Send notification, silently fail if not available."""
    try:
        rumps.notification(title, subtitle, message)
    except RuntimeError:
        pass  # Notifications not available (missing Info.plist)


class AutoLockApp(rumps.App if RUMPS_AVAILABLE else object):
    """Menu bar app for AutoLock with face detection monitoring."""

    # Status icons
    ICON_UNLOCKED = "🔓"
    ICON_LOCKED = "🔒"
    ICON_PAUSED = "⏸️"
    ICON_ERROR = "⚠️"

    def __init__(self):
        if not RUMPS_AVAILABLE:
            raise ImportError("rumps is required for menu bar app")

        super().__init__("AutoLock", title=self.ICON_UNLOCKED, quit_button=None)

        # Load settings
        self.settings = load_settings()

        # Initialize detector
        self.detector = FaceDetector(
            min_confidence=self.settings['confidence_threshold']
        )

        # State
        self.is_monitoring = False
        self.is_paused = False
        self.debug_mode = False
        self.last_face_time = time.time()
        self.face_detected = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.prefs_window = None

        # Debug frame storage (for main thread display)
        self._debug_frame = None
        self._debug_faces = []
        self._debug_timer = None

        # Lock state
        self.is_real_locked = False    # True after real lock activated

        # Build menu
        self._build_menu()

    def _build_menu(self):
        """Build the menu bar menu."""
        has_password = get_password_from_keychain() is not None
        pwd_label = "🔑 Change Password..." if has_password else "🔑 Set Password..."

        self.menu = [
            rumps.MenuItem("Start Monitoring", callback=self.toggle_monitoring),
            rumps.MenuItem("Pause", callback=self.toggle_pause),
            None,  # Separator
            self._create_timeout_menu(),
            rumps.MenuItem(pwd_label, callback=self._set_unlock_password),
            self._create_face_menu(),
            self._create_telegram_menu(),
            None,
            rumps.MenuItem("Open Debug Window", callback=self.open_debug_window),
            rumps.MenuItem("Preferences...", callback=self.show_preferences),
            None,
            rumps.MenuItem("Lock Now", callback=self.lock_now),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

    def _create_face_menu(self) -> rumps.MenuItem:
        """Create face registration submenu."""
        registered = is_face_registered()
        status = "✅ Đã đăng ký" if registered else "❌ Chưa đăng ký"
        menu = rumps.MenuItem(f"👤 Khuôn mặt: {status}")

        menu.add(rumps.MenuItem("Đăng ký khuôn mặt...", callback=self._register_face))
        if registered:
            menu.add(rumps.MenuItem("Xóa khuôn mặt", callback=self._delete_face))

        return menu

    def _create_telegram_menu(self) -> rumps.MenuItem:
        """Create Telegram notification submenu."""
        token = self.settings.get('telegram_bot_token', '')
        chat_id = self.settings.get('telegram_chat_id', '')
        configured = bool(token and chat_id)
        status = "✅ Đã cấu hình" if configured else "❌ Chưa cấu hình"
        menu = rumps.MenuItem(f"📱 Telegram: {status}")

        if configured:
            # Show current config (masked token)
            masked_token = token[:8] + "..." + token[-4:] if len(token) > 12 else "***"
            menu.add(rumps.MenuItem(f"Token: {masked_token}"))
            menu.add(rumps.MenuItem(f"Chat ID: {chat_id}"))
            menu.add(None)  # Separator

        menu.add(rumps.MenuItem("Cấu hình Telegram...", callback=self._config_telegram))
        if configured:
            menu.add(rumps.MenuItem("Test gửi thông báo", callback=self._test_telegram))
            menu.add(None)  # Separator
            # Notification toggles with state checkmarks
            item_unlock = rumps.MenuItem("Thông báo Unlock", callback=self._toggle_notify_unlock)
            item_unlock.state = 1 if self.settings.get('notify_unlock', True) else 0
            menu.add(item_unlock)

            item_lock = rumps.MenuItem("Thông báo Lock", callback=self._toggle_notify_lock)
            item_lock.state = 1 if self.settings.get('notify_lock', False) else 0
            menu.add(item_lock)
            menu.add(None)  # Separator
            menu.add(rumps.MenuItem("Xóa cấu hình", callback=self._clear_telegram))

        return menu

    def _config_telegram(self, _):
        """Configure Telegram bot token and chat ID."""
        import subprocess

        # Prompt for bot token
        script = '''
        tell application "System Events"
            display dialog "Nhập Bot Token từ @BotFather:" default answer "" with title "Telegram Config"
            set token to text returned of result
            return token
        end tell
        '''
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
        if result.returncode != 0:
            return
        token = result.stdout.strip()
        if not token:
            return

        # Prompt for chat ID
        script = '''
        tell application "System Events"
            display dialog "Nhập Chat ID (lấy từ @userinfobot):" default answer "" with title "Telegram Config"
            set chatid to text returned of result
            return chatid
        end tell
        '''
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
        if result.returncode != 0:
            return
        chat_id = result.stdout.strip()
        if not chat_id:
            return

        # Save settings
        self.settings['telegram_bot_token'] = token
        self.settings['telegram_chat_id'] = chat_id
        save_settings(self.settings)

        self._update_menu_title("📱 Telegram", "📱 Telegram: ✅ Đã cấu hình")
        safe_notification("AutoLock", "Telegram", "Đã lưu cấu hình Telegram")
        self.log("Telegram configured")

    def _test_telegram(self, _):
        """Send test notification via Telegram."""
        token = self.settings.get('telegram_bot_token', '')
        chat_id = self.settings.get('telegram_chat_id', '')

        if notify_unlock(token, chat_id, 0.95):
            safe_notification("AutoLock", "Telegram", "Đã gửi test thành công!")
        else:
            safe_notification("AutoLock", "Telegram", "Gửi thất bại. Kiểm tra lại cấu hình.")

    def _clear_telegram(self, _):
        """Clear Telegram configuration."""
        self.settings['telegram_bot_token'] = ''
        self.settings['telegram_chat_id'] = ''
        save_settings(self.settings)
        self._update_menu_title("📱 Telegram", "📱 Telegram: ❌ Chưa cấu hình")
        self.log("Telegram config cleared")

    def _toggle_notify_unlock(self, sender):
        """Toggle unlock notification."""
        self.settings['notify_unlock'] = not self.settings.get('notify_unlock', True)
        save_settings(self.settings)
        sender.state = 1 if self.settings['notify_unlock'] else 0
        self.log(f"Notify unlock: {self.settings['notify_unlock']}")

    def _toggle_notify_lock(self, sender):
        """Toggle lock notification."""
        self.settings['notify_lock'] = not self.settings.get('notify_lock', False)
        save_settings(self.settings)
        sender.state = 1 if self.settings['notify_lock'] else 0
        self.log(f"Notify lock: {self.settings['notify_lock']}")

    def _toggle_notify_failed(self, sender):
        """Toggle failed unlock notification."""
        self.settings['notify_failed'] = not self.settings.get('notify_failed', True)
        save_settings(self.settings)
        sender.state = 1 if self.settings['notify_failed'] else 0
        self.log(f"Notify failed: {self.settings['notify_failed']}")

    def _send_telegram_unlock(self, similarity: float):
        """Send unlock notification to Telegram (async)."""
        token = self.settings.get('telegram_bot_token', '')
        chat_id = self.settings.get('telegram_chat_id', '')
        if token and chat_id and self.settings.get('notify_unlock', True):
            import threading
            threading.Thread(
                target=notify_unlock,
                args=(token, chat_id, similarity),
                daemon=True
            ).start()

    def _send_telegram_failed(self, similarity: float):
        """Send failed unlock notification to Telegram (async). Always sends if configured."""
        token = self.settings.get('telegram_bot_token', '')
        chat_id = self.settings.get('telegram_chat_id', '')
        if token and chat_id:
            import threading
            threading.Thread(
                target=notify_failed_unlock,
                args=(token, chat_id, similarity),
                daemon=True
            ).start()

    def _send_telegram_lock(self):
        """Send lock notification to Telegram (async)."""
        token = self.settings.get('telegram_bot_token', '')
        chat_id = self.settings.get('telegram_chat_id', '')
        if token and chat_id and self.settings.get('notify_lock', False):
            import threading
            threading.Thread(
                target=notify_lock,
                args=(token, chat_id),
                daemon=True
            ).start()

    def _register_face(self, _):
        """Capture and register user's face with camera preview."""
        self.log("Opening camera for face registration...")

        # Open camera
        cap = cv2.VideoCapture(self.settings['camera_index'])
        if not cap.isOpened():
            safe_notification("AutoLock", "Lỗi", "Không mở được camera")
            return

        captured_frame = None
        window_name = "Đăng ký khuôn mặt - Nhấn SPACE để chụp, ESC để hủy"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect face and draw rectangle
            detected, faces = self.detector.detect(frame)
            display = frame.copy()

            if detected:
                for face in faces:
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display, "Nhan SPACE de chup", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Khong thay khuon mat", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 32 and detected:  # SPACE
                captured_frame = frame.copy()

                # Show saving message on screen
                cv2.putText(display, "Dang luu...", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                cv2.imshow(window_name, display)
                cv2.waitKey(500)  # Show for 0.5s

                # Register face
                if register_face(captured_frame):
                    # Show success message
                    cv2.rectangle(display, (0, 0), (display.shape[1], 120), (0, 100, 0), -1)
                    cv2.putText(display, "Luu thanh cong!", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    cv2.putText(display, "Dang dong...", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                    cv2.imshow(window_name, display)
                    cv2.waitKey(1500)  # Show success for 1.5s

                    clear_encoding_cache()  # Refresh cache with new face
                    self._update_menu_title("👤 Khuôn mặt", "👤 Khuôn mặt: ✅ Đã đăng ký")
                    self.log("Face registered")
                else:
                    # Show error message
                    cv2.rectangle(display, (0, 0), (display.shape[1], 80), (0, 0, 150), -1)
                    cv2.putText(display, "Luu that bai! Thu lai.", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.imshow(window_name, display)
                    cv2.waitKey(2000)  # Show error for 2s
                break

        cap.release()
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)  # Process window close event on macOS

        if captured_frame is None:
            self.log("Face registration cancelled")

    def _delete_face(self, _):
        """Delete registered face."""
        if delete_registered_face():
            safe_notification("AutoLock", "Đã xóa", "Khuôn mặt đã được xóa")
            self._update_menu_title("👤 Khuôn mặt", "👤 Khuôn mặt: ❌ Chưa đăng ký")
            self.log("Face deleted")

    def _create_timeout_menu(self) -> rumps.MenuItem:
        """Create lock timeout submenu."""
        timeout = self.settings.get('lock_timeout', 10)
        label = "Ngay lập tức" if timeout <= 1 else f"{timeout}s"
        menu = rumps.MenuItem(f"Lock Timeout: {label}")

        # Immediate option
        menu.add(rumps.MenuItem(
            "Ngay lập tức",
            callback=lambda _: self._set_lock_timeout(1)
        ))
        # Other options
        for seconds in [5, 10, 15, 30, 60]:
            menu.add(rumps.MenuItem(
                f"{seconds}s",
                callback=lambda _, s=seconds: self._set_lock_timeout(s)
            ))
        return menu

    def _set_unlock_password(self, _):
        """Prompt user to set password in Keychain for face unlock."""
        script = '''
        tell application "System Events"
            display dialog "Enter your Mac login password for auto-unlock:" default answer "" with hidden answer buttons {"Cancel", "Save"} default button "Save" with title "AutoLock - Set Password"
            if button returned of result is "Save" then
                return text returned of result
            end if
        end tell
        '''
        try:
            result = subprocess.run(['osascript', '-e', script],
                                   capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and result.stdout.strip():
                password = result.stdout.strip()
                if save_password_to_keychain(password):
                    self.log("Password saved to Keychain")
                    safe_notification("AutoLock", "Password Saved",
                                    "Face unlock enabled")
                else:
                    safe_notification("AutoLock", "Error",
                                    "Failed to save password")
        except Exception as e:
            self.log(f"Password dialog error: {e}")

    def log(self, message: str):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _set_lock_timeout(self, seconds: int):
        """Update lock timeout."""
        self.settings['lock_timeout'] = seconds
        save_settings(self.settings)
        label = "Ngay lập tức" if seconds <= 1 else f"{seconds}s"
        self._update_menu_title("Lock Timeout", f"Lock Timeout: {label}")
        self.log(f"Lock timeout: {label}")

    def _update_menu_title(self, prefix: str, new_title: str):
        """Update menu item title by prefix."""
        try:
            for key in list(self.menu.keys()):
                if isinstance(key, str) and key.startswith(prefix):
                    self.menu[key].title = new_title
                    break
        except Exception:
            pass

    def toggle_monitoring(self, sender):
        """Toggle monitoring on/off."""
        if self.is_monitoring:
            self._stop_monitoring()
            sender.title = "Start Monitoring"
            self.title = self.ICON_UNLOCKED
        else:
            if self._start_monitoring():
                sender.title = "Stop Monitoring"
                self.title = self.ICON_LOCKED
            else:
                self.title = self.ICON_ERROR

    def toggle_pause(self, sender):
        """Pause/resume monitoring."""
        if not self.is_monitoring:
            return

        self.is_paused = not self.is_paused

        if self.is_paused:
            sender.title = "Resume"
            self.title = self.ICON_PAUSED
            self.log("Monitoring paused")
        else:
            sender.title = "Pause"
            self.title = self.ICON_LOCKED
            self.last_face_time = time.time()
            self.log("Monitoring resumed")

    def open_debug_window(self, _):
        """Open standalone debug window (runs CLI script in separate process)."""
        import os
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'auto_lock.py')
        python_path = sys.executable

        self.log("Opening debug window...")

        # Launch CLI script with debug mode in separate process
        subprocess.Popen(
            [python_path, script_path, '-d', '-t', '9999'],  # long timeout so it doesn't lock
            cwd=os.path.dirname(os.path.dirname(__file__))
        )

        safe_notification("AutoLock", "Debug Window",
                         "Debug window opened. Press Q to close.")

    def _start_debug_timer(self):
        """Start timer for debug frame display (runs on main thread)."""
        if self._debug_timer is None:
            self._debug_timer = rumps.Timer(self._display_debug_frame, 0.05)  # 20 FPS
            self._debug_timer.start()

    def _stop_debug_timer(self):
        """Stop debug display timer."""
        if self._debug_timer is not None:
            self._debug_timer.stop()
            self._debug_timer = None

    def _display_debug_frame(self, _):
        """Display debug frame (called from main thread via timer)."""
        if not self.debug_mode or self._debug_frame is None:
            return

        frame = self._debug_frame.copy()
        faces = self._debug_faces
        current_time = time.time()

        h, w = frame.shape[:2]

        # Draw blue circle to show frame is updating
        cv2.circle(frame, (w - 30, 30), 15, (255, 0, 0), -1)

        # Draw red rectangles around detected faces
        for face in faces:
            fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            confidence = face[4] if len(face) > 4 else 1.0

            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 3)
            cv2.rectangle(frame, (fx, fy - 25), (fx + 60, fy), (0, 0, 255), -1)
            cv2.putText(frame, f"{confidence:.0%}", (fx + 5, fy - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Status display
        status = "FACE DETECTED" if self.face_detected else "NO FACE"
        status_color = (0, 200, 0) if self.face_detected else (0, 0, 200)
        cv2.rectangle(frame, (5, 5), (220, 40), status_color, -1)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Countdown timer
        absence = current_time - self.last_face_time
        countdown = max(0, self.settings['timeout'] - absence)
        cv2.rectangle(frame, (5, 45), (180, 75), (50, 50, 50), -1)
        cv2.putText(frame, f"Lock in: {countdown:.1f}s", (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Face count
        cv2.rectangle(frame, (5, 80), (150, 110), (50, 50, 50), -1)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 103),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("AutoLock Debug - Press Q to close", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.debug_mode = False
            self._stop_debug_timer()
            cv2.destroyAllWindows()

    def _start_monitoring(self) -> bool:
        """Start camera monitoring."""
        self.log("Starting monitoring...")

        self.cap = cv2.VideoCapture(self.settings['camera_index'])
        if not self.cap.isOpened():
            safe_notification("AutoLock - Error", "Camera Error",
                              "Could not open camera. Check permissions.")
            self.log("Failed to open camera")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.is_monitoring = True
        self.is_paused = False
        self.last_face_time = time.time()
        self.is_real_locked = False

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        lock_t = self.settings.get('lock_timeout', 10)
        safe_notification("AutoLock", "Monitoring Started",
                          f"Lock after {lock_t}s without face")
        self.log(f"Monitoring started (detector: {self.detector.backend})")
        return True

    def _stop_monitoring(self):
        """Stop camera monitoring."""
        self.is_monitoring = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        safe_notification("AutoLock", "Monitoring Stopped",
                          "Auto lock disabled")
        self.log("Monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop - lock screen when no face detected."""
        while self.is_monitoring:
            if self.is_paused:
                time.sleep(0.5)
                continue

            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(1)
                    continue

                face_detected, faces = self.detector.detect(frame)
                current_time = time.time()

                if face_detected:
                    self.last_face_time = current_time
                    if not self.face_detected:
                        self.log("Face detected")
                    self.face_detected = True

                    # Auto-unlock when face detected and we locked the screen
                    if self.is_real_locked:
                        # Verify face matches registered user
                        is_owner, similarity = verify_face(frame)

                        if is_owner:
                            if get_password_from_keychain():
                                self.log(f"🔓 Verified (sim={similarity:.2f}) - Unlocking...")
                                unlock_screen()
                                self.is_real_locked = False
                                self.title = self.ICON_UNLOCKED
                                # Send Telegram notification
                                self._send_telegram_unlock(similarity)
                            else:
                                # No password - just reset state
                                self.is_real_locked = False
                                self.title = self.ICON_UNLOCKED
                        else:
                            self.log(f"⚠️ Face not matched (sim={similarity:.2f})")
                            # Send failed unlock notification
                            self._send_telegram_failed(similarity)
                else:
                    if self.face_detected:
                        self.log("Face lost")
                    self.face_detected = False

                    # Lock screen after timeout
                    absence = current_time - self.last_face_time
                    lock_timeout = self.settings.get('lock_timeout', 10)

                    if not self.is_real_locked and absence >= lock_timeout:
                        self._activate_real_lock()

                # Store frame and faces for debug display
                if self.debug_mode:
                    self._debug_frame = frame
                    self._debug_faces = faces

            except Exception as e:
                self.log(f"Monitor error: {e}")

            time.sleep(self.settings['check_interval'])

    def _draw_debug_frame(self, frame, faces, face_detected, current_time):
        """Draw debug information on frame and display."""
        h, w = frame.shape[:2]

        # Draw a blue circle in corner to show frame is updating
        cv2.circle(frame, (w - 30, 30), 15, (255, 0, 0), -1)

        # Draw red rectangle around each detected face
        for face in faces:
            fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            confidence = face[4] if len(face) > 4 else 1.0

            # Red box for face - thick line
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 3)

            # Confidence label with background
            label = f"{confidence:.0%}"
            cv2.rectangle(frame, (fx, fy - 25), (fx + 60, fy), (0, 0, 255), -1)
            cv2.putText(frame, label, (fx + 5, fy - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Status text with background
        status = "FACE DETECTED" if face_detected else "NO FACE"
        status_color = (0, 200, 0) if face_detected else (0, 0, 200)
        cv2.rectangle(frame, (5, 5), (220, 40), status_color, -1)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Countdown timer with background
        absence = current_time - self.last_face_time
        countdown = max(0, self.settings['timeout'] - absence)
        cv2.rectangle(frame, (5, 45), (180, 75), (50, 50, 50), -1)
        cv2.putText(frame, f"Lock in: {countdown:.1f}s", (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Face count
        cv2.rectangle(frame, (5, 80), (150, 110), (50, 50, 50), -1)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 103),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("AutoLock Debug - Press Q to close", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.debug_mode = False
            cv2.destroyAllWindows()

    def _activate_real_lock(self):
        """Activate real macOS lock screen."""
        self.log("Locking screen...")
        try:
            self.is_real_locked = True
            self.title = self.ICON_LOCKED

            # Lock via screen saver
            subprocess.run(['open', '-a', 'ScreenSaverEngine'],
                          capture_output=True, timeout=5)
            self.log("Screen locked - password or iPhone required")

            # Send Telegram notification
            self._send_telegram_lock()
        except Exception as e:
            self.log(f"Lock error: {e}")

    def _lock_screen(self):
        """Manual lock."""
        self._activate_real_lock()

    def lock_now(self, _):
        """Lock screen immediately."""
        self._lock_screen()

    def show_preferences(self, _):
        """Show preferences window."""
        from preferences import PreferencesWindow

        if self.prefs_window is None:
            self.prefs_window = PreferencesWindow(on_save=self._on_settings_changed)

        self.prefs_window.show()

    def _on_settings_changed(self, new_settings: dict):
        """Handle settings changes from preferences window."""
        self.settings = load_settings()

        # Update detector confidence
        if hasattr(self.detector, 'min_confidence'):
            self.detector.min_confidence = self.settings['confidence_threshold']

        # Handle autostart
        if new_settings.get('start_at_login', False) != is_autostart_enabled():
            toggle_autostart(new_settings['start_at_login'])

        self.log("Settings updated")

    def quit_app(self, _):
        """Clean up and quit."""
        self._stop_monitoring()
        self.detector.close()
        if self.prefs_window:
            self.prefs_window.close()
        rumps.quit_application()


def ensure_single_instance():
    """Ensure only one instance of the app runs."""
    import os
    import signal

    current_pid = os.getpid()
    killed_any = False

    # Use ps to get processes, checking that executable is Python (not bash with python in args)
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True, text=True
        )
        for line in result.stdout.split('\n'):
            parts = line.split()
            if len(parts) < 11:
                continue

            pid_str = parts[1]
            # Command is from column 11 onwards
            command = ' '.join(parts[10:])

            # Check if this is a Python process (executable contains /python or starts with python)
            is_python = command.startswith('python') or '/python' in command.split()[0] if command else False

            # Check if running our app
            is_our_app = 'menu_bar.py' in command or 'src.app' in command or '-m src' in command

            if is_python and is_our_app:
                try:
                    pid = int(pid_str)
                    if pid != current_pid:
                        os.kill(pid, signal.SIGKILL)
                        print(f"Killed previous instance: {pid}")
                        killed_any = True
                except (ValueError, ProcessLookupError):
                    pass
    except Exception:
        pass

    # Wait for processes to fully terminate and menu bar cleanup
    if killed_any:
        time.sleep(1.0)


if __name__ == "__main__":
    if RUMPS_AVAILABLE:
        import signal

        ensure_single_instance()
        app = AutoLockApp()

        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\nQuitting...")
            app.quit_app(None)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        app.run()
    else:
        print("rumps not available - cannot run as menu bar app")

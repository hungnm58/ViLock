"""Bluetooth Unlocker - Auto unlock MacBook when iPhone is nearby."""

import subprocess
import time
from typing import Optional

try:
    import Quartz
    QUARTZ_AVAILABLE = True
except ImportError:
    QUARTZ_AVAILABLE = False

# Keychain item name for storing password
KEYCHAIN_ITEM = "autolock_bluetooth_unlock"


def is_screen_locked() -> bool:
    """Check if macOS screen is currently locked.

    Returns:
        True if screen is locked
    """
    if not QUARTZ_AVAILABLE:
        return False

    try:
        session = Quartz.CGSessionCopyCurrentDictionary()
        if session:
            return bool(session.get("CGSSessionScreenIsLocked", 0))
        return False
    except Exception:
        return False


def wake_display() -> bool:
    """Wake up the display from sleep.

    Returns:
        True if successful
    """
    try:
        result = subprocess.run(
            ['caffeinate', '-u', '-t', '1'],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def get_password_from_keychain(keychain_item: str = KEYCHAIN_ITEM) -> Optional[str]:
    """Retrieve password from macOS Keychain.

    Args:
        keychain_item: Name of the keychain item

    Returns:
        Password string or None if not found
    """
    try:
        result = subprocess.run(
            ['security', 'find-generic-password', '-s', keychain_item, '-w'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def save_password_to_keychain(password: str, keychain_item: str = KEYCHAIN_ITEM) -> bool:
    """Save password to macOS Keychain.

    Args:
        password: Password to save
        keychain_item: Name of the keychain item

    Returns:
        True if successful
    """
    try:
        import getpass
        username = getpass.getuser()

        # Try to update existing or add new
        result = subprocess.run(
            ['security', 'add-generic-password', '-a', username,
             '-s', keychain_item, '-w', password, '-U'],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def send_password(password: str) -> bool:
    """Send password via keyboard simulation using AppleScript.

    Uses 'text type' for more reliable character handling.

    Args:
        password: Password to type

    Returns:
        True if successful
    """
    try:
        # Write password to temp file to avoid shell escaping issues
        import tempfile
        import os

        # Create temp file with password
        fd, temp_path = tempfile.mkstemp(suffix='.txt')
        try:
            os.write(fd, password.encode('utf-8'))
            os.close(fd)

            # Use temp file for reliable password entry
            script = f'''
            set pwd to do shell script "cat '{temp_path}'"
            tell application "System Events"
                keystroke pwd
                delay 0.1
                keystroke return
            end tell
            '''

            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True, timeout=10
            )
            return result.returncode == 0
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    except Exception as e:
        print(f"send_password error: {e}")
        return False


# Track last unlock attempt to prevent rapid retries
_last_unlock_attempt = 0
UNLOCK_COOLDOWN = 2  # seconds


def unlock_screen(keychain_item: str = KEYCHAIN_ITEM) -> bool:
    """Full unlock sequence: wake display + enter password.

    Args:
        keychain_item: Keychain item containing password

    Returns:
        True if unlock was attempted
    """
    global _last_unlock_attempt

    try:
        # Check cooldown to prevent rapid retries
        current_time = time.time()
        if current_time - _last_unlock_attempt < UNLOCK_COOLDOWN:
            print(f"Unlock cooldown active, waiting...")
            return False

        _last_unlock_attempt = current_time

        # Get password from keychain first
        password = get_password_from_keychain(keychain_item)
        if not password:
            print("No password in keychain")
            return False

        print(f"Password retrieved (length: {len(password)})")

        # All-in-one: wake + dismiss screensaver + type password
        import tempfile
        import os
        fd, temp_path = tempfile.mkstemp(suffix='.txt')
        try:
            os.write(fd, password.encode('utf-8'))
            os.close(fd)

            script = f'''
            do shell script "caffeinate -u -t 1"
            delay 0.1
            set pwd to do shell script "cat '{temp_path}'"
            tell application "System Events"
                key code 53
                delay 0.1
                keystroke pwd
                keystroke return
            end tell
            '''
            result = subprocess.run(['osascript', '-e', script],
                                   capture_output=True, timeout=10)
            print(f"Unlock script result: {result.returncode == 0}")
            return result.returncode == 0
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    except Exception as e:
        print(f"unlock_screen error: {e}")
        return False


def delete_password_from_keychain(keychain_item: str = KEYCHAIN_ITEM) -> bool:
    """Delete password from Keychain.

    Args:
        keychain_item: Name of the keychain item

    Returns:
        True if successful
    """
    try:
        result = subprocess.run(
            ['security', 'delete-generic-password', '-s', keychain_item],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False

"""Settings persistence using plistlib for native macOS storage."""

import plistlib
from pathlib import Path
from typing import Any

# Data stored in project's ./data directory
DATA_DIR = Path(__file__).parent.parent / "data"
SETTINGS_PATH = DATA_DIR / "settings.plist"
LAUNCH_AGENT_PATH = Path.home() / "Library" / "LaunchAgents" / "com.autolock.plist"

DEFAULT_SETTINGS = {
    'lock_timeout': 10,         # Seconds without face before lock screen
    'check_interval': 0.2,      # Seconds between detection checks (faster = quicker unlock)
    'camera_index': 0,          # Camera device index
    'confidence_threshold': 0.5, # Face detection confidence (0.0-1.0)
    'start_at_login': False,    # Auto-start when macOS boots
    'enabled': True,            # Monitoring enabled
    'telegram_bot_token': '',   # Telegram bot token from @BotFather
    'telegram_chat_id': '',     # Telegram chat ID
    'notify_unlock': True,      # Send notification on unlock
    'notify_lock': False,       # Send notification on lock
    'notify_failed': True,      # Send notification on failed unlock attempt
}


def load_settings() -> dict:
    """Load settings from plist file.

    Returns:
        Settings dict with defaults for missing keys.
    """
    settings = DEFAULT_SETTINGS.copy()

    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, 'rb') as f:
                saved = plistlib.load(f)
                settings.update(saved)
        except (plistlib.InvalidFileException, OSError):
            pass  # Use defaults on error

    return settings


def save_settings(settings: dict) -> bool:
    """Save settings to plist file.

    Args:
        settings: Settings dict to save

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_PATH, 'wb') as f:
            plistlib.dump(settings, f)
        return True
    except OSError:
        return False


def get_setting(key: str, default: Any = None) -> Any:
    """Get a single setting value.

    Args:
        key: Setting key name
        default: Default value if key not found

    Returns:
        Setting value or default.
    """
    settings = load_settings()
    return settings.get(key, default)


def set_setting(key: str, value: Any) -> bool:
    """Set a single setting value.

    Args:
        key: Setting key name
        value: Value to set

    Returns:
        True if saved successfully.
    """
    settings = load_settings()
    settings[key] = value
    return save_settings(settings)


def reset_to_defaults() -> bool:
    """Reset all settings to defaults.

    Returns:
        True if saved successfully.
    """
    return save_settings(DEFAULT_SETTINGS.copy())

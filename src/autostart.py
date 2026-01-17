"""LaunchAgent management for auto-start at login."""

import plistlib
import subprocess
import sys
from pathlib import Path

LAUNCH_AGENT_PATH = Path.home() / "Library" / "LaunchAgents" / "com.autolock.plist"
LABEL = "com.autolock"


def get_app_path() -> str:
    """Get the path to the main app script.

    Returns:
        Path to app.py or bundled executable
    """
    # Check if running from .app bundle
    if getattr(sys, 'frozen', False):
        return sys.executable

    # Running as script - find app.py
    return str(Path(__file__).parent / "app.py")


def is_autostart_enabled() -> bool:
    """Check if auto-start is currently enabled.

    Returns:
        True if LaunchAgent is loaded
    """
    result = subprocess.run(
        ['launchctl', 'list'],
        capture_output=True, text=True
    )
    return LABEL in result.stdout


def enable_autostart() -> bool:
    """Enable auto-start at login.

    Creates LaunchAgent plist and loads it.

    Returns:
        True if successful
    """
    app_path = get_app_path()
    python_path = sys.executable

    # For .app bundle, use direct path
    if app_path.endswith('.app') or getattr(sys, 'frozen', False):
        program_args = [app_path]
    else:
        program_args = [python_path, '-m', 'src.app']

    plist_content = {
        'Label': LABEL,
        'ProgramArguments': program_args,
        'RunAtLoad': True,
        'KeepAlive': {
            'SuccessfulExit': False  # Restart on crash, not on normal exit
        },
        'WorkingDirectory': str(Path(app_path).parent.parent),
        'StandardOutPath': '/tmp/autolock.log',
        'StandardErrorPath': '/tmp/autolock-error.log',
        'ProcessType': 'Interactive',
    }

    try:
        LAUNCH_AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(LAUNCH_AGENT_PATH, 'wb') as f:
            plistlib.dump(plist_content, f)

        # Unload first if already loaded
        subprocess.run(
            ['launchctl', 'unload', str(LAUNCH_AGENT_PATH)],
            capture_output=True
        )

        # Load the agent
        result = subprocess.run(
            ['launchctl', 'load', str(LAUNCH_AGENT_PATH)],
            capture_output=True
        )

        return result.returncode == 0

    except (OSError, subprocess.SubprocessError):
        return False


def disable_autostart() -> bool:
    """Disable auto-start at login.

    Unloads and removes LaunchAgent plist.

    Returns:
        True if successful
    """
    try:
        if LAUNCH_AGENT_PATH.exists():
            subprocess.run(
                ['launchctl', 'unload', str(LAUNCH_AGENT_PATH)],
                capture_output=True
            )
            LAUNCH_AGENT_PATH.unlink()

        return True

    except (OSError, subprocess.SubprocessError):
        return False


def toggle_autostart(enable: bool) -> bool:
    """Toggle auto-start on or off.

    Args:
        enable: True to enable, False to disable

    Returns:
        True if operation successful
    """
    if enable:
        return enable_autostart()
    return disable_autostart()

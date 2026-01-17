"""Bluetooth Scanner - Detect iPhone proximity via RSSI."""

import subprocess
import json
from typing import Optional, Tuple


def get_device_rssi(mac_address: str) -> Optional[int]:
    """Get RSSI of a Bluetooth device from system_profiler.

    Args:
        mac_address: Device MAC address (format: XX-XX-XX-XX-XX-XX or XX:XX:XX:XX:XX:XX)

    Returns:
        RSSI value (negative int) or None if not found/connected
    """
    try:
        result = subprocess.run(
            ['system_profiler', 'SPBluetoothDataType', '-json'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        # Normalize MAC address format
        mac_normalized = mac_address.upper().replace(':', '-')

        # Navigate JSON to find device
        bluetooth_data = data.get('SPBluetoothDataType', [])
        for controller in bluetooth_data:
            # Check connected devices
            connected = controller.get('device_connected', [])
            for device_list in connected:
                if isinstance(device_list, dict):
                    for device_name, device_info in device_list.items():
                        if isinstance(device_info, dict):
                            device_addr = device_info.get('device_address', '').upper().replace(':', '-')
                            if device_addr == mac_normalized:
                                rssi = device_info.get('device_rssi')
                                if rssi is not None:
                                    return int(rssi)
        return None
    except Exception:
        return None


def is_device_connected(mac_address: str) -> bool:
    """Check if device is currently connected via Bluetooth.

    Args:
        mac_address: Device MAC address

    Returns:
        True if device is connected
    """
    try:
        # Normalize format for blueutil (uses XX-XX-XX-XX-XX-XX)
        mac_normalized = mac_address.upper().replace(':', '-')

        result = subprocess.run(
            ['blueutil', '--is-connected', mac_normalized],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() == '1'
    except Exception:
        return False


def is_device_nearby(mac_address: str, threshold_rssi: int = -60) -> Tuple[bool, Optional[int]]:
    """Check if device is within proximity threshold.

    Args:
        mac_address: Device MAC address
        threshold_rssi: RSSI threshold (default -60 = ~2m)

    Returns:
        Tuple of (is_nearby, rssi_value)
    """
    # First check if connected
    if not is_device_connected(mac_address):
        return (False, None)

    rssi = get_device_rssi(mac_address)
    if rssi is not None and rssi > threshold_rssi:
        return (True, rssi)

    return (False, rssi)


def get_paired_devices() -> list:
    """Get list of all paired Bluetooth devices.

    Returns:
        List of dicts with 'name' and 'address' keys
    """
    try:
        result = subprocess.run(
            ['blueutil', '--paired', '--format', 'json'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        return []
    except Exception:
        return []

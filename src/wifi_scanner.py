"""WiFi Proximity Scanner - Detect iPhone on same network."""

import subprocess
import socket
from typing import Optional, Tuple


def ping_host(ip_address: str, timeout: float = 1.0) -> bool:
    """Check if host is reachable via ping.

    Args:
        ip_address: IP address to ping
        timeout: Timeout in seconds

    Returns:
        True if host responds to ping
    """
    try:
        result = subprocess.run(
            ['ping', '-c', '1', '-W', str(int(timeout * 1000)), ip_address],
            capture_output=True, timeout=timeout + 1
        )
        return result.returncode == 0
    except Exception:
        return False


def check_tcp_port(ip_address: str, port: int = 62078, timeout: float = 1.0) -> bool:
    """Check if device has specific port open (iPhone lockdown port).

    Port 62078 is the iPhone lockdownd service port.

    Args:
        ip_address: IP address to check
        port: Port number (62078 is iPhone lockdown)
        timeout: Connection timeout

    Returns:
        True if port is open
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip_address, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def is_iphone_nearby(ip_address: str) -> Tuple[bool, str]:
    """Check if iPhone is reachable on the network.

    Uses multiple methods:
    1. Ping (fastest)
    2. TCP port check (more reliable for iPhones)

    Args:
        ip_address: iPhone's IP address

    Returns:
        Tuple of (is_nearby, method_used)
    """
    if not ip_address:
        return (False, "no_ip")

    # Method 1: Ping
    if ping_host(ip_address, timeout=0.5):
        return (True, "ping")

    # Method 2: Check iPhone lockdown port (62078)
    if check_tcp_port(ip_address, 62078, timeout=0.5):
        return (True, "port")

    return (False, "unreachable")


def find_iphones_on_network() -> list:
    """Scan local network for potential iPhones.

    Uses ARP table and checks for Apple device characteristics.

    Returns:
        List of potential iPhone IP addresses
    """
    iphones = []
    try:
        # Get ARP table
        result = subprocess.run(['arp', '-a'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return []

        # Parse ARP output for IP addresses
        for line in result.stdout.split('\n'):
            if '(' in line and ')' in line:
                # Extract IP
                start = line.index('(') + 1
                end = line.index(')')
                ip = line[start:end]

                # Check if it might be an iPhone (check port 62078)
                if check_tcp_port(ip, 62078, timeout=0.3):
                    iphones.append(ip)

    except Exception:
        pass

    return iphones


def resolve_hostname(hostname: str) -> Optional[str]:
    """Resolve hostname to IP address.

    Args:
        hostname: Hostname like 'iPhone.local' or 'iPhone-cua-Hung.local'

    Returns:
        IP address or None
    """
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return None

"""Notification module - Send alerts via Telegram."""

import urllib.request
import urllib.parse
import json
from datetime import datetime
from typing import Optional


def send_telegram_notification(
    bot_token: str,
    chat_id: str,
    message: str,
    silent: bool = False
) -> bool:
    """Send notification via Telegram Bot.

    Args:
        bot_token: Telegram bot token from @BotFather
        chat_id: Telegram chat ID
        message: Message to send
        silent: If True, send without sound

    Returns:
        True if sent successfully
    """
    if not bot_token or not chat_id:
        return False

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_notification": silent
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("ok", False)

    except Exception as e:
        print(f"Telegram notification error: {e}")
        return False


def notify_unlock(bot_token: str, chat_id: str, similarity: float = 0.0) -> bool:
    """Send unlock notification.

    Args:
        bot_token: Telegram bot token
        chat_id: Telegram chat ID
        similarity: Face match similarity score

    Returns:
        True if sent successfully
    """
    now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    message = (
        f"🔓 <b>MacBook Unlocked</b>\n"
        f"⏰ {now}\n"
        f"👤 Face match: {similarity:.0%}"
    )
    return send_telegram_notification(bot_token, chat_id, message)


def notify_lock(bot_token: str, chat_id: str) -> bool:
    """Send lock notification.

    Args:
        bot_token: Telegram bot token
        chat_id: Telegram chat ID

    Returns:
        True if sent successfully
    """
    now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    message = (
        f"🔒 <b>MacBook Locked</b>\n"
        f"⏰ {now}\n"
        f"👁 No face detected"
    )
    return send_telegram_notification(bot_token, chat_id, message, silent=True)


def notify_failed_unlock(bot_token: str, chat_id: str, similarity: float = 0.0) -> bool:
    """Send failed unlock attempt notification.

    Args:
        bot_token: Telegram bot token
        chat_id: Telegram chat ID
        similarity: Face match similarity score

    Returns:
        True if sent successfully
    """
    now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    message = (
        f"⚠️ <b>Unlock Attempt Failed</b>\n"
        f"⏰ {now}\n"
        f"👤 Face match: {similarity:.0%}\n"
        f"❌ Face not recognized!"
    )
    return send_telegram_notification(bot_token, chat_id, message)

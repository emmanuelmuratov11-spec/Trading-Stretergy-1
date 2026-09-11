"""Telegram alerts -- the recommended phone channel.

Setup (2 minutes, free, no phone number shared with anyone):
  1. In Telegram, message @BotFather and send /newbot. Copy the token.
  2. Message your new bot once (it cannot message you until you do).
  3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates and copy
     result[0].message.chat.id
  4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
"""

from __future__ import annotations

import html
import logging
import os

import requests

from quantbot.notify.base import Alert

log = logging.getLogger(__name__)


class TelegramNotifier:
    name = "telegram"

    def __init__(self) -> None:
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.token and self.chat_id)

    def send(self, alert: Alert) -> bool:
        if not self.enabled:
            return False
        text = f"<b>{html.escape(alert.title)}</b>\n<pre>{html.escape(alert.body)}</pre>"
        # Telegram rejects messages over 4096 chars outright.
        if len(text) > 4000:
            text = text[:3980] + "\n...(truncated)</pre>"
        resp = requests.post(
            f"https://api.telegram.org/bot{self.token}/sendMessage",
            json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML",
                  "disable_notification": not alert.urgent},
            timeout=20,
        )
        if resp.status_code != 200:
            log.error("telegram send failed %s: %s", resp.status_code, resp.text[:200])
            return False
        return True

    def send_photo(self, path: str, caption: str = "") -> bool:
        if not self.enabled or not os.path.exists(path):
            return False
        with open(path, "rb") as fh:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendPhoto",
                data={"chat_id": self.chat_id, "caption": caption[:1000]},
                files={"photo": (os.path.basename(path), fh, "image/png")},
                timeout=60,
            )
        if resp.status_code != 200:
            log.error("telegram photo failed %s: %s", resp.status_code, resp.text[:200])
            return False
        return True

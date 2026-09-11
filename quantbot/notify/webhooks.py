"""Webhook-based channels: Discord, Slack, and ntfy.sh."""

from __future__ import annotations

import logging
import os

import requests

from quantbot.notify.base import Alert

log = logging.getLogger(__name__)


class _WebhookBase:
    name = "webhook"
    env_var = ""

    def __init__(self) -> None:
        self.url = os.environ.get(self.env_var, "").strip()
        self.enabled = bool(self.url)

    def payload(self, alert: Alert) -> dict:
        raise NotImplementedError

    def send(self, alert: Alert) -> bool:
        if not self.enabled:
            return False
        resp = requests.post(self.url, json=self.payload(alert), timeout=20)
        if resp.status_code >= 300:
            log.error("%s send failed %s: %s", self.name, resp.status_code, resp.text[:200])
            return False
        return True


class DiscordNotifier(_WebhookBase):
    name = "discord"
    env_var = "DISCORD_WEBHOOK_URL"

    def payload(self, alert: Alert) -> dict:
        body = alert.body if len(alert.body) < 1800 else alert.body[:1800] + "\n...(truncated)"
        return {"content": f"**{alert.title}**\n```\n{body}\n```"}


class SlackNotifier(_WebhookBase):
    name = "slack"
    env_var = "SLACK_WEBHOOK_URL"

    def payload(self, alert: Alert) -> dict:
        return {"text": f"*{alert.title}*\n```{alert.body}```"}


class NtfyNotifier(_WebhookBase):
    """ntfy.sh -- push notifications with no account at all.

    Install the ntfy app, subscribe to a hard-to-guess topic name, then set
    NTFY_TOPIC to it. Note that topics are public to anyone who knows the
    name, so pick something random.
    """

    name = "ntfy"
    env_var = "NTFY_TOPIC"

    def __init__(self) -> None:
        topic = os.environ.get("NTFY_TOPIC", "").strip()
        server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        self.url = f"{server}/{topic}" if topic else ""
        self.enabled = bool(topic)

    def payload(self, alert: Alert) -> dict:
        return {"topic": self.url.rsplit("/", 1)[-1], "title": alert.title,
                "message": alert.body[:3800],
                "priority": 5 if alert.urgent else 3,
                "tags": ["chart_with_upwards_trend"]}

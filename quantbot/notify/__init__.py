"""Alert delivery.

Secrets (bot tokens, chat ids, webhook URLs) are read from the environment and
never from config files, so nothing sensitive can be committed by accident.
A channel that is not configured is skipped silently rather than raising --
a missing Discord webhook should not stop a Telegram alert from going out.
"""

from __future__ import annotations

import logging

from quantbot.notify.base import Notifier, Alert
from quantbot.notify.console import ConsoleNotifier
from quantbot.notify.telegram import TelegramNotifier
from quantbot.notify.webhooks import DiscordNotifier, NtfyNotifier, SlackNotifier

log = logging.getLogger(__name__)

_REGISTRY = {
    "console": ConsoleNotifier,
    "telegram": TelegramNotifier,
    "discord": DiscordNotifier,
    "slack": SlackNotifier,
    "ntfy": NtfyNotifier,
}


def build_notifiers(channels: list[str]) -> list[Notifier]:
    out: list[Notifier] = []
    for name in channels:
        klass = _REGISTRY.get(name.strip().lower())
        if klass is None:
            log.warning("unknown alert channel %r; known: %s", name, sorted(_REGISTRY))
            continue
        try:
            n = klass()
        except Exception as exc:
            log.warning("could not construct %s notifier: %s", name, exc)
            continue
        if n.enabled:
            out.append(n)
        else:
            log.warning("%s notifier is not configured (missing env vars); skipping", name)
    if not out:
        out.append(ConsoleNotifier())
    return out


def send_all(notifiers: list[Notifier], alert: Alert) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for n in notifiers:
        try:
            results[n.name] = bool(n.send(alert))
        except Exception as exc:
            log.error("%s notifier failed: %s", n.name, exc)
            results[n.name] = False
    return results


def send_photo_all(notifiers: list[Notifier], path: str, caption: str = "") -> dict[str, bool]:
    """Best-effort image delivery. A channel that cannot carry images is not an
    error -- the text alert has already gone out and carries the substance."""
    results: dict[str, bool] = {}
    for n in notifiers:
        try:
            results[n.name] = bool(n.send_photo(path, caption))
        except Exception as exc:
            log.error("%s photo delivery failed: %s", n.name, exc)
            results[n.name] = False
    return results


__all__ = ["Notifier", "Alert", "build_notifiers", "send_all", "send_photo_all"]

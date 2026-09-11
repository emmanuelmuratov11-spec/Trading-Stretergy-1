from __future__ import annotations

from quantbot.notify.base import Alert


class ConsoleNotifier:
    """Always-available fallback. Also what CI logs show."""

    name = "console"
    enabled = True

    def send(self, alert: Alert) -> bool:
        bar = "=" * 60
        print(f"\n{bar}\n{alert.title}\n{bar}\n{alert.body}\n{bar}\n", flush=True)
        return True

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Alert:
    title: str
    body: str
    urgent: bool = False
    meta: dict = field(default_factory=dict)

    def as_text(self) -> str:
        return f"{self.title}\n\n{self.body}"


class Notifier(Protocol):
    name: str
    enabled: bool

    def send(self, alert: Alert) -> bool: ...

    def send_photo(self, path: str, caption: str = "") -> bool:
        """Deliver an image. Channels that cannot are free to return False."""
        ...

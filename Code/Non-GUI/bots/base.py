"""Controller interface used by bots, the text UI, and (later) the GUI."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Controller(ABC):
    @abstractmethod
    def decide(self, view, decision):
        """Return a legal choice for `decision` given the (redacted) `view`."""
        raise NotImplementedError

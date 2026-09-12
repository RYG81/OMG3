"""Optional ComfyUI progress and cancellation integration."""

from __future__ import annotations


class ProgressReporter:
    """Report bounded progress when running inside ComfyUI.

    The helper remains a no-op in unit tests and standalone imports where ComfyUI
    modules are unavailable. Interruption exceptions are never swallowed.
    """

    def __init__(self, total: int):
        self.total = max(1, int(total))
        self.current = 0
        try:
            from comfy.utils import ProgressBar
        except ImportError:
            self._bar = None
        else:
            self._bar = ProgressBar(self.total)

    def check_interrupted(self) -> None:
        try:
            from comfy.model_management import throw_exception_if_processing_interrupted
        except ImportError:
            return
        throw_exception_if_processing_interrupted()

    def update(self, amount: int = 1) -> None:
        self.check_interrupted()
        previous = self.current
        self.current = min(self.total, self.current + max(0, int(amount)))
        delta = self.current - previous
        if self._bar is not None and delta:
            self._bar.update(delta)

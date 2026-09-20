"""settled-computer: event-driven settling for computer-use agents.

Every action waits until the screen reacts and stops changing, then returns the
settled frame plus a verdict — no fixed sleeps, no second screenshot round trip.
"""
from .engine import (
    BusyProbe,
    Frame,
    Grabber,
    LatencyBook,
    Region,
    SettleConfig,
    SettleResult,
    act_and_settle,
    browser_busy_probe,
    install_browser_probe,
    make_mss_grabber,
    playwright_grabber,
    wait_settled,
)

__version__ = "0.1.0a1"

__all__ = [
    "BusyProbe",
    "Frame",
    "Grabber",
    "LatencyBook",
    "Region",
    "SettleConfig",
    "SettleResult",
    "act_and_settle",
    "browser_busy_probe",
    "install_browser_probe",
    "make_mss_grabber",
    "playwright_grabber",
    "wait_settled",
    "__version__",
]

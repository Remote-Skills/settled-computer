#!/usr/bin/env python3
"""
settled_computer.server - local (stdio) MCP server for desktop computer use with
event-driven settling.

Every action tool (click, type_text, press_key, scroll, drag, mouse_move) performs the action,
then waits until the screen reacts and stops changing (see settle.py), and returns the settled
screenshot plus a one-line note ("Screen settled 0.42s after the action" / "No visible change..."
/ "Still changing after 8s"). No fixed sleeps, and no separate screenshot round trip.

The `act` tool runs a whole sequence (click -> type -> key) inside ONE tool call, validates every
step before touching the desktop, and stops at the first real anomaly, so the agent model spends
one turn on what used to be three.

Design rules worth knowing
--------------------------
* Tools are serialized with one lock. Hosts may issue parallel tool calls; without the lock their
  settle windows overlap and each verdict is contaminated by the other action.
* Input (mouse/keyboard) runs on one dedicated worker thread, so a 3 s drag or a long paste never
  blocks the event loop.
* screenshot() ALWAYS returns an image. Action results may omit the image ("no image = unchanged")
  only while the last image the model received is younger than SETTLE_MCP_ELIDE_TTL seconds: the
  server cannot know what is still in the model's context (new chat, pruned images, compaction).
* A region that keeps animating (video, ticker) is auto-ignored after two consecutive bails, for
  SETTLE_MCP_AUTO_IGNORE_SECS seconds, then re-checked, so it expires once the motion stops.
* Values set with configure() are pinned: adaptive learning and per-action defaults never override them.

Install
-------
    pip install "mcp[cli]" mss pyautogui pillow numpy      # + pyperclip for non-ASCII typing
    pip install opencv-python                              # optional: ~3x faster JPEG encode
    sudo apt install python3-tk                             # Linux only: pyautogui exits without it
    python settle_mcp.py --check                            # verify capture + coordinates

Register (Claude Desktop: claude_desktop_config.json / Claude Code: `claude mcp add`)
-------------------------------------------------------------------------------------
    {
      "mcpServers": {
        "settled-computer": {
          "command": "python",
          "args": ["/absolute/path/to/settle_mcp.py"]
        }
      }
    }
    claude mcp add settled-computer -- python /absolute/path/to/settle_mcp.py

Keep settle.py in the same folder. Coordinates are in the pixels of the image the tools return.

Environment variables
---------------------
    SETTLE_MCP_MONITOR           monitor index (mss numbering, 1 = primary)          default 1
    SETTLE_MCP_MAX_WIDTH         max width of images returned to the model in px     default 1280
    SETTLE_MCP_QUALITY           JPEG quality of returned images                     default 70
    SETTLE_MCP_FAILSAFE          0 disables pyautogui's fail-safe                    default 1
    SETTLE_MCP_ELIDE_TTL         seconds an unchanged screen may omit its image      default 45 (0 = never omit)
    SETTLE_MCP_AUTO_IGNORE_SECS  seconds an auto-detected animating region is        default 30 (0 = off)
                                 ignored before it is re-checked

SAFETY: this server lets a model move your mouse and type on your real desktop. Keep the
pyautogui fail-safe on (slam the mouse into the top-left corner to abort all actions), and
prefer running it inside a VM or a dedicated user session. macOS needs Screen Recording and
Accessibility permission for the app hosting this process; Linux needs an X11 session
(Wayland blocks capture and synthetic input).

Never print to stdout in this file: stdout is the MCP transport. Logs go to stderr.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import io
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Literal, Optional, Sequence

try:  # pydantic requires typing_extensions.TypedDict on Python < 3.12
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict

import numpy as np

try:
    import mss
    import pyautogui
    from PIL import Image as PILImage
except (Exception, SystemExit) as exc:  # missing dependency, or no display (pyautogui exits without tkinter on Linux)
    sys.stderr.write(
        f"settle_mcp: cannot start ({exc!r}).\n"
        "Install: pip install 'mcp[cli]' mss pyautogui pillow numpy  (Linux also needs: sudo apt install python3-tk)\n"
        "and run inside a desktop session.\n"
    )
    raise SystemExit(1)

try:  # optional, ~3x faster than PIL for BGRA -> resized JPEG
    import cv2
except ImportError:
    cv2 = None

from pydantic import ConfigDict

try:  # mcp >= 2.0 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import Image, MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server, Image
    from mcp.server.fastmcp.exceptions import ToolError

from .engine import LatencyBook, SettleConfig, SettleResult, _as_u32, act_and_settle, wait_settled

MONITOR = int(os.environ.get("SETTLE_MCP_MONITOR", "1"))
MAX_WIDTH = int(os.environ.get("SETTLE_MCP_MAX_WIDTH", "1280"))
JPEG_QUALITY = int(os.environ.get("SETTLE_MCP_QUALITY", "70"))
ELIDE_TTL = float(os.environ.get("SETTLE_MCP_ELIDE_TTL", "45"))
AUTO_IGNORE_SECS = float(os.environ.get("SETTLE_MCP_AUTO_IGNORE_SECS", "30"))

pyautogui.PAUSE = 0.0  # pyautogui's default 0.1s pause after every call is itself a fixed sleep
pyautogui.FAILSAFE = os.environ.get("SETTLE_MCP_FAILSAFE", "1") != "0"

_IS_MAC = platform.system() == "Darwin"

# Per-action defaults. They apply only to fields the user has NOT pinned through configure().
_KIND_DEFAULTS = {
    "type": {"react_deadline": 0.2, "quiet_time": 0.12},
    "scroll": {"quiet_time": 0.15},
    "hover": {"react_deadline": 0.2},
}


def log(msg: str) -> None:
    sys.stderr.write(f"settle_mcp: {msg}\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------- tool allowlist
def _tool_enabled(name: str) -> bool:
    """SETTLE_MCP_TOOLS=a,b,c allows only the named tools. Unset = all enabled.
    lets an operator strip input-injection (e.g. `SETTLE_MCP_TOOLS=screenshot,screen_info,wait`
    for a look-but-don't-touch server) — disabled tools fail with a clear error."""
    raw = os.environ.get("SETTLE_MCP_TOOLS")
    if not raw:
        return True
    return name in {t.strip() for t in raw.split(",") if t.strip()}


def gated(name: str):
    """Decorator for mutating tools: refuse the call when the allowlist excludes it."""
    import functools

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            if not _tool_enabled(name):
                raise ToolError(
                    f"tool {name!r} is disabled on this server by the SETTLE_MCP_TOOLS allowlist")
            return await fn(*args, **kwargs)
        return wrapper
    return deco


# --------------------------------------------------------------------------- serialization
_LOCK: Optional[asyncio.Lock] = None
_INPUT_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="settle-input")  # one stable input thread


def serialized(fn):
    """Run tool calls one at a time. GUI actions are inherently sequential: overlapping settle windows
    would attribute one action's effects to the other and interleave inputs."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        global _LOCK
        if _LOCK is None:
            _LOCK = asyncio.Lock()
        async with _LOCK:
            return await fn(*args, **kwargs)
    return wrapper


async def _run_input(fn) -> None:
    """Execute blocking mouse/keyboard code off the event loop, on the dedicated input thread."""
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(_INPUT_POOL, fn)
    except pyautogui.FailSafeException:
        raise ToolError("pyautogui fail-safe triggered: the pointer is in a screen corner. "
                        "Move it away and retry (or set SETTLE_MCP_FAILSAFE=0).") from None


# --------------------------------------------------------------------------- capture + coordinates
def _make_grabber(monitor_index: int):
    """mss capture bound to one monitor. Frames are BGRA uint8. Must be created and used on the
    same thread (mss requirement on Windows), so it is created lazily inside the event loop."""
    sct = (getattr(mss, "MSS", None) or mss.mss)()  # mss.mss is deprecated in newer releases
    mon = sct.monitors[monitor_index]

    def grab() -> np.ndarray:
        return np.asarray(sct.grab(mon))  # contiguous BGRA: enables settle.py's fast compare

    return grab, mon


def _same_frame(a: Optional[np.ndarray], b: Optional[np.ndarray],
                ignore_regions: Sequence[tuple] = ()) -> bool:
    """Equal frames, optionally ignoring regions (screen fractions), e.g. a video that is always
    animating and must not defeat the 'no image = unchanged' elision. Compares packed uint32
    pixels and masks the difference instead of copying 8 MB frames."""
    if a is None or b is None or a.shape != b.shape:
        return False
    ne = _as_u32(a) != _as_u32(b)
    if not ne.any():
        return True
    h, w = ne.shape
    for x0, y0, x1, y1 in ignore_regions:
        ne[max(0, int(y0 * h) - 8):min(h, int(np.ceil(y1 * h)) + 8),
           max(0, int(x0 * w) - 8):min(w, int(np.ceil(x1 * w)) + 8)] = False
    return not ne.any()


def _overlap(a: tuple, b: tuple) -> float:
    """Intersection area divided by the smaller box's area (0..1)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (ix * iy) / smaller if smaller > 0 else 0.0


class State:
    def __init__(self) -> None:
        self.grab = None
        self.mon: dict = {}
        self.native_w = self.native_h = 0
        self.img_w = self.img_h = 0
        self.base = SettleConfig()
        self.pinned: set = set()                     # fields set via configure(): never overridden
        self.book = LatencyBook()
        self.last_frame: Optional[np.ndarray] = None
        self.last_sent: Optional[np.ndarray] = None  # last frame the model actually received
        self.last_sent_at = 0.0
        self.auto_regions: dict = {}                 # box -> monotonic time it was (re)confirmed animating
        self.pending_residual: Optional[tuple] = None

    def ensure(self) -> None:
        if self.grab is not None:
            return
        self.grab, self.mon = _make_grabber(MONITOR)
        frame = self.grab()
        self.native_h, self.native_w = frame.shape[:2]
        scale = min(1.0, MAX_WIDTH / self.native_w)
        self.img_w = max(1, round(self.native_w * scale))
        self.img_h = max(1, round(self.native_h * scale))
        self.last_frame = frame
        log(f"monitor {MONITOR}: {self.mon}; native {self.native_w}x{self.native_h}; "
            f"images {self.img_w}x{self.img_h}; jpeg encoder "
            f"{'cv2' if cv2 is not None else 'PIL'}")

    def to_point(self, x: float, y: float) -> tuple[int, int]:
        """Image pixel -> global pyautogui coordinate (handles downscaling, HiDPI, monitor offset)."""
        self.ensure()  # image size is unknown until the first capture
        if not (0 <= x < self.img_w and 0 <= y < self.img_h):
            raise ToolError(f"({x}, {y}) is outside the {self.img_w}x{self.img_h} screenshot; "
                            "coordinates are pixels of the screenshot image")
        px = self.mon["left"] + x * self.mon["width"] / self.img_w
        py = self.mon["top"] + y * self.mon["height"] / self.img_h
        return round(px), round(py)

    # ---- what the model has seen
    def mark_sent(self, frame: np.ndarray) -> None:
        self.last_sent = frame
        self.last_sent_at = time.monotonic()

    def can_elide(self) -> bool:
        return (ELIDE_TTL > 0 and self.last_sent is not None
                and time.monotonic() - self.last_sent_at <= ELIDE_TTL)

    # ---- regions we do not wait on
    def effective_ignore(self) -> tuple:
        now = time.monotonic()
        for box in [b for b, t in self.auto_regions.items() if now - t > 600]:
            del self.auto_regions[box]  # forgotten entirely after 10 minutes
        active = [b for b, t in self.auto_regions.items() if now - t < AUTO_IGNORE_SECS]
        return tuple(self.base.ignore_regions) + tuple(active)

    def learn_residual(self, box: tuple) -> bool:
        """Called on every 'residual' bail. Returns True when the region is (re)activated for auto-ignore:
        immediately if it was confirmed before, otherwise on the second consecutive bail."""
        if AUTO_IGNORE_SECS <= 0:
            return False
        now = time.monotonic()
        pad = 0.01
        padded = (max(0.0, box[0] - pad), max(0.0, box[1] - pad), min(1.0, box[2] + pad), min(1.0, box[3] + pad))
        for known in list(self.auto_regions):
            if _overlap(known, padded) > 0.5:
                self.auto_regions[known] = now
                return True
        if self.pending_residual is not None and _overlap(self.pending_residual, padded) > 0.5:
            self.auto_regions[padded] = now
            self.pending_residual = None
            return True
        self.pending_residual = padded
        return False

    # ---- encoding
    def _encode_sync(self, frame: np.ndarray) -> bytes:
        if cv2 is not None and frame.ndim == 3 and frame.shape[2] == 4:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if bgr.shape[1] != self.img_w or bgr.shape[0] != self.img_h:
                bgr = cv2.resize(bgr, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                return enc.tobytes()
        img = PILImage.fromarray(np.ascontiguousarray(frame[:, :, 2::-1]))  # BGRA -> RGB
        if img.size != (self.img_w, self.img_h):
            img = img.resize((self.img_w, self.img_h), PILImage.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()

    async def encode(self, frame: np.ndarray) -> Image:
        data = await asyncio.to_thread(self._encode_sync, frame)  # keep the event loop free
        return Image(data=data, format="jpeg")


state = State()


def _coords_note() -> str:
    return f"Coordinates are pixels of this image ({state.img_w}x{state.img_h})."


async def _settle_for(key: str, action) -> SettleResult:
    """Run one blocking input action on the input thread, wait for the UI to settle, record the sample.
    Config precedence: configure()-pinned values > per-action defaults > adaptive learning."""
    state.ensure()
    cfg = state.book.config_for(key, state.base, pinned=frozenset(state.pinned))
    defaults = {k: v for k, v in _KIND_DEFAULTS.get(key, {}).items() if k not in state.pinned}
    cfg = replace(cfg, ignore_regions=state.effective_ignore(), **defaults)
    res = await act_and_settle(state.grab, lambda: _run_input(action), cfg)
    state.book.record(key, res)
    state.last_frame = res.frame
    return res


def _residual_advice(box: tuple) -> str:
    if state.learn_residual(box):
        return (f" Auto-ignoring this region for the next {AUTO_IGNORE_SECS:.0f}s so later actions do not wait "
                "on it (it is re-checked afterwards).")
    return (" If the moving region is irrelevant (video, ticker), call configure(ignore_regions="
            f"[[{box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f}]]) to stop waiting on it.")


async def _finish(res: SettleResult, kind: str = "action") -> list:
    """Turn a SettleResult into the tool return. When nothing new is visible AND the model received an
    image recently (ELIDE_TTL), the image is omitted to save vision tokens; otherwise it is attached."""
    note = res.note_for_model(kind)
    frame = res.frame
    if res.reason == "residual" and res.motion_box:
        note += _residual_advice(res.motion_box)
    if res.reason in ("no_reaction", "residual") and state.can_elide():
        ignore = list(state.effective_ignore())
        if res.reason == "residual" and res.motion_box:
            ignore.append(res.motion_box)
        if _same_frame(frame, state.last_sent, ignore):
            age = time.monotonic() - state.last_sent_at
            where = "Outside the moving region the" if res.reason == "residual" else "The"
            return [f"{note} {where} screen is unchanged since the last image you received "
                    f"({age:.0f}s ago), so no new image is attached. Call screenshot() if you are unsure."]
    img = await state.encode(frame)
    state.mark_sent(frame)
    return [img, f"{note} {_coords_note()}"]


# --------------------------------------------------------------------------- key handling
_KEY_ALIASES = {
    "cmd": "command" if _IS_MAC else "win", "super": "command" if _IS_MAC else "win",
    "meta": "command" if _IS_MAC else "win", "windows": "win", "control": "ctrl",
    "return": "enter", "esc": "escape", "del": "delete", "pgup": "pageup", "pgdn": "pagedown",
    "option": "option" if _IS_MAC else "alt", "alt": "option" if _IS_MAC else "alt",
}


def _normalize_keys(spec: str) -> list:
    parts = [p for p in spec.lower().replace(" ", "").split("+") if p]
    keys = [_KEY_ALIASES.get(p, p) for p in parts]
    bad = [k for k in keys if k not in pyautogui.KEYBOARD_KEYS]
    if not keys or bad:
        raise ToolError(f"unknown key(s) {bad or spec!r}; examples: 'enter', 'ctrl+s', 'alt+tab', 'shift+f10'")
    return keys


def _press(combo: list) -> None:
    pyautogui.press(combo[0]) if len(combo) == 1 else pyautogui.hotkey(*combo)


def _type(text: str) -> None:
    if text.isascii():
        pyautogui.write(text, interval=0)
        return
    try:
        import pyperclip
    except ImportError as exc:
        raise ToolError("non-ASCII text needs `pip install pyperclip` (it pastes via the clipboard)") from exc
    pyperclip.copy(text)  # overwrites the clipboard
    pyautogui.hotkey("command" if _IS_MAC else "ctrl", "v")


def _scroll(direction: str, amount: int, target: Optional[tuple]) -> None:
    if target:
        pyautogui.moveTo(*target)
    if direction in ("up", "down"):
        pyautogui.scroll(amount if direction == "up" else -amount)
    else:
        pyautogui.hscroll(amount if direction == "right" else -amount)


# --------------------------------------------------------------------------- MCP tools
class ActStep(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="forbid")  # a misspelled key is an error that names it
    click: list[float]      # [x, y] in screenshot pixels
    dblclick: list[float]
    rightclick: list[float]
    move: list[float]
    type: str
    key: str
    scroll: list            # ["up"|"down"|"left"|"right", amount, optional x, optional y]
    wait: float             # seconds


_INSTRUCTIONS = (
    "Desktop control with event-driven settling: every action waits until the screen reacts and "
    "stops changing, then returns the settled screenshot plus a short note.\n"
    "Rules that save you turns:\n"
    "- Do not call screenshot() right after an action - the action's returned image IS the "
    "after-state. Use screenshot() when you have no recent frame; it always returns an image.\n"
    "- Prefer the `act` tool for sequences of known actions (e.g. click a field, type, press "
    "enter): one call, one final image. Every step is validated before anything runs.\n"
    "- An action response with NO image means the screen is unchanged since the last image you "
    "received a few seconds ago; if you no longer have that image, call screenshot().\n"
    "- 'No visible change' does not mean failure: focusing an already-focused field or setting a "
    "state that is already set changes nothing. Judge by the image, not only the note.\n"
    "- Follow the notes' explicit suggestions (e.g. configure ignore_regions for a video region)."
)

mcp = _Server("settled-computer", instructions=_INSTRUCTIONS)


@mcp.tool()
@serialized
async def screenshot():
    """Look at the current screen (waits until it is visually stable). ALWAYS returns an image. Every action
    already returns the settled screen, so use this only when you have no recent frame or after wait()."""
    state.ensure()
    cfg = replace(state.base, react_deadline=0.0, ignore_regions=state.effective_ignore())
    res = await wait_settled(state.grab, cfg)
    state.last_frame = res.frame
    if res.reason == "timeout":
        note = f"Screen was still changing after {res.waited:.1f}s (loading or animating); call wait()."
    elif res.reason == "residual" and res.motion_box:
        b = res.motion_box
        note = (f"Screen is stable except a region [{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}] that keeps "
                f"animating (bailed after {res.waited:.1f}s). Everything outside it is stable; if that region "
                "is what you are waiting on, call wait()." + _residual_advice(b))
    else:
        note = "Screen is stable."
    img = await state.encode(res.frame)
    state.mark_sent(res.frame)
    return [img, f"{note} {_coords_note()}"]


@mcp.tool()
@gated('click')
@serialized
async def click(x: float, y: float, button: Literal["left", "right", "middle"] = "left", clicks: int = 1):
    """Click at (x, y) in screenshot pixels. clicks=2 double-clicks. Returns the settled screen."""
    px, py = state.to_point(x, y)
    n = max(1, min(clicks, 3))
    return await _finish(await _settle_for("click", lambda: pyautogui.click(px, py, clicks=n, button=button)),
                         "click")


@mcp.tool()
@gated('type_text')
@serialized
async def type_text(text: str):
    """Type text into the focused control. Newlines press Enter. Returns the settled screen."""
    return await _finish(await _settle_for("type", lambda: _type(text)), "type")


@mcp.tool()
@gated('press_key')
@serialized
async def press_key(keys: str):
    """Press a key or shortcut, e.g. 'enter', 'tab', 'ctrl+s', 'alt+tab', 'cmd+space'. Returns the settled screen."""
    combo = _normalize_keys(keys)
    return await _finish(await _settle_for("key", lambda: _press(combo)), "key")


@mcp.tool()
@gated('scroll')
@serialized
async def scroll(direction: Literal["up", "down", "left", "right"], amount: int = 5,
                 x: Optional[float] = None, y: Optional[float] = None):
    """Scroll by `amount` wheel clicks, optionally at (x, y) in screenshot pixels. Returns the settled screen."""
    target = state.to_point(x, y) if x is not None and y is not None else None
    amount = max(1, min(amount, 50))
    return await _finish(await _settle_for("scroll", lambda: _scroll(direction, amount, target)), "scroll")


@mcp.tool()
@gated('drag')
@serialized
async def drag(x1: float, y1: float, x2: float, y2: float, duration: float = 0.3):
    """Left-button drag from (x1, y1) to (x2, y2) in screenshot pixels. Returns the settled screen."""
    a, b = state.to_point(x1, y1), state.to_point(x2, y2)
    secs = max(0.05, min(duration, 3.0))

    def action() -> None:
        pyautogui.moveTo(*a)
        pyautogui.dragTo(*b, duration=secs, button="left")

    return await _finish(await _settle_for("drag", action), "drag")


@mcp.tool()
@gated('mouse_move')
@serialized
async def mouse_move(x: float, y: float):
    """Move the pointer to (x, y) to reveal hover menus or tooltips. Returns the settled screen.
    Not a way to check anything - it returns the screen like every other action does."""
    px, py = state.to_point(x, y)
    return await _finish(await _settle_for("hover", lambda: pyautogui.moveTo(px, py)), "hover")


# ---- act(): validate everything first, then run
_STOP_KINDS = {"click", "dblclick", "rightclick", "key"}  # a no-op here usually means a mis-click


def _note_kind(kind: str) -> str:
    return {"wait": "wait", "move": "hover", "type": "type"}.get(kind, "action")


def _plan_step(i: int, kind: str, arg):
    """Validate one act() step WITHOUT touching the desktop; return an async callable that performs it
    and settles. Validating every step up front means a typo in step 5 cannot leave 1-4 half-executed."""
    def xy(a) -> tuple:
        try:
            x, y = float(a[0]), float(a[1])
        except (TypeError, ValueError, IndexError, KeyError):
            raise ToolError(f"step {i + 1}: {kind} expects [x, y] in screenshot pixels, got {a!r}") from None
        return state.to_point(x, y)

    if kind in ("click", "dblclick", "rightclick"):
        px, py = xy(arg)
        clicks, button = (2, "left") if kind == "dblclick" else (1, "right" if kind == "rightclick" else "left")
        return lambda: _settle_for("click", lambda: pyautogui.click(px, py, clicks=clicks, button=button))
    if kind == "move":
        px, py = xy(arg)
        return lambda: _settle_for("hover", lambda: pyautogui.moveTo(px, py))
    if kind == "type":
        text = str(arg)
        return lambda: _settle_for("type", lambda: _type(text))
    if kind == "key":
        combo = _normalize_keys(str(arg))
        return lambda: _settle_for("key", lambda: _press(combo))
    if kind == "scroll":
        if not isinstance(arg, (list, tuple)) or not arg:
            raise ToolError(f'step {i + 1}: scroll expects ["up"|"down"|"left"|"right", amount, optional x, y]')
        direction = arg[0]
        if direction not in ("up", "down", "left", "right"):
            raise ToolError(f"step {i + 1}: bad scroll direction {direction!r}")
        try:
            amount = max(1, min(int(arg[1]) if len(arg) > 1 else 5, 50))
        except (TypeError, ValueError):
            raise ToolError(f"step {i + 1}: scroll amount must be an integer, got {arg[1]!r}") from None
        target = xy(arg[2:4]) if len(arg) >= 4 else None
        return lambda: _settle_for("scroll", lambda: _scroll(direction, amount, target))
    if kind == "wait":
        try:
            seconds = max(0.1, min(float(arg), 30.0))
        except (TypeError, ValueError):
            raise ToolError(f"step {i + 1}: wait expects a number of seconds, got {arg!r}") from None

        async def run() -> SettleResult:
            cfg = replace(state.base, react_deadline=seconds, max_wait=seconds + 5.0,
                          ignore_regions=state.effective_ignore())
            res = await wait_settled(state.grab, cfg, baseline=state.last_frame)
            state.last_frame = res.frame
            return res
        return run
    raise ToolError(f"step {i + 1}: unknown kind {kind!r}; valid: click, dblclick, rightclick, "
                    "move, type, key, scroll, wait")


def _should_stop(res: SettleResult, kind: str, policy: str) -> bool:
    if res.reason == "timeout":
        return True
    if res.reason == "no_reaction":
        return policy == "always" or (policy == "auto" and kind in _STOP_KINDS)
    return False  # settled, or a residual (confined animation is not a failure)


@mcp.tool()
@gated('act')
@serialized
async def act(steps: list[ActStep], screenshot: Literal["final", "none"] = "final",
              stop_on_no_reaction: Literal["auto", "always", "never"] = "auto"):
    """Run a sequence of up to 8 actions in ONE call, each settling before the next. Steps are single-key
    objects: {"click":[x,y]}, {"dblclick":[x,y]}, {"rightclick":[x,y]}, {"type":"text"}, {"key":"enter"},
    {"scroll":["down",5]} (optionally ["down",5,x,y]), {"move":[x,y]}, {"wait":1.0}.
    Every step is validated before anything runs. A timeout always stops the sequence. 'No visible change'
    stops it only for click/dblclick/rightclick/key under stop_on_no_reaction="auto" (a click that does nothing
    is usually a mis-click); use "never" when a no-op is expected (e.g. the field is already focused) or
    "always" to stop on any no-op. Pass screenshot="none" to skip the final image when it completed cleanly."""
    state.ensure()
    if not isinstance(steps, list) or not (1 <= len(steps) <= 8):
        raise ToolError("steps must be a list of 1-8 single-key action objects, "
                        'e.g. [{"click":[640,400]}, {"type":"hello"}, {"key":"enter"}]')
    plan = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or len(step) != 1:
            raise ToolError(f"step {i + 1}: expected a single-key object like "
                            '{"click":[640,400]}, got ' + repr(step))
        kind, arg = next(iter(step.items()))
        plan.append((kind, _plan_step(i, kind, arg)))

    done: list[str] = []
    quiet: list[str] = []  # steps that produced no visible change but did not stop the sequence
    res: Optional[SettleResult] = None
    for i, (kind, run) in enumerate(plan):
        res = await run()
        if _should_stop(res, kind, stop_on_no_reaction):
            img = await state.encode(res.frame)
            state.mark_sent(res.frame)
            earlier = f" Done before it: {', '.join(done)}." if done else ""
            remaining = ", ".join(f"{j + 1}:{plan[j][0]}" for j in range(i + 1, len(plan)))
            later = f" Not run: {remaining}." if remaining else ""
            hint = (" If no change was expected here (e.g. the field was already focused), re-run the remaining "
                    'steps with stop_on_no_reaction="never".') if res.reason == "no_reaction" else ""
            return [img, f"STOPPED at step {i + 1} ({kind}): {res.note_for_model(_note_kind(kind))}"
                         f"{earlier}{later}{hint} {_coords_note()}"]
        if res.reason == "no_reaction":
            quiet.append(f"{i + 1}:{kind}")
        done.append(f"{i + 1}:{kind}")

    summary = f"All {len(plan)} steps completed ({', '.join(done)})."
    if quiet:
        summary += f" No visible change from: {', '.join(quiet)}."
    if res is not None and res.reason == "residual" and res.motion_box:
        summary += " A confined region is still animating; the rest of the screen is stable." + _residual_advice(res.motion_box)
    if screenshot == "none" and res is not None and res.reason in ("settled", "residual"):
        return [summary + ' No image attached (screenshot="none").']
    img = await state.encode(res.frame)
    state.mark_sent(res.frame)
    return [img, f"{summary} {_coords_note()}"]


@mcp.tool()
@serialized
async def wait(seconds: float = 3.0):
    """Wait up to `seconds` for something to change (e.g. a slow page or a long operation), then wait
    for it to settle. Use after a 'still changing'/'timeout' note or when a known slow operation is running."""
    state.ensure()
    seconds = max(0.1, min(seconds, 60.0))
    cfg = replace(state.base, react_deadline=seconds, max_wait=seconds + 5.0,
                  ignore_regions=state.effective_ignore())
    res = await wait_settled(state.grab, cfg, baseline=state.last_frame)
    state.last_frame = res.frame
    if res.reason == "no_reaction":
        note = res.note_for_model("wait")
    elif res.reason == "timeout":
        note = f"Still changing after {res.waited:.1f}s."
    else:
        note = f"Screen changed and settled after {res.waited:.1f}s."
    img = await state.encode(res.frame)
    state.mark_sent(res.frame)
    return [img, f"{note} {_coords_note()}"]


@mcp.tool()
@serialized
async def configure(quiet_time: Optional[float] = None, react_deadline: Optional[float] = None,
                    max_wait: Optional[float] = None, ignore_regions: Optional[list[list[float]]] = None,
                    residual_bail_after: Optional[float] = None) -> str:
    """Tune settling. quiet_time: seconds of stillness that count as settled (0.05-5). react_deadline: how long
    to wait for any change after an action (0-10). max_wait: hard cap per action (0.5-60). ignore_regions: list of
    [x0, y0, x1, y1] screen fractions (0-1) to mask out, e.g. a clock or video; pass [] to clear (this also clears
    auto-detected regions). residual_bail_after: return early when only a confined region keeps animating (0.5-10).
    Values set here are pinned: adaptive learning and per-action defaults never override them."""
    cfg = state.base
    if quiet_time is not None:
        cfg = replace(cfg, quiet_time=min(max(quiet_time, 0.05), 5.0))
        state.pinned.add("quiet_time")
    if react_deadline is not None:
        cfg = replace(cfg, react_deadline=min(max(react_deadline, 0.0), 10.0))
        state.pinned.add("react_deadline")
    if max_wait is not None:
        cfg = replace(cfg, max_wait=min(max(max_wait, 0.5), 60.0))
        state.pinned.add("max_wait")
    if residual_bail_after is not None:
        cfg = replace(cfg, residual_bail_after=min(max(residual_bail_after, 0.5), 10.0))
    if ignore_regions is not None:
        for r in ignore_regions:
            if len(r) != 4 or not all(0.0 <= v <= 1.0 for v in r) or r[0] >= r[2] or r[1] >= r[3]:
                raise ToolError(f"bad region {r!r}: expected [x0, y0, x1, y1] fractions with x0<x1, y0<y1")
        cfg = replace(cfg, ignore_regions=tuple(tuple(r) for r in ignore_regions))
        if not ignore_regions:
            state.auto_regions.clear()
            state.pending_residual = None
    state.base = cfg
    return (f"quiet_time={cfg.quiet_time}s react_deadline={cfg.react_deadline}s max_wait={cfg.max_wait}s "
            f"residual_bail_after={cfg.residual_bail_after}s "
            f"ignore_regions={[list(r) for r in cfg.ignore_regions]} pinned={sorted(state.pinned)}")


@mcp.tool()
@serialized
async def screen_info() -> str:
    """Describe the capture: image size the tools use, monitor geometry, platform."""
    state.ensure()
    auto = [[round(v, 2) for v in b] for b in state.effective_ignore()[len(state.base.ignore_regions):]]
    return (f"Images are {state.img_w}x{state.img_h}px (native {state.native_w}x{state.native_h}); "
            f"monitor {MONITOR} geometry {state.mon}; platform {platform.system()}; "
            f"auto-ignored regions {auto}.")


# --------------------------------------------------------------------------- entry point
def _check() -> None:
    grab, mon = _make_grabber(MONITOR)
    grab()  # warm up
    t0 = time.perf_counter()
    frame = grab()
    ms = (time.perf_counter() - t0) * 1000
    print(f"monitor {MONITOR}: {mon}")
    print(f"frame {frame.shape[1]}x{frame.shape[0]}, grab {ms:.0f} ms, pyautogui size {tuple(pyautogui.size())}")
    print(f"pointer now at {tuple(pyautogui.position())}; fail-safe {'on' if pyautogui.FAILSAFE else 'OFF'}")


def main() -> None:
    """Console-script entry point (`settled-computer`)."""
    ap = argparse.ArgumentParser(description="Settled computer-use MCP server (stdio)")
    ap.add_argument("--check", action="store_true", help="test capture and coordinates, then exit")
    args = ap.parse_args()
    if args.check:
        _check()
    else:
        mcp.run()  # stdio transport


if __name__ == "__main__":
    main()

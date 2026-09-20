#!/usr/bin/env python3
"""
settle.py - event-driven "wait until the UI stops changing" for computer-use agents.

Replaces the fixed `sleep(N)` most agent harnesses run after every click/keypress.

How it works
------------
1. Grab a baseline frame BEFORE performing the action (so a fast reaction is never missed).
2. Perform the action.
3. Poll frames (~20 Hz), diffing at full resolution in 8x8-pixel cells, and wait for:
     phase 1 (react):  the screen changes, or `react_deadline` passes -> "no_reaction"
     phase 2 (settle): no significant change for `quiet_time`          -> "settled"
   with `max_wait` as a hard cap                                       -> "timeout"
4. Return the last full-resolution frame, so it doubles as the screenshot for the model
   (no second capture) plus a short text note describing what happened.

Optional extra signals:
  - `busy_probe`: any async/sync callable returning True while the app is busy
    (browser: in-flight fetch/XHR, DOM mutations, running animations; desktop: busy cursor,
    process CPU, etc.). See `browser_busy_probe` below.
  - `ignore_regions`: fractions of the screen to mask out (clock, ticker, video).

Usage
-----
    grab = make_mss_grabber()                                  # desktop
    res = await act_and_settle(grab, lambda: pyautogui.click(x, y))
    send_to_model(image=res.frame, text=res.note_for_model())

    # browser (Playwright, async API)
    await install_browser_probe(page)
    cfg = SettleConfig(pixel_delta=12, busy_probe=browser_busy_probe(page))
    res = await act_and_settle(playwright_grabber(page), lambda: page.mouse.click(x, y), cfg)

Run the self-test:  python settle.py --selftest
Requires: numpy. Optional: mss (desktop capture), Pillow (browser screenshots).
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import io
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Optional, Sequence, Union

import numpy as np

Frame = np.ndarray  # full-resolution uint8 frame: H x W x 4 (BGRA, fastest), H x W x 3, or H x W
Grabber = Callable[[], Union[Frame, Awaitable[Frame]]]
BusyProbe = Callable[[], Union[bool, Awaitable[bool]]]
Region = tuple  # (x0, y0, x1, y1) as fractions of the screen, 0..1


# --------------------------------------------------------------------------- config / result
@dataclass
class SettleConfig:
    poll_interval: float = 0.05          # seconds between frames (~20 Hz)
    react_deadline: float = 0.30         # max wait for ANY change after the action
    quiet_time: float = 0.20             # continuous quiet needed to call it settled
    max_wait: float = 8.0                # hard cap
    pixel_delta: int = 0                 # 0 = exact compare (fast path, right for desktop capture);
                                         # >0 (e.g. 12) tolerates noise, e.g. JPEG browser screenshots
    block: int = 8                       # frames are compared in block x block pixel cells
    react_blocks: int = 2                # changed cells that count as "the UI reacted" (one typed char ~ 2-4)
    activity_blocks: int = 8             # changed cells that keep resetting the quiet timer
                                         # (below this, e.g. a blinking caret, counts as residual motion)
    ignore_regions: Sequence[Region] = ()
    busy_probe: Optional[BusyProbe] = None
    residual_bail_after: float = 1.5     # a confined animation (video/spinner) bails after this long
    confined_area: float = 0.35          # changed area (screen fraction) still counts as "confined"


@dataclass
class SettleResult:
    reason: str             # "settled" | "no_reaction" | "timeout" | "residual"
    reacted: bool           # did the screen (or busy probe) show any reaction
    waited: float           # seconds spent waiting (excludes the action itself)
    residual_motion: float  # 0..1: share of quiet polls with tiny changes (spinner/caret hint)
    frame: Frame            # last full-resolution frame, ready to send to the model
    motion_box: Optional[tuple] = None  # (x0, y0, x1, y1) fractions bounding the animating area
    spread: bool = False                # timeout only: motion covers more than `confined_area`

    @property
    def settled(self) -> bool:
        return self.reason in ("settled", "no_reaction")

    def note_for_model(self, kind: str = "action") -> str:
        """One-line verdict for the model. `kind` ("wait", "move"/"hover", "type", anything else)
        tunes the no-reaction wording, because "missed its target" makes no sense for a wait."""
        if self.reason == "settled":
            msg = f"Screen settled {self.waited:.2f}s after the action."
        elif self.reason == "no_reaction":
            if kind == "wait":
                msg = f"Nothing changed during {self.waited:.1f}s."
            elif kind in ("move", "hover"):
                msg = f"No visible change within {self.waited:.2f}s (no hover effect appeared)."
            elif kind == "type":
                msg = (f"No visible change within {self.waited:.2f}s of typing. The focus may be elsewhere, "
                       "or the field does not echo characters (e.g. a password box).")
            else:
                msg = (f"No visible change within {self.waited:.2f}s of the action. It may simply have had no "
                       "visible effect (the control was already focused or already in that state), missed its "
                       "target, or the app is slow to respond.")
        elif self.reason == "residual":
            box = self.motion_box
            where = f" in region [{box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f}]" if box else ""
            msg = (f"Screen settled {self.waited:.2f}s after the action, except a small region{where} that "
                   "keeps animating (video, ticker, spinner or progress indicator?). Everything outside it is "
                   "stable. If that region is the loading indicator for what you are waiting on, call wait() "
                   "instead of acting.")
        else:
            msg = (f"Screen was still changing after {self.waited:.1f}s (timeout). "
                   "Likely loading or animating; wait and re-check before acting.")
            box = self.motion_box
            if self.spread and box is not None:
                msg += (f" Motion is spread across region [{box[0]:.2f},{box[1]:.2f},"
                        f"{box[2]:.2f},{box[3]:.2f}].")
        if self.residual_motion > 0.3 and self.reason != "residual":
            msg += " A small region is still animating (spinner, ticker or video?)."
        return msg


# --------------------------------------------------------------------------- core
async def _maybe_await(value):
    return await value if inspect.isawaitable(value) else value


def _mask(shape, regions: Sequence[Region]) -> np.ndarray:
    """Boolean grid (True = watched) at block resolution; `regions` are screen fractions to ignore."""
    m = np.ones(shape, dtype=bool)
    h, w = shape
    for x0, y0, x1, y1 in regions:
        m[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))] = False
    return m


def _as_u32(frame: Frame) -> np.ndarray:
    """One uint32 per pixel so equality checks touch 4x fewer elements (BGRA/RGBA frames are
    viewed in place; other layouts are packed once)."""
    if frame.ndim == 2:
        return frame.astype(np.uint32)
    if frame.shape[2] == 4 and frame.dtype == np.uint8 and frame.flags.c_contiguous:
        return frame.view(np.uint32).reshape(frame.shape[:2])
    packed = np.zeros(frame.shape[:2] + (4,), np.uint8)
    packed[..., :3] = frame[..., :3]
    return packed.view(np.uint32).reshape(frame.shape[:2])


def _changed_blocks(prev: Frame, cur: Frame, delta: int, block: int, mask: np.ndarray) -> tuple[int, Optional[tuple]]:
    """Number of block x block cells containing a changed pixel, plus the fractional bounding box
    of the change (None when nothing changed) for the confined-animation report.
    Comparing at full resolution and then reducing with any() keeps thin text strokes visible,
    which a strided downsample would skip. delta == 0 uses a fast exact compare (~1 ms per 1080p
    frame when nothing changed); delta > 0 ignores per-channel differences up to `delta`."""
    if prev.shape != cur.shape:            # resolution changed mid-wait: treat everything as changed
        return int(mask.size), (0.0, 0.0, 1.0, 1.0)
    if delta <= 0:
        changed = _as_u32(prev) != _as_u32(cur)
    else:
        d = np.maximum(prev, cur) - np.minimum(prev, cur)   # uint8-safe |a - b|
        if d.ndim == 3:
            d = d[..., :3].max(axis=2)
        changed = d > delta
    if not changed.any():
        return 0, None
    hb, wb = changed.shape[0] // block, changed.shape[1] // block
    grid = changed[:hb * block, :wb * block].reshape(hb, block, wb, block).any(axis=(1, 3)) & mask
    n = int(grid.sum())
    if n == 0:
        return 0, None
    rows = np.nonzero(grid.any(axis=1))[0]
    cols = np.nonzero(grid.any(axis=0))[0]
    h, w = grid.shape
    box = (float(cols[0] / w), float(rows[0] / h),
           float((cols[-1] + 1) / w), float((rows[-1] + 1) / h))
    return n, box


async def wait_settled(grab: Grabber, cfg: Optional[SettleConfig] = None,
                       baseline: Optional[Frame] = None) -> SettleResult:
    """Wait until the screen reacts and then stops changing.

    Pass `baseline` = a frame captured BEFORE the action. Without it, a reaction that
    finishes before the first poll is invisible and gets misreported as "no_reaction".
    """
    cfg = cfg or SettleConfig()
    t0 = time.monotonic()
    prev = baseline if baseline is not None else await _maybe_await(grab())
    mask = _mask((prev.shape[0] // cfg.block, prev.shape[1] // cfg.block), cfg.ignore_regions)

    frame = prev
    reacted = False
    last_activity = t0
    last_busy = t0                 # a busy app gets a short grace before "no_reaction", because
    quiet_polls = tiny_polls = 0   # the busy->idle transition usually precedes the paint
    poll = min(cfg.poll_interval, 0.015)   # burst-then-relax: fast first polls catch fast UIs,
    burst_until = t0 + 0.1                 # then fall back to the configured cadence
    motion: deque = deque()                # (time, box) of recent changed polls, for the
    waited_polls = 0                       # confined-animation detector

    while True:
        await asyncio.sleep(0.0 if waited_polls == 0 else poll)  # first poll is immediate: the
        waited_polls += 1                                        # reaction is often already there
        frame = await _maybe_await(grab())
        now = time.monotonic()

        n, box = _changed_blocks(prev, frame, cfg.pixel_delta, cfg.block, mask)
        prev = frame
        busy = bool(cfg.busy_probe and await _maybe_await(cfg.busy_probe()))
        if busy:
            last_busy = now

        first_reaction = (not reacted) and n >= cfg.react_blocks
        reacted = reacted or first_reaction
        if first_reaction or n >= cfg.activity_blocks or busy:
            last_activity = now
            quiet_polls = tiny_polls = 0
        else:
            quiet_polls += 1
            tiny_polls += 1 if n > 0 else 0

        if n > 0 and box is not None:
            motion.append((now, box))
        while motion and motion[0][0] < now - cfg.residual_bail_after:
            motion.popleft()
        motion_box = None
        if motion:
            motion_box = (min(b[0] for _, b in motion), min(b[1] for _, b in motion),
                          max(b[2] for _, b in motion), max(b[3] for _, b in motion))

        elapsed = now - t0
        if now > burst_until:
            poll = cfg.poll_interval
        residual = tiny_polls / quiet_polls if quiet_polls else 0.0

        if reacted and now - last_activity >= cfg.quiet_time:
            return SettleResult("settled", True, elapsed, residual, frame)
        if (reacted and motion_box is not None and elapsed >= cfg.residual_bail_after
                and (motion_box[2] - motion_box[0]) * (motion_box[3] - motion_box[1]) <= cfg.confined_area):
            # One confined region keeps animating (video, spinner, ticker): it will not settle,
            # so return the current frame with its bounding box instead of burning max_wait.
            return SettleResult("residual", True, elapsed, residual, frame, motion_box)
        if not reacted and not busy and elapsed >= cfg.react_deadline \
                and now - last_busy >= min(cfg.react_deadline, 0.1):
            return SettleResult("no_reaction", False, elapsed, residual, frame)
        if elapsed >= cfg.max_wait:
            spread = (motion_box is not None
                      and (motion_box[2] - motion_box[0]) * (motion_box[3] - motion_box[1]) > cfg.confined_area)
            return SettleResult("timeout", reacted, elapsed, residual, frame, motion_box, spread=spread)


async def act_and_settle(grab: Grabber, action: Callable[[], object],
                         cfg: Optional[SettleConfig] = None) -> SettleResult:
    """Capture a baseline, run the action (sync or async), then wait for the UI to settle."""
    baseline = await _maybe_await(grab())
    await _maybe_await(action())
    return await wait_settled(grab, cfg, baseline=baseline)


# --------------------------------------------------------------------------- adaptive timeouts
class LatencyBook:
    """Learns how long settling takes per key (e.g. "click") so the `max_wait` cap tracks reality.

    - Timeouts are recorded too (as the wait they hit). Learning from successes alone would let a
      key that often times out learn a short cap from its few fast runs.
    - Learning only ever shortens the cap below the configured value, and never touches fields the
      user pinned through configure() (`pinned`).
    - quiet_time is deliberately NOT learned: it protects against pauses between UI phases
      (debounced search, dialog then network), which the time-to-last-motion cannot reveal."""

    def __init__(self, keep: int = 50):
        self._samples = defaultdict(lambda: deque(maxlen=keep))

    def record(self, key: str, result: SettleResult) -> None:
        if result.reason in ("settled", "timeout"):
            self._samples[key].append(result.waited)

    def max_wait_for(self, key: str, default: float = 8.0, floor: float = 1.0,
                     factor: float = 3.0) -> float:
        xs = sorted(self._samples.get(key, ()))
        if len(xs) < 5:
            return default
        p95 = xs[min(len(xs) - 1, int(len(xs) * 0.95))]
        return max(floor, min(default, p95 * factor))

    def config_for(self, key: str, base: Optional[SettleConfig] = None,
                   pinned: frozenset = frozenset()) -> SettleConfig:
        base = base or SettleConfig()
        if "max_wait" in pinned:
            return base
        return replace(base, max_wait=self.max_wait_for(key, base.max_wait))


# --------------------------------------------------------------------------- frame sources
def make_mss_grabber(monitor: int = 1) -> Grabber:
    """Desktop capture via `pip install mss`. Returns contiguous BGRA frames, which enables the fast
    exact-compare path; drop the alpha channel (frame[:, :, :3]) only when encoding for the model."""
    import mss

    sct = (getattr(mss, "MSS", None) or mss.mss)()  # mss.mss is deprecated in newer releases

    def grab() -> Frame:
        return np.asarray(sct.grab(sct.monitors[monitor]))

    return grab


def playwright_grabber(page, quality: int = 40) -> Grabber:
    """Browser frames via Playwright (async API). Use SettleConfig(pixel_delta=12) with these frames
    to absorb JPEG noise. Screenshots cost ~50-150 ms each, so for browsers lean mainly on
    `browser_busy_probe`."""
    from PIL import Image

    async def grab() -> Frame:
        data = await page.screenshot(type="jpeg", quality=quality)
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))

    return grab


# --------------------------------------------------------------------------- browser busy signal
_BROWSER_PROBE_JS = """
(() => {
  if (window.__settle) return;
  const s = window.__settle = { inflight: 0, lastMutation: performance.now() };
  const origFetch = window.fetch;
  window.fetch = function (...args) {
    s.inflight++;
    return origFetch.apply(this, args).finally(() => { s.inflight--; });
  };
  const origSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.send = function (...args) {
    s.inflight++;
    this.addEventListener('loadend', () => { s.inflight--; }, { once: true });
    return origSend.apply(this, args);
  };
  new MutationObserver(() => { s.lastMutation = performance.now(); })
    .observe(document, { subtree: true, childList: true, attributes: true, characterData: true });
})();
"""

_BROWSER_BUSY_JS = """
(quietMs) => {
  const s = window.__settle;
  if (!s) return false;
  if (document.readyState !== 'complete') return true;
  if (s.inflight > 0) return true;
  if (performance.now() - s.lastMutation < quietMs) return true;
  // Finite animations/transitions still running. Infinite ones (CSS spinners) are ignored
  // on purpose, otherwise pages with a permanent spinner would never settle.
  return document.getAnimations().some(a => {
    if (a.playState !== 'running') return false;
    const t = a.effect && a.effect.getComputedTiming ? a.effect.getComputedTiming() : null;
    return !t || t.endTime !== Infinity;
  });
}
"""


async def install_browser_probe(page) -> None:
    """Inject the in-page tracker now and on every future navigation (Playwright async API)."""
    await page.add_init_script(_BROWSER_PROBE_JS)
    try:
        await page.evaluate(_BROWSER_PROBE_JS)
    except Exception:
        pass  # page may be mid-navigation; the init script covers the next load


def browser_busy_probe(page, mutation_quiet_ms: int = 100) -> BusyProbe:
    async def probe() -> bool:
        try:
            return bool(await page.evaluate(_BROWSER_BUSY_JS, mutation_quiet_ms))
        except Exception:
            return True  # execution context destroyed => navigation in progress

    return probe


# --------------------------------------------------------------------------- self-test
class _FakeScreen:
    """1080p synthetic screen driven by wall-clock time, for testing without a display."""
    H, W = 1080, 1920

    def __init__(self, react_at=None, animate_until=None, forever=False, caret=False,
                 glyph_at=None, box=None, flash=False):
        self.t0 = time.monotonic()
        self.react_at = react_at
        self.animate_until = animate_until
        self.forever = forever
        self.caret = caret
        self.glyph_at = glyph_at
        self.box = box  # optional (x0, y0, x1, y1) pixel region the animation moves within
        self.flash = flash  # full-screen animation: the whole screen rewrites every grab
        self.flash_phase = False

    def now(self) -> float:
        return time.monotonic() - self.t0

    def grab(self) -> Frame:
        t = self.now()
        f = np.zeros((self.H, self.W, 4), np.uint8)  # BGRA like mss
        if self.flash and self.react_at is not None and t >= self.react_at:
            end = float("inf") if self.forever else self.animate_until
            if t < end:
                self.flash_phase = not self.flash_phase  # changes on EVERY grab: cannot alias
                if self.flash_phase:
                    f[:] = 200  # whole-screen animation: nothing is ever "settled"
            return f
        if self.caret and int(t / 0.25) % 2 == 0:
            f[100:120, 100:103] = 255  # tiny blinking caret (3 cells)
        if self.react_at is not None and t >= self.react_at:
            end = float("inf") if self.forever else self.animate_until
            step = int(min(t, end) * 30)  # moves until `end`, then freezes (or runs forever)
            if self.box:
                x0, y0, x1, y1 = self.box
                x = x0 + (step * 37) % max(1, (x1 - x0) - 300)
                f[y0:y1, x:x + 300] = 200  # animation confined to a region (video/spinner)
            else:
                x = (step * 37) % (self.W - 300)
                f[400:700, x:x + 300] = 200  # moving block == animation
        if self.glyph_at is not None and t >= self.glyph_at:
            f[600:616, 300:310] = 255  # one typed character (~4 cells), appears once and stays
        return f


async def _selftest() -> int:
    failures = 0

    def check(name: str, cond: bool, res: SettleResult) -> None:
        nonlocal failures
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: reason={res.reason} "
              f"waited={res.waited:.2f}s residual={res.residual_motion:.2f}")
        failures += 0 if cond else 1

    # 1. reaction, animation, then settle (~0.6s animation end + 0.2s quiet)
    s = _FakeScreen(react_at=0.1, animate_until=0.6)
    r = await wait_settled(s.grab, SettleConfig())
    check("animation then settle", r.reason == "settled" and 0.65 <= r.waited <= 1.1, r)

    # 2. nothing happens -> "no_reaction" after react_deadline, not the full timeout
    s = _FakeScreen()
    r = await wait_settled(s.grab, SettleConfig())
    check("no reaction detected", r.reason == "no_reaction" and 0.3 <= r.waited <= 0.5, r)

    # 3. never settles -> hard timeout
    s = _FakeScreen(react_at=0.05, forever=True)
    r = await wait_settled(s.grab, SettleConfig(max_wait=1.0))
    check("endless animation hits timeout", r.reason == "timeout" and r.waited >= 1.0, r)

    # 3b. regression: a single typed character (a few cells) must count as a reaction
    s = _FakeScreen(glyph_at=0.1)
    r = await wait_settled(s.grab, SettleConfig())
    check("one typed character is detected", r.reason == "settled" and r.residual_motion == 0, r)

    # 4. blinking caret is ignored but reported as residual motion
    s = _FakeScreen(react_at=0.1, animate_until=0.3, caret=True)
    r = await wait_settled(s.grab, SettleConfig(quiet_time=0.6))
    check("caret ignored, residual reported", r.reason == "settled" and r.residual_motion > 0, r)

    # 5. baseline matters: an instant one-frame change right after the action
    def instant(screen: _FakeScreen):
        def action():
            t = screen.now()
            screen.react_at = t
            screen.animate_until = t  # static block appears immediately
        return action

    s = _FakeScreen()
    r = await act_and_settle(s.grab, instant(s), SettleConfig())
    check("baseline catches instant change", r.reason == "settled" and r.reacted, r)

    s = _FakeScreen()
    instant(s)()
    r = await wait_settled(s.grab, SettleConfig())  # no baseline -> the pitfall
    check("no baseline misses it (expected pitfall)", r.reason == "no_reaction", r)

    # 6. busy probe holds the wait while the app is "thinking" with an unchanged screen
    s = _FakeScreen(react_at=0.5, animate_until=0.5)
    cfg = SettleConfig(busy_probe=lambda: s.now() < 0.5)
    r = await wait_settled(s.grab, cfg)
    check("busy probe prevents early exit", r.reason == "settled" and r.waited >= 0.6, r)

    # 7. adaptive timeouts
    book = LatencyBook()
    for w in (0.3, 0.4, 0.35, 0.5, 0.45, 0.4):
        book.record("app:click", SettleResult("settled", True, w, 0.0, np.zeros((1, 1))))
    mw = book.max_wait_for("app:click")
    ok = 1.0 <= mw < 8.0
    print(f"[{'PASS' if ok else 'FAIL'}] adaptive max_wait: {mw:.2f}s")
    failures += 0 if ok else 1

    # 8. tolerant path (pixel_delta > 0): +-5 noise is ignored, a real change is not
    rng = np.random.default_rng(0)
    flat = np.full((1080, 1920, 3), 100, np.uint8)
    noisy = np.clip(flat.astype(np.int16) + rng.integers(-5, 6, flat.shape), 0, 255).astype(np.uint8)
    moved = flat.copy()
    moved[300:340, 300:400] = 200
    grid = _mask((1080 // 8, 1920 // 8), [])
    n_noise, box_noise = _changed_blocks(flat, noisy, 12, 8, grid)
    n_real, box_real = _changed_blocks(flat, moved, 12, 8, grid)
    ok = n_noise == 0 and box_noise is None and n_real >= 8 and box_real is not None
    print(f"[{'PASS' if ok else 'FAIL'}] pixel_delta tolerance: noise cells={n_noise}, real-change cells={n_real}")
    failures += 0 if ok else 1

    # 9. confined animation (video) -> "residual" bail with a bounding box, well before max_wait
    s = _FakeScreen(react_at=0.05, forever=True, box=(1100, 250, 1600, 650))
    r = await wait_settled(s.grab, SettleConfig(max_wait=8.0, residual_bail_after=1.5))
    ok = (r.reason == "residual" and 1.5 <= r.waited <= 3.0 and r.motion_box is not None
          and r.motion_box[0] >= 0.5 and r.reacted)
    print(f"[{'PASS' if ok else 'FAIL'}] confined animation bails early: reason={r.reason} "
          f"waited={r.waited:.2f}s box={r.motion_box}")
    failures += 0 if ok else 1

    # 9b. a full-screen animation is NOT confined: still hits the timeout, no box shortcut
    s = _FakeScreen(react_at=0.05, forever=True, flash=True)
    r = await wait_settled(s.grab, SettleConfig(max_wait=1.5, residual_bail_after=0.5))
    ok = r.reason == "timeout" and r.waited >= 1.5
    print(f"[{'PASS' if ok else 'FAIL'}] full-screen animation still times out: reason={r.reason} "
          f"waited={r.waited:.2f}s")
    failures += 0 if ok else 1

    # 10. learning must not override explicit configuration, and must not ignore timeouts
    book = LatencyBook()
    for _ in range(10):
        book.record("click", SettleResult("settled", True, 0.23, 0.0, np.zeros((1, 1))))
    user = SettleConfig(quiet_time=1.0, max_wait=30.0)
    pinned_cfg = book.config_for("click", user, pinned=frozenset({"max_wait"}))
    free_cfg = book.config_for("click", user)
    ok = (pinned_cfg.max_wait == 30.0 and pinned_cfg.quiet_time == 1.0
          and free_cfg.max_wait == 1.0 and free_cfg.quiet_time == 1.0)
    print(f"[{'PASS' if ok else 'FAIL'}] configure() pins beat learning: pinned max_wait={pinned_cfg.max_wait} "
          f"unpinned={free_cfg.max_wait} quiet_time stays {free_cfg.quiet_time}")
    failures += 0 if ok else 1

    book = LatencyBook()
    for _ in range(8):
        book.record("load", SettleResult("settled", True, 0.23, 0.0, np.zeros((1, 1))))
    before = book.max_wait_for("load")
    for _ in range(3):
        book.record("load", SettleResult("timeout", True, 1.0, 0.0, np.zeros((1, 1))))
    after = book.max_wait_for("load")
    ok = before == 1.0 and after >= 2.9
    print(f"[{'PASS' if ok else 'FAIL'}] timeouts raise the learned cap: {before:.1f}s -> {after:.1f}s")
    failures += 0 if ok else 1

    # 11. notes are worded per action kind, and timeouts report their spread
    z = np.zeros((1, 1))
    n_wait = SettleResult("no_reaction", False, 1.0, 0.0, z).note_for_model("wait")
    n_click = SettleResult("no_reaction", False, 0.3, 0.0, z).note_for_model("click")
    n_spread = SettleResult("timeout", True, 8.0, 0.0, z, (0.0, 0.0, 1.0, 1.0), spread=True).note_for_model()
    ok = ("Nothing changed" in n_wait and "missed" not in n_wait and "missed its target" in n_click
          and "spread across" in n_spread)
    print(f"[{'PASS' if ok else 'FAIL'}] per-kind notes and timeout spread")
    failures += 0 if ok else 1

    # 12. a timeout reports where the motion is (this was silently dropped before)
    s = _FakeScreen(react_at=0.05, forever=True, flash=True)
    r = await wait_settled(s.grab, SettleConfig(max_wait=1.0))
    ok = r.reason == "timeout" and r.motion_box is not None and r.spread
    print(f"[{'PASS' if ok else 'FAIL'}] timeout carries motion box: box={r.motion_box} spread={r.spread}")
    failures += 0 if ok else 1

    print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Event-driven UI settle-wait for computer-use agents")
    ap.add_argument("--selftest", action="store_true", help="run synthetic-screen tests")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(1 if asyncio.run(_selftest()) else 0)
    ap.print_help()


if __name__ == "__main__":
    main()

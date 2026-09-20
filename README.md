# settled-computer

**Fast computer use for AI agents.** If your computer-use agent types `time.sleep(2)` between
every click, re-screenshots after every action, and burns vision tokens re-reading an unchanged
screen — that's the problem this fixes.

An MCP server for desktop computer use with **event-driven settling**: every action waits
until the screen actually reacts and stops changing, then returns the settled screenshot and
a one-line verdict. No fixed sleeps, no separate screenshot round trips, no model-side guessing
about *when* to look.

Built for agent loops (Hermes, Claude Desktop, Claude Code, any MCP host). The core idea:
**the server answers "did the UI react, and has it stopped?" at 20 Hz with a ~1 ms frame diff,
so the model doesn't need an extra vision turn just to find out when it is safe to look.**
Whether the reaction was the *right* one is still the model's job: judge by the image.

## Why your computer-use agent is slow (and how this fixes it)

Naive computer-use loops look like this:

```
click -> guess a sleep -> screenshot (full vision turn) -> hope
```

Timings are guesses: too short means acting on half-loaded UIs, too long means wasted seconds.
Each "is it ready yet?" costs a model turn and a screenshot that enters context.

settled-computer replaces that with:

```
click -> server watches the screen -> returns the AFTER frame + verdict
```

- Reacted and settled in 0.4 s → `Screen settled 0.40s after the action.`
- Nothing changed → `No visible change…` (it may simply have had no visible effect: an
  already-focused field, an already-set state). If the model received an image within the last
  45 s and the screen is unchanged, the image is omitted to save vision tokens.
- One region keeps moving (video/spinner) → bails at ~1.5 s with the region's bounding box
  instead of stalling 8 s, and after two consecutive bails the region is auto-ignored.

## Tools

| tool | what it does |
|---|---|
| `act` | **The main one.** Runs 1–8 actions in ONE call: `[{"click":[x,y]}, {"type":"text"}, {"key":"enter"}]`. **Every step is validated before anything runs** (a typo in step 5 cannot leave steps 1–4 half-executed). Each step settles before the next. A timeout always stops the sequence; "no visible change" stops it only for click/dblclick/rightclick/key (`stop_on_no_reaction="auto"`, the default). Use `"never"` when a no-op is expected (e.g. the field is already focused) or `"always"` to stop on any no-op. The stop message lists what ran, what did not, and the exact recovery hint. `screenshot="none"` returns zero images on a clean run. |
| `screenshot` | Look at the screen. **Always returns an image.** Every action already returns the settled frame, so this is for "I have no recent frame". |
| `click` / `type_text` / `press_key` / `scroll` / `drag` / `mouse_move` | Single actions, each returning the settled screen + note. |
| `wait` | Wait for a slow operation to finish changing. |
| `configure` | Tune `quiet_time`, `react_deadline`, `max_wait`, `residual_bail_after`, `ignore_regions`. **Values you set are pinned**: adaptive learning and per-action defaults never override them. `ignore_regions=[]` also clears auto-detected regions. |
| `screen_info` | Image size (coordinates are pixels of THAT image), monitor geometry, platform, auto-ignored regions. |

The server's MCP `instructions` field ships the usage contract to every client, including the
caveat that "no visible change" is not the same as "failed".

## How settling works

1. Grab a **baseline** frame before the action (fast reactions are never missed).
2. Perform the action (on a dedicated input thread, so a long drag or paste never blocks the loop).
3. Poll frames (burst: first poll immediately, 15 ms cadence for 100 ms, then 20 Hz),
   diffing full-resolution in 8×8-pixel cells (uint32 exact compare, ~1 ms per static 1080p frame).
4. Wait for: reaction (any change) → stillness (`quiet_time`) → return.

### Adaptive `max_wait` (and why `quiet_time` is not learned)

`LatencyBook` learns a per-action `max_wait` cap (3× the p95 of past runs, floor 1 s). It records
**timeouts too**, so an action that often times out cannot learn a short cap from its few fast
runs, and it **never overrides a value pinned through `configure`**.

`quiet_time` is deliberately *not* learned. It exists to bridge pauses *between* UI phases
(debounced search, dialog then network fetch), and the time-to-last-motion that a learner can
observe says nothing about those pauses. A learned value would ratchet toward "fast but wrong"
with no feedback signal.

### Config precedence

`configure()`-pinned values › per-action defaults (`type`: quiet 0.12 s / react 0.2 s,
`scroll`: quiet 0.15 s, `hover`: react 0.2 s) › adaptive learning (`max_wait` only).

## Design decisions

- **One lock around every tool.** Hosts may issue parallel tool calls; without serialization
  their settle windows overlap and each verdict is contaminated by the other action.
- **"No image = unchanged" has a time limit** (`SETTLE_MCP_ELIDE_TTL`, default 45 s, `0` disables).
  The server cannot know what is still in the model's context (new chat on a long-lived server,
  hosts that prune old images, compaction), so `screenshot()` never omits an image and action
  results omit one only while the last image the model received is fresh.
- **Auto-ignore with expiry.** After two consecutive residual bails on overlapping regions the
  region is ignored for `SETTLE_MCP_AUTO_IGNORE_SECS` (default 30 s), then re-checked; if it is
  still animating it is re-activated immediately, if not it simply expires.
- **A spinner is not decoration.** The residual note says everything outside the region is stable
  and tells the model to call `wait()` if that region is what it is waiting for.

## Comparison with alternatives

| | settled-computer | naive sleep loop | native per-call drivers (e.g. cua-driver) |
|---|---|---|---|
| waits between actions | measured (frame diff, 20 Hz) | guessed `time.sleep` | none — model re-screenshots to check |
| images per action | 1 (0 when unchanged) | 1–3 | 1 + verification shots |
| model turns for 3 actions | **1** (`act()` batch) | 6+ | 7+ |
| learns your machine | yes (per-action `max_wait`) | no | no |
| video/spinner handling | 1.5 s bail + region hint | stalls forever | stalls or false-settles |

Not a replacement for the action layer (it *uses* pyautogui) — it replaces the guesswork
around it. Works alongside any MCP host: Claude Desktop, Claude Code, Hermes, custom loops.

## Safety — read this first

This server **moves your mouse and types on your real desktop**. It is a prompt-injection
surface: any text, webpage, popup, or document visible on screen can instruct the model
driving it to click and type on your behalf. Treat screen contents as untrusted input.

- **Tool allowlist**: `SETTLE_MCP_TOOLS=screenshot,screen_info,wait` strips all input
  injection (click/type/keys/scroll/drag/act are refused with a clear error). The single
  most effective hardening when you don't need full control.
- Keep the **pyautogui fail-safe on** (default): slam the mouse into the top-left corner to abort.
  The next tool call reports it as a readable error instead of a crash.
- Prefer a VM or a dedicated user session.
- `act()` validates every step before running and halts on timeouts and (by default) on
  click/key no-ops, so a mid-sequence mis-click surfaces instead of compounding.
- macOS needs Screen Recording + Accessibility permission for the host app; Linux needs X11
  (Wayland blocks capture and synthetic input).

## Install

```bash
pip install "settled-computer[desktop]"          # capture + input (what the MCP server needs)
pip install "settled-computer[desktop,fast]"     # + opencv: ~3x faster JPEG encode
pip install "settled-computer[desktop,unicode]"  # + non-ASCII typing (clipboard paste)
```

Extras: `fast` = `opencv-python`, `unicode` = `pyperclip`, `desktop` = `mss`/`pyautogui`/`pillow`.
The core (just `numpy` + `mcp`) is enough to run the engine self-test or browser automation
via Playwright grabbers.

**Linux:** `pip` cannot install `python3-tk`, but pyautogui exits without it:

```bash
sudo apt install python3-tk          # Debian/Ubuntu
```

**Platform status (0.x alpha):** Windows 10/11 is the primary, measured target (numbers
below). Linux/X11 works and is Xvfb-tested in CI; Wayland blocks capture and synthetic
input. **macOS is unverified** — the code path is expected to work with Screen Recording +
Accessibility permissions granted to the host app, but no measured numbers exist yet.
Hence 0.x alpha.

Works with mcp 1.x (`FastMCP`) and 2.x (`MCPServer`).

```bash
settled-computer --check     # verify capture + coordinates (console script)
settled-computer             # serve MCP over stdio
python settled_computer/engine.py --selftest   # 16 synthetic-screen tests, no display needed
```

### Register with an MCP host

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "settled-computer": {
      "command": "settled-computer",
      "args": []
    }
  }
}
```

Claude Code:

```bash
claude mcp add settled-computer -- settled-computer
```

Hermes (`config.yaml`):

```yaml
mcp_servers:
  settled-computer:
    command: <python-or-venv-path>
    args: ["-m", "settled_computer.server"]
```

Running from a source checkout instead of an install? `python settle_mcp.py` still works
(shim into the package).

### Environment variables

| var | default | meaning |
|---|---|---|
| `SETTLE_MCP_MONITOR` | `1` | monitor index (mss numbering, 1 = primary) |
| `SETTLE_MCP_MAX_WIDTH` | `1280` | max width of returned images (px) |
| `SETTLE_MCP_QUALITY` | `70` | JPEG quality of returned images |
| `SETTLE_MCP_FAILSAFE` | `1` | `0` disables pyautogui's fail-safe |
| `SETTLE_MCP_ELIDE_TTL` | `45` | seconds an unchanged screen may omit its image (`0` = never omit) |
| `SETTLE_MCP_AUTO_IGNORE_SECS` | `30` | how long an auto-detected animating region is ignored before re-checking (`0` = off) |
| `SETTLE_MCP_TOOLS` | *(all)* | allowlist; e.g. `screenshot,screen_info,wait` = observation only, input tools refused |

## Notes the model sees

Every action returns the settled image (unless elided, see above) plus a short note:

- `Screen settled 0.38s after the action.` — proceed.
- `No visible change within 0.30s… It may simply have had no visible effect…` — judge by the image;
  the click may have been fine (already focused / already set) or may have missed.
- `Screen settled 1.51s after the action, except a small region [x0,y0,x1,y1] that keeps animating…`
  + either an auto-ignore confirmation or the exact `configure(ignore_regions=…)` call.
- `Screen was still changing after 8.0s (timeout).` — loading; `wait()` then re-check.
  Includes the motion bounding box when motion is spread across the screen.
- Responses with **no image** say so explicitly and how old the last image is.
- `act` failures: `STOPPED at step N (kind): … Done before it: … Not run: … <recovery hint>`.

## Known limits

- A pixel change is not proof of success: a wrong click that opens the wrong dialog also "settles".
- A blinking caret can register as a reaction, so "no visible change" is less reliable in text fields.
- Pixel-only settling cannot tell "app is thinking" from "app is done" on a static screen.
- Confined-region detection cannot distinguish a video from a progress indicator; the note says so.
- Adaptive `max_wait` is keyed by action type, not by application.

## Performance (Windows 10, 1366×768, as measured by the author)

| operation | settled-computer | native per-call driver |
|---|---|---|
| cheap round trip | **38–55 ms** (persistent stdio) | 340–405 ms (process spawn) |
| screen capture | 16 ms (mss) | 387–963 ms |
| frame diff | 1–5 ms | n/a (model compares screenshots) |
| 3-action sequence | **1 call, 1 image** | ≥7 model turns, ≥4 screenshots |

The last row reflects `act` batching, not settling: any driver could batch. Run your own:
`python bench_mcp.py` (MCP round trips) and `python bench_native.py` (cua-driver).
`python find_motion.py` locates what keeps changing on your screen (e.g. to pick `ignore_regions`).

## Safety

This server lets a model move your mouse and type on your real desktop.

- Keep the **pyautogui fail-safe on** (default): slam the mouse into the top-left corner to abort.
  The next tool call reports it as a readable error instead of a crash.
- Prefer a VM or a dedicated user session.
- `act()` validates every step before running and halts on timeouts and (by default) on
  click/key no-ops, so a mid-sequence mis-click surfaces instead of compounding.
- macOS needs Screen Recording + Accessibility permission for the host app; Linux needs X11
  (Wayland blocks capture and synthetic input).

## Files

```
settled_computer/
├── engine.py        Settle engine: wait_settled, act_and_settle, LatencyBook, selftest (no MCP dep)
└── server.py        MCP server: 11 tools, encoding, act() sequencer, tool allowlist
settle.py            shim -> settled_computer.engine (old checkouts)
settle_mcp.py        shim -> settled_computer.server (old MCP registrations)
bench_mcp.py         MCP round-trip benchmark (persistent stdio client)
bench_native.py      cua-driver benchmark (subprocess per call, for comparison)
find_motion.py       Locate perpetually-animating screen regions (ignore_regions picker)
```

## License

MIT (see `LICENSE`). Chosen so MCP hosts and agent distributions can bundle it freely;
any future paid tier will be an open-core split (hosted/managed features around the same
open server), not a relicense.

"""MCP-side benchmark: settled-computer over stdio (newline-delimited JSON framing)."""
import json, statistics, subprocess, threading, time, sys

SETTLE = [r"C:\Users\X1\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe",
          r"C:\Users\X1\Desktop\fast-computer-use-mcp\settle_mcp.py"]
N = 7

proc = subprocess.Popen(SETTLE, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL)
lines = []
def reader():
    for line in iter(proc.stdout.readline, b""):
        lines.append(line)
threading.Thread(target=reader, daemon=True).start()

_id = 0
def rpc(method, params=None, timeout=60):
    global _id
    _id += 1
    msg = {"jsonrpc": "2.0", "id": _id, "method": method}
    if params is not None: msg["params"] = params
    proc.stdin.write((json.dumps(msg) + "\n").encode()); proc.stdin.flush()
    want = _id
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        for line in lines:
            try: resp = json.loads(line)
            except Exception: continue
            if resp.get("id") == want:
                lines.clear()
                ms = (time.perf_counter() - t0) * 1000
                if "error" in resp: raise RuntimeError(resp["error"])
                return resp["result"], ms
        time.sleep(0.001)
    raise TimeoutError(method)

rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "bench", "version": "0"}})
proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n").encode())
proc.stdin.flush()
time.sleep(0.5); lines.clear()

def bench(label, name, args, n=N):
    ts = []
    for _ in range(n):
        _, ms = rpc("tools/call", {"name": name, "arguments": args}); ts.append(ms)
        print(f"  run {len(ts)}: {ms:6.0f} ms", flush=True)
    print(f"{label:46s} min {min(ts):6.0f}  avg {statistics.mean(ts):6.0f}  max {max(ts):6.0f} ms\n", flush=True)

print(f"=== settled-computer MCP ({N} runs each) ===", flush=True)
bench("screenshot (capture + settle)", "screenshot", {})
bench("screen_info (cheap call)", "screen_info", {})
bench("mouse_move 680,384 (hover + settle)", "mouse_move", {"x": 680, "y": 384})
proc.kill()
print("done")

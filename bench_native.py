"""Native cua-driver benchmark (subprocess per call, like an agent loop would)."""
import json, statistics, subprocess, time

CUA = r"C:\Users\X1\AppData\Local\Programs\Cua\cua-driver\bin\cua-driver.EXE"
N = 7

def call(tool, args=None):
    cmd = [CUA, "call", tool]
    if args: cmd += ["--args", json.dumps(args)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    ms = (time.perf_counter() - t0) * 1000
    if r.returncode != 0:
        raise RuntimeError(f"{tool}: rc={r.returncode} {r.stderr[:200]}")
    return ms, r.stdout

def bench(label, tool, args=None, n=N, outfile=None):
    ts = []
    for i in range(n):
        if outfile and i == n - 1:  # last run writes the file so we can verify content
            args = dict(args or {}, screenshot_out_file=outfile)
        ms, out = call(tool, args); ts.append(ms)
        print(f"  run {len(ts)}: {ms:6.0f} ms", flush=True)
    print(f"{label:46s} min {min(ts):6.0f}  avg {statistics.mean(ts):6.0f}  max {max(ts):6.0f} ms\n", flush=True)

print(f"=== native cua-driver ({N} runs each) ===", flush=True)
bench("get_screen_size (cheap call)", "get_screen_size")
bench("get_cursor_position (cheap call)", "get_cursor_position")
bench("get_desktop_state (full capture)", "get_desktop_state",
      outfile=r"C:\Users\X1\Desktop\fast-computer-use-mcp\bench_native.png")
print("done")

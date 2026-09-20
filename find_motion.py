"""Find what keeps changing on the screen: diff frames at 20 Hz, locate hot cells."""
import numpy as np, time
import mss

sct = (getattr(mss, "MSS", None) or mss.mss)()
mon = sct.monitors[1]
def grab():
    return np.asarray(sct.grab(mon))

BLOCK = 8
prev = grab()
h, w = prev.shape[:2]
hb, wb = h // BLOCK, w // BLOCK
grid_shape = (hb, wb)
heat = np.zeros(grid_shape, np.int32)
polls = 0
changed_polls = 0
per_poll = []

t_end = time.monotonic() + 4.0
while time.monotonic() < t_end:
    time.sleep(0.05)
    cur = grab()
    polls += 1
    if cur.shape != prev.shape:
        print("resolution changed!"); prev = cur; continue
    changed = (prev.view(np.uint32).reshape(h, w) != cur.view(np.uint32).reshape(h, w))
    prev = cur
    if not changed.any():
        per_poll.append(0); continue
    changed_polls += 1
    g = changed[:hb*BLOCK, :wb*BLOCK].reshape(hb, BLOCK, wb, BLOCK).any(axis=(1, 3))
    heat += g
    per_poll.append(int(g.sum()))

print(f"polls={polls} polls_with_change={changed_polls} ({changed_polls/polls:.0%})")
print("blocks changed per poll: min", min(per_poll), "max", max(per_poll),
      "median", sorted(per_poll)[len(per_poll)//2])
# hottest cells -> screen coords
ys, xs = np.where(heat > polls * 0.3)
if len(ys):
    print(f"hot cells (>30% of polls): {len(ys)} cells")
    print(f"  x range {xs.min()*BLOCK}-{(xs.max()+1)*BLOCK}px, y range {ys.min()*BLOCK}-{(ys.max()+1)*BLOCK}px")
    print(f"  as fractions: x {xs.min()*BLOCK/w:.2f}-{(xs.max()+1)*BLOCK/w:.2f}, y {ys.min()*BLOCK/h:.2f}-{(ys.max()+1)*BLOCK/h:.2f}")
else:
    ys2, xs2 = np.where(heat > 0)
    if len(ys2):
        print(f"no persistent hotspots; {len(ys2)} cells changed at least once")
        print(f"  x range {xs2.min()*BLOCK}-{(xs2.max()+1)*BLOCK}px, y range {ys2.min()*BLOCK}-{(ys2.max()+1)*BLOCK}px")
    else:
        print("screen is fully static")

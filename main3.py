import random
import time
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

# ---------------------------
# Controls
# ---------------------------
rows = 71
cols = 81
loops = 230
delay_s = 0.0
RENDER_OVERHEAD_S = 0.0 

# Which heuristics to run (booleans)
RUN_MANHATTAN = True
RUN_EUCLIDEAN = True
RUN_OCTILE    = True
RUN_ZERO      = True   # Dijkstra baseline


# ---------------------------
# Helpers
# ---------------------------

#Heuristic Management
def selected_heuristics_from_flags():
    selected = []
    if RUN_MANHATTAN: selected.append("manhattan")
    if RUN_EUCLIDEAN: selected.append("euclidean")
    if RUN_OCTILE:    selected.append("octile")
    if RUN_ZERO:      selected.append("zero")
    return selected

def _run_one_window(grid, start, goal, heuristic, delay_s, verbose):
    # Each process opens its own window
    animate(grid, start, goal, delay_s=delay_s, heuristic=heuristic, verbose=verbose)

def run_heuristics_parallel(grid, start, goal, heuristics, delay_s, verbose=False):
    """
    True parallel windows: one process per heuristic.
    """
    procs = []
    for h in heuristics:
        p = Process(target=_run_one_window, args=(grid, start, goal, h, delay_s, verbose))
        p.start()
        procs.append(p)

    # Keep parent alive until all windows closed
    for p in procs:
        p.join()

# Reports Helpers
def estimate_animation_time(frames_count, delay_s, render_overhead_s=0.0):
    """
    Instant estimate:
    total ≈ frames * (delay + render_overhead)
    render_overhead_s = average GUI draw cost per frame (optional).
    """
    return frames_count * (delay_s + render_overhead_s)

def calibrate_render_overhead(fig, im, frames, sample=30):
    """
    Measures average GUI draw overhead per frame using a small sample.
    """
    sample = min(sample, len(frames))
    t0 = time.perf_counter()
    for i in range(sample):
        im.set_data(frames[i])
        fig.canvas.draw()   # forces render
        fig.canvas.flush_events()
    t1 = time.perf_counter()
    return (t1 - t0) / sample

# ---------------------------
# Maze generation (perfect maze) + optional loops
# ---------------------------
def generate_maze(rows, cols, loops, seed=None):
    if rows % 2 == 0 or cols % 2 == 0:
        raise ValueError("Use odd rows/cols (e.g., 31x51).")

    rng = random.Random(seed)
    grid = np.ones((rows, cols), dtype=np.uint8)

    start_cell = (1, 1)
    grid[start_cell] = 0
    stack = [start_cell]
    dirs = [(-2,0), (2,0), (0,-2), (0,2)]

    # randomized DFS carving
    while stack:
        r, c = stack[-1]
        rng.shuffle(dirs)
        carved = False
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < rows-1 and 1 <= nc < cols-1 and grid[nr, nc] == 1:
                grid[nr, nc] = 0
                grid[r + dr//2, c + dc//2] = 0
                stack.append((nr, nc))
                carved = True
                break
        if not carved:
            stack.pop()

    # add loops (remove some walls between corridors)
    candidates = []
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if grid[r, c] == 1:
                if grid[r, c-1] == 0 and grid[r, c+1] == 0:
                    candidates.append((r, c))
                elif grid[r-1, c] == 0 and grid[r+1, c] == 0:
                    candidates.append((r, c))

    rng.shuffle(candidates)
    for (r, c) in candidates[:min(loops, len(candidates))]:
        grid[r, c] = 0

    start = (1, 1)
    goal = (rows-2, cols-2)
    if grid[goal] == 1:
        # find nearest open cell near bottom-right
        for rr in range(rows-2, 0, -1):
            for cc in range(cols-2, 0, -1):
                if grid[rr, cc] == 0:
                    goal = (rr, cc)
                    break
            else:
                continue
            break

    return grid, start, goal

# ---------------------------
# Heuristics
# ---------------------------
def h_manhattan(n, goal):
    return abs(n[0] - goal[0]) + abs(n[1] - goal[1])

def h_euclidean(n, goal):
    return math.hypot(n[0] - goal[0], n[1] - goal[1])

def h_octile(n, goal):
    # Octile distance is usually for 8-neighbor grids,
    # but it still underestimates on 4-neighbor grids (admissible, just less sharp than Manhattan).
    dx = abs(n[0] - goal[0])
    dy = abs(n[1] - goal[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

def h_zero(n, goal):
    # Equivalent to Dijkstra (no heuristic guidance)
    return 0.0

HEURISTICS = {
    "manhattan": h_manhattan,
    "euclidean": h_euclidean,
    "octile": h_octile,
    "zero": h_zero,  # Dijkstra baseline
}

# ---------------------------
# A* with trace (open/closed/current) + final path at end
# ---------------------------

def reconstruct(came_from, cur):
    path = [cur]
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path

def is_adjacent(a, b):
    """4-neighborhood adjacency check."""
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1

def astar_trace(grid, start, goal, heuristic="manhattan", verbose=False, log_limit=200):
    rows, cols = grid.shape

    # Allow heuristic="manhattan" or heuristic=function
    if isinstance(heuristic, str):
        h_fn = HEURISTICS[heuristic]
        h_name = heuristic
    else:
        h_fn = heuristic
        h_name = getattr(heuristic, "__name__", "custom")

    def nbrs(r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                yield (nr, nc)

    open_heap = []
    came = {}
    g = {start: 0}
    h0 = float(h_fn(start, goal))
    best_f = {start: h0}
    closed = set()
    open_set = {start}

    counter = 0
    heapq.heappush(open_heap, (best_f[start], h0, counter, start))

    # --- protocol state ---
    prev_expanded = None
    expanded_count = 0

    # if verbose:
    #     print(f"\nA* OPEN PROTOCOL (heuristic={h_name})")
    #     print("Format: [k] node  g  h     f      | note")
    #     print("-" * 85)

    while open_heap:
        f, hn, _, cur = heapq.heappop(open_heap)

        # Skip stale heap entries
        if f != best_f.get(cur, None):
            continue
        if cur in closed:
            continue

        open_set.discard(cur)
        closed.add(cur)

        # # --- protocol logging on EXPANSION ---
        # if verbose and expanded_count < log_limit:
        #     gc = g[cur]
        #     note = ""
        #     if prev_expanded is not None and not is_adjacent(prev_expanded, cur):
        #         note = f"--- JUMP --- from {prev_expanded} to {cur}"
        #     print(f"[{expanded_count:03d}] {cur}  g={gc:3d}  h={hn:6.2f}  f={f:7.2f}   {note}")
        #     prev_expanded = cur
        #     expanded_count += 1
        # elif verbose and expanded_count == log_limit:
        #     print(f"... log_limit reached ({log_limit}). Suppressing further lines.")
        #     expanded_count += 1
        # # ---
        yield cur, set(open_set), set(closed), None

        if cur == goal:
            path = reconstruct(came, cur)
            if verbose:
                print("-" * 85)
                print(f"SOLVED: path_len={len(path)-1}, expanded={len(closed)}")
            yield cur, set(open_set), set(closed), path
            return

        for nxt in nbrs(*cur):
            tg = g[cur] + 1
            if tg < g.get(nxt, float("inf")):
                came[nxt] = cur
                g[nxt] = tg

                hn2 = float(h_fn(nxt, goal))
                fn = tg + hn2

                best_f[nxt] = fn
                counter += 1
                heapq.heappush(open_heap, (fn, hn2, counter, nxt))
                if nxt not in closed:
                    open_set.add(nxt)

    yield None, set(open_set), set(closed), None

# ---------------------------
# Reports
# ---------------------------
def run_astar_diagnostics(grid, start, goal, heuristic="manhattan", delay_s=0.01, render_overhead_s=0.0):
    """
    Runs A* once WITHOUT animation to measure pure compute time + stats.
    Adds:
      - frames_est (how many generator steps happened)
      - time_per_step_s
      - time_per_expansion_s
      - anim_est_s (frames_est * (delay + render_overhead))
    """
    free_cells = int(np.sum(grid == 0))

    t0 = time.perf_counter()

    expanded = 0
    discovered_set = set([start])
    max_open = 1
    final_path = None
    frames_est = 0  # number of yields from astar_trace (proxy for animation frames)

    for cur, open_nodes, closed_nodes, path in astar_trace(grid, start, goal, heuristic=heuristic):
        frames_est += 1
        expanded = len(closed_nodes)
        max_open = max(max_open, len(open_nodes))
        discovered_set |= open_nodes
        discovered_set |= closed_nodes

        if path is not None:
            final_path = path
            break

    t1 = time.perf_counter()
    solve_time_no_anim = t1 - t0

    # basic timing derivations
    time_per_step_s = solve_time_no_anim / max(1, frames_est)
    time_per_expansion_s = solve_time_no_anim / max(1, expanded)
    anim_est_s = frames_est * (delay_s + render_overhead_s)

    if final_path is None:
        return {
            "solved": False,
            "solve_time_no_anim_s": solve_time_no_anim,
            "frames_est": frames_est,
            "time_per_step_s": time_per_step_s,
            "time_per_expansion_s": time_per_expansion_s,
            "anim_est_s": anim_est_s,
            "free_cells": free_cells,
            "expanded": expanded,
            "discovered": len(discovered_set),
            "pct_searched_expanded": (expanded / free_cells * 100.0) if free_cells else 0.0,
            "max_open": max_open,
            "path_len": None,
            "efficiency_expanded_per_step": None,
        }

    path_len = len(final_path) - 1
    pct_searched = (expanded / free_cells * 100.0) if free_cells else 0.0
    efficiency = (expanded / path_len) if path_len > 0 else None

    return {
        "solved": True,
        "solve_time_no_anim_s": solve_time_no_anim,
        "frames_est": frames_est,
        "time_per_step_s": time_per_step_s,
        "time_per_expansion_s": time_per_expansion_s,
        "anim_est_s": anim_est_s,
        "free_cells": free_cells,
        "expanded": expanded,
        "discovered": len(discovered_set),
        "pct_searched_expanded": pct_searched,
        "max_open": max_open,
        "path_len": path_len,
        "efficiency_expanded_per_step": efficiency,
    }
# ---------------------------
# Animation
# ---------------------------
def animate(grid, start, goal, delay_s=0.01, heuristic="manhattan", verbose=False):
    rows, cols = grid.shape

    # --- diagnostics (algorithm time, no animation) ---
    stats = run_astar_diagnostics(grid, start, goal, heuristic=heuristic, delay_s=delay_s)

    base = np.zeros((rows, cols), dtype=np.uint8)
    base[grid == 1] = 1

    cmap = ListedColormap([
        "#f7f7f7", "#111111", "#7aa6ff", "#7f7f7f",
        "#5bd67a", "#ffb347", "#ff4d4d", "#8a2be2"
    ])
    norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), cmap.N)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.5])

    ax = fig.add_subplot(gs[0, 0])
    ax_report = fig.add_subplot(gs[0, 1])
    ax_report.axis("off")

    # --- layout tweaks: more left, smaller title ---
    fig.subplots_adjust(left=0.015, right=0.98, top=0.92, bottom=0.06, wspace=0.15)

    ax.set_title(f"A* on Random Maze ({heuristic})", fontsize=10, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    im = ax.imshow(base, cmap=cmap, norm=norm, interpolation="nearest")

    # ---- BUILD FRAMES (timed) ----
    t_build0 = time.perf_counter()

    frames = []
    final_path = None

    for cur, open_nodes, closed_nodes, path in astar_trace(
        grid, start, goal, heuristic=heuristic, verbose=verbose, log_limit=250
    ):
        frame = base.copy()

        for r, c in closed_nodes: frame[r, c] = 3
        for r, c in open_nodes:   frame[r, c] = 2

        if path is not None:
            final_path = path

        if final_path is not None:
            for r, c in final_path: frame[r, c] = 4

        if cur is not None:
            frame[cur] = 5

        frame[start] = 6
        frame[goal]  = 7
        frames.append(frame)

    t_build1 = time.perf_counter()
    build_time_s = t_build1 - t_build0

    frames_count = len(frames)
    est_anim_time = estimate_animation_time(frames_count, delay_s, render_overhead_s=RENDER_OVERHEAD_S)

    # ---- REPORT TEXT (placeholder for playback time) ----
    if stats["solved"]:
            def lr(label, value, width=34):
                left = f"{label}:"
                right = str(value)
                if len(left) + len(right) + 1 > width:
                    # if too long, just fall back (prevents ugly negative spacing)
                    return f"{left} {right}"
                return left + " " * (width - len(left) - len(right)) + right
            
            W = 25  # wider/narrower report width
            report = (
                "A* Diagnostics\n"
                "-------------------------\n"
                + lr("Solved", "YES" if stats["solved"] else "NO", W) + "\n"
                + lr("Solve time", f"{stats['solve_time_no_anim_s']:.6f} s", W) + "\n"
                + lr("Est. playback", f"{est_anim_time:.2f} s", W) + "\n"
                + lr("Playback time", "(measuring...)", W) + "\n"
                + lr("Frames (est)", stats["frames_est"], W) + "\n"
                + lr("Anim est", f"{stats['anim_est_s']:.2f} s", W) + "\n"
                + lr("t / frame", f"{stats['time_per_step_s']:.6f} s", W) + "\n"
                + lr("t / expand", f"{stats['time_per_expansion_s']:.6f} s", W) + "\n"
                + lr("Free cells", stats["free_cells"], W) + "\n"
                + lr("Expanded", stats["expanded"], W) + "\n"
                + lr("Discovered", stats["discovered"], W) + "\n"
                + lr("% searched", f"{stats['pct_searched_expanded']:.2f}%", W) + "\n"
                + lr("Path length", stats["path_len"], W) + "\n"
                + lr("Max OPEN", stats["max_open"], W) + "\n"
                + lr("Expand/step", f"{stats['efficiency_expanded_per_step']:.2f}", W) + "\n"
            )

    txt = fig.text(
    0.985, 0.92, report,          # x close to right edge, y near top
    ha="right", va="top",         # anchor the BLOCK to the right edge
    fontsize=9,
    family="monospace"
    )

    # ---- REAL PLAYBACK TIMING (measured inside GUI loop) ----
    playback_t0 = None
    playback_t1 = None
    last_i = len(frames) - 1

    def update(i):
        nonlocal playback_t0, playback_t1, report

        if i == 0 and playback_t0 is None:
            playback_t0 = time.perf_counter()

        im.set_data(frames[i])

        if i == last_i and playback_t1 is None:
            playback_t1 = time.perf_counter()
            playback_time_s = playback_t1 - playback_t0

            # Update report text in-place
            report2 = report.replace(
                "Playback time:        (measuring...)",
                f"Playback time:        {playback_time_s:.3f} s"
            )
            txt.set_text(report2)
            fig.canvas.draw_idle()

        return (im,)

    # Keep FuncAnimation (the part that worked for you)
    anim = FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=int(delay_s * 1000),
        blit=True,
        repeat=False
    )

    # IMPORTANT: do NOT call tight_layout() here; it fights subplots_adjust and can break layout/backends
    plt.show()
# ---------------------------
# Controller: run one or many heuristics
# ---------------------------
def run_heuristics(grid, start, goal, heuristics=("manhattan",), mode="windows", delay_s=0.01, verbose=False):
    """
    heuristics: tuple/list of heuristic names, e.g. ("manhattan","euclidean","octile","zero")
    mode: "windows" -> one window per heuristic (simple)
          "subplots" -> one figure with multiple panels (compare side-by-side, no report panels)
    """
    heuristics = list(heuristics)

    if mode == "windows":
        for h in heuristics:
            animate(grid, start, goal, delay_s=delay_s, heuristic=h, verbose=verbose)

    # elif mode == "subplots":
    #     n = len(heuristics)
    #     fig = plt.figure(figsize=(6*n, 6))
    #     gs = fig.add_gridspec(1, n)

    #     cmap = ListedColormap([
    #         "#f7f7f7", "#111111", "#7aa6ff", "#7f7f7f",
    #         "#5bd67a", "#ffb347", "#ff4d4d", "#8a2be2"
    #     ])
    #     norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), cmap.N)

    #     # precompute base once
    #     base = np.zeros_like(grid, dtype=np.uint8)
    #     base[grid == 1] = 1

    #     # store animations to avoid garbage collection
    #     anims = []

    #     for i, h in enumerate(heuristics):
    #         ax = fig.add_subplot(gs[0, i])
    #         ax.set_title(h, fontsize=10)
    #         ax.set_xticks([]); ax.set_yticks([])
    #         ax.set_aspect("equal")
    #         im = ax.imshow(base, cmap=cmap, norm=norm, interpolation="nearest")

    #         frames = []
    #         final_path = None

    #         for cur, open_nodes, closed_nodes, path in astar_trace(grid, start, goal, heuristic=h, verbose=False):
    #             frame = base.copy()
    #             for r, c in closed_nodes: frame[r, c] = 3
    #             for r, c in open_nodes:   frame[r, c] = 2
    #             if path is not None:
    #                 final_path = path
    #             if final_path is not None:
    #                 for r, c in final_path: frame[r, c] = 4
    #             if cur is not None:
    #                 frame[cur] = 5
    #             frame[start] = 6
    #             frame[goal]  = 7
    #             frames.append(frame)

    #         def make_update(local_im, local_frames):
    #             def update(k):
    #                 local_im.set_data(local_frames[k])
    #                 return (local_im,)
    #             return update

    #         anim = FuncAnimation(
    #             fig, make_update(im, frames),
    #             frames=len(frames),
    #             interval=int(delay_s * 1000),
    #             blit=True,
    #             repeat=False
    #         )
    #         anims.append(anim)

    #     plt.tight_layout()
    #     plt.show()

    else:
        raise ValueError("mode must be 'windows' or 'subplots'")

# ---------------------------
# Run (NEW maze every run)
# ---------------------------
if __name__ == "__main__":
    grid, start, goal = generate_maze(rows, cols, loops, seed=None)

    heuristics_to_run = selected_heuristics_from_flags()
    if not heuristics_to_run:
        raise ValueError("No heuristics selected. Set at least one RUN_* flag to True.")

    # Parallel windows (real simultaneous runs)
    run_heuristics_parallel(grid, start, goal, heuristics_to_run, delay_s=delay_s, verbose=False)

    # sequential heuretics instead:
    # run_heuristics(grid, start, goal, heuristics=heuristics_to_run, mode="windows", delay_s=delay_s, verbose=False)
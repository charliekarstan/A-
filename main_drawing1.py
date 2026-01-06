# main4_editor.py
import time
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Button

import main3 as core   # <-- this is your main4.py

# ---------------------------
# Controls
# ---------------------------
rows = 21
cols = 31
delay_s = 0.0   

UI_FPS = 60
STOPWATCH_FPS = 30

# ---------------------------
# A* 
# ---------------------------
def h_manhattan(n, goal):
    return abs(n[0] - goal[0]) + abs(n[1] - goal[1])

def reconstruct(came_from, cur):
    path = [cur]
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path

def astar_trace(grid, start, goal):
    """
    Improved A*:
    - neighbor ordering: smaller h first
    - tie-breaker: prefer larger g when f ties (push -g)
    Heap tuple: (f, h, -g, counter, node)
    """
    R, C = grid.shape
    DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

    def nbrs_ordered(r, c):
        cand = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr, nc] == 0:
                cand.append((nr, nc))
        cand.sort(key=lambda x: h_manhattan(x, goal))
        return cand

    open_heap = []
    open_set = {start}
    closed = set()

    came = {}
    g = {start: 0}

    h0 = float(h_manhattan(start, goal))
    f0 = h0
    best_f = {start: f0}

    counter = 0
    heapq.heappush(open_heap, (f0, h0, 0, counter, start))

    while open_heap:
        f, hn, neg_g, _, cur = heapq.heappop(open_heap)

        if f != best_f.get(cur, None):
            continue
        if cur in closed:
            continue

        open_set.discard(cur)
        closed.add(cur)

        yield cur, set(open_set), set(closed), None

        if cur == goal:
            path = reconstruct(came, cur)
            yield cur, set(open_set), set(closed), path
            return

        gc = g[cur]
        for nxt in nbrs_ordered(*cur):
            tg = gc + 1
            if tg < g.get(nxt, float("inf")):
                came[nxt] = cur
                g[nxt] = tg

                hn2 = float(h_manhattan(nxt, goal))
                fn = tg + hn2
                best_f[nxt] = fn

                counter += 1
                heapq.heappush(open_heap, (fn, hn2, -tg, counter, nxt))
                if nxt not in closed:
                    open_set.add(nxt)

    yield None, set(open_set), set(closed), None

# OPTIONAL but smart: make core use the same A* if you call core.* later
core.astar_trace = astar_trace
core.h_manhattan = h_manhattan
core.reconstruct = reconstruct

# ---------------------------
# Drawing helper: Bresenham
# ---------------------------
def bresenham_cells(a, b):
    (r0, c0) = a
    (r1, c1) = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1

    r, c = r0, c0
    cells = [(r, c)]

    if dc > dr:
        err = dc // 2
        while c != c1:
            c += sc
            err -= dr
            if err < 0:
                r += sr
                err += dc
            cells.append((r, c))
    else:
        err = dr // 2
        while r != r1:
            r += sr
            err -= dc
            if err < 0:
                c += sc
                err += dr
            cells.append((r, c))

    return cells

# ---------------------------
# Editor + Player UI
# (your class unchanged below, except it calls astar_trace defined above)
# ---------------------------
class MazeEditor:
    def __init__(self, rows, cols, delay_s=0.0):
        self.rows = rows
        self.cols = cols
        self.delay_s = delay_s

        self.grid = np.zeros((rows, cols), dtype=np.uint8)
        self._force_border_walls()

        self.start = (1, 1)
        self.goal = (rows - 2, cols - 2)

        self.mode = "wall"
        self.is_dragging = False
        self.last_cell = None

        self.view = None
        self.dirty = False

        self.anim = None
        self.frames = []
        self.cur_cells = []
        self.goal_frame_idx = None
        self._goal_stop_frozen = False

        self.play_t0 = None
        self._last_sw_update = 0.0

        self._build_figure()
        self._start_ui_timer()

    def _force_border_walls(self):
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

    def _stop_anim(self):
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
            self.anim = None

    def _is_border(self, r, c):
        return r == 0 or c == 0 or r == self.rows - 1 or c == self.cols - 1

    def _make_cmap(self):
        cmap = ListedColormap([
            "#f7f7f7", "#111111", "#7aa6ff", "#7f7f7f",
            "#5bd67a", "#ffb347", "#ff4d4d", "#8a2be2"
        ])
        norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), cmap.N)
        return cmap, norm

    def _compose_base(self):
        base = np.zeros((self.rows, self.cols), dtype=np.uint8)
        base[self.grid == 1] = 1
        base[self.start] = 6
        base[self.goal] = 7
        return base

    def _mark_dirty(self):
        self.dirty = True

    def _flush_if_dirty(self):
        if self.dirty:
            self._force_border_walls()
            self.grid[self.start] = 0
            self.grid[self.goal] = 0
            self.view = self._compose_base()
            self.im.set_data(self.view)
            self.dirty = False
            self.fig.canvas.draw_idle()

    def _build_figure(self):
        cmap, norm = self._make_cmap()

        self.fig = plt.figure(figsize=(12, 7))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[4.8, 1.2])

        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_ui = self.fig.add_subplot(gs[0, 1])
        self.ax_ui.axis("off")

        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.16, wspace=0.10)

        self.ax.set_title("Maze Editor - A* Manhattan (core imported)", fontsize=11, pad=8)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_aspect("equal")

        self.view = self._compose_base()
        self.im = self.ax.imshow(self.view, cmap=cmap, norm=norm, interpolation="nearest")

        self.status = self.ax_ui.text(
            0.02, 0.98, self._status_text(),
            ha="left", va="top", fontsize=9, family="monospace",
            transform=self.ax_ui.transAxes
        )
        self.stopwatch = self.ax_ui.text(
            0.02, 0.35, "Stopwatch\n---------\n0.000 s",
            ha="left", va="top", fontsize=10, family="monospace",
            transform=self.ax_ui.transAxes
        )

        self._add_buttons_bottom()

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _start_ui_timer(self):
        interval_ms = max(1, int(1000 / UI_FPS))
        self.ui_timer = self.fig.canvas.new_timer(interval=interval_ms)
        self.ui_timer.add_callback(self._flush_if_dirty)
        self.ui_timer.start()

    def _add_buttons_bottom(self):
        y, h = 0.04, 0.07
        x0, w, gap = 0.06, 0.13, 0.012

        def add_btn(ix, label, cb):
            axb = self.fig.add_axes([x0 + ix*(w+gap), y, w, h])
            b = Button(axb, label)
            b.on_clicked(cb)
            return b

        self.btn_wall  = add_btn(0, "Draw WALL", self._set_mode_wall)
        self.btn_erase = add_btn(1, "ERASE", self._set_mode_erase)
        self.btn_start = add_btn(2, "Set START", self._set_mode_start)
        self.btn_goal  = add_btn(3, "Set GOAL", self._set_mode_goal)
        self.btn_play  = add_btn(4, "PLAY", self._play)
        self.btn_clear = add_btn(5, "Clear Solution", self._clear_solution_overlay)
        self.btn_all   = add_btn(6, "Clear ALL", self._clear_all)

    def _status_text(self):
        return (
            "Controls\n--------\n"
            f"Mode:  {self.mode}\n"
            f"Start: {self.start}\n"
            f"Goal:  {self.goal}\n\n"
            "Mouse\n-----\n"
            "Click+Drag = paint\n"
            "Border locked\n\n"
            "Run\n---\n"
            "PLAY runs A* Manhattan\n"
        )

    def _refresh_status(self):
        self.status.set_text(self._status_text())
        self._mark_dirty()

    def _set_mode_wall(self, _=None):
        self.mode = "wall"; self._refresh_status()

    def _set_mode_erase(self, _=None):
        self.mode = "erase"; self._refresh_status()

    def _set_mode_start(self, _=None):
        self.mode = "set_start"; self._refresh_status()

    def _set_mode_goal(self, _=None):
        self.mode = "set_goal"; self._refresh_status()

    def _event_to_cell(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        c = int(math.floor(event.xdata + 0.5))
        r = int(math.floor(event.ydata + 0.5))
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (r, c)
        return None

    def _apply_cell(self, r, c):
        if self._is_border(r, c):
            return
        if (r, c) == self.start or (r, c) == self.goal:
            return

        if self.mode == "wall":
            self.grid[r, c] = 1
        elif self.mode == "erase":
            self.grid[r, c] = 0
        elif self.mode == "set_start":
            if self.grid[r, c] == 0 and (r, c) != self.goal:
                self.start = (r, c)
        elif self.mode == "set_goal":
            if self.grid[r, c] == 0 and (r, c) != self.start:
                self.goal = (r, c)

    def _apply_line(self, a, b):
        for (r, c) in bresenham_cells(a, b):
            self._apply_cell(r, c)
        self._mark_dirty()

    def _on_press(self, event):
        if event.button != 1 or event.inaxes != self.ax:
            return
        cell = self._event_to_cell(event)
        if cell is None:
            return
        self.is_dragging = True
        self.last_cell = cell
        self._apply_line(cell, cell)
        self._refresh_status()

    def _on_release(self, event):
        self.is_dragging = False
        self.last_cell = None

    def _on_motion(self, event):
        if not self.is_dragging:
            return
        cell = self._event_to_cell(event)
        if cell is None or self.last_cell is None:
            return
        if cell != self.last_cell:
            self._apply_line(self.last_cell, cell)
            self.last_cell = cell

    def _clear_solution_overlay(self, _=None):
        self._stop_anim()

        self.frames = []
        self.cur_cells = []
        self.goal_frame_idx = None
        self._goal_stop_frozen = False

        self._mark_dirty()
        self.play_t0 = None
        self._last_sw_update = 0.0
        self.stopwatch.set_text("Stopwatch\n---------\n0.000 s")
        self.fig.canvas.draw_idle()

    def _clear_all(self, _=None):
        self._stop_anim()

        self.grid[:, :] = 0
        self._force_border_walls()
        self.start = (1, 1)
        self.goal = (self.rows - 2, self.cols - 2)
        self._clear_solution_overlay()
        self._refresh_status()

    def _set_stopwatch(self, seconds):
        self.stopwatch.set_text(f"Stopwatch\n---------\n{seconds:0.3f} s")

    def _play(self, _=None):
        if self.anim is not None:
            try: self.anim.event_source.stop()
            except Exception: pass
            self.anim = None

        self._goal_stop_frozen = False
        self.play_t0 = None
        self._last_sw_update = 0.0
        self._set_stopwatch(0.0)

        if self.grid[self.start] == 1 or self.grid[self.goal] == 1:
            print("Start/Goal is on a wall. Fix it.")
            return

        base = self._compose_base()
        frames = []
        goal_hit_idx = None

        for cur, open_nodes, closed_nodes, path in astar_trace(self.grid, self.start, self.goal):
            frame = base.copy()

            for r, c in closed_nodes:
                if (r, c) not in (self.start, self.goal): frame[r, c] = 3
            for r, c in open_nodes:
                if (r, c) not in (self.start, self.goal): frame[r, c] = 2

            if path is not None:
                for r, c in path:
                    if (r, c) not in (self.start, self.goal): frame[r, c] = 4

            if cur is not None and cur not in (self.start, self.goal):
                frame[cur] = 5

            frame[self.start] = 6
            frame[self.goal]  = 7

            frames.append(frame)

            if goal_hit_idx is None and cur == self.goal:
                goal_hit_idx = len(frames) - 1

        if not frames:
            return

        self.frames = frames
        self.goal_frame_idx = goal_hit_idx if goal_hit_idx is not None else (len(frames) - 1)
        interval_ms = max(1, int(self.delay_s * 1000))

        def update(i):
            if not self.frames or i >= len(self.frames):
                return (self.im,)

            now = time.perf_counter()
            if self.play_t0 is None:
                self.play_t0 = now

            self.im.set_data(self.frames[i])

            if (not self._goal_stop_frozen) and (i >= self.goal_frame_idx):
                self._goal_stop_frozen = True
                self._set_stopwatch(now - self.play_t0)
                self.fig.canvas.draw_idle()
                return (self.im,)

            if not self._goal_stop_frozen and (now - self._last_sw_update) >= (1.0 / STOPWATCH_FPS):
                self._last_sw_update = now
                self._set_stopwatch(now - self.play_t0)

            return (self.im,)

        self.anim = FuncAnimation(self.fig, update, frames=len(self.frames),
                                  interval=interval_ms, blit=True, repeat=False)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

if __name__ == "__main__":
    app = MazeEditor(rows=rows, cols=cols, delay_s=delay_s)
    app.show()
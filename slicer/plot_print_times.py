#!/usr/bin/env python3
"""
Compute print times from the printer path JSON files and show two figures:

  1) Max print time, normalized to the 1-printhead time (dashed line = 1/N)
     (was plot_max_print_times.py)
  2) Min / max print-time ratio per printhead count
     (was print_time_var.py)

Print times are computed in memory from the printer JSON files with the same
timing model as gcode_generator.py (print / jog / rotation speeds below).
Nothing is written to disk unless you pass --save (which saves only the two PDFs).

Usage:
    python plot_print_times.py
    python plot_print_times.py --jobs bunny heart benchy cowboy duck
    python plot_print_times.py --data-dir ../results/shells --no-legend
    python plot_print_times.py --save          # also write print_times.pdf / print_time_var.pdf

Where the data is found (both layouts work):
    <data>/<job>_<N>_printers_.../*.json
    <data>/<job>/<anything>n<N>/*.json
Default data folder: whichever of the current folder, ../results/shells,
results/shells or the script's own folder contains the most of the jobs.

Requires: numpy, matplotlib, cylinder_fitting (pip install cylinder_fitting)
"""

import argparse
import glob
import json
import os
import re
import sys
import warnings
from itertools import groupby
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
JOBS = ["bunny", "heart", "benchy", "duck"]
# Legend names (jobs not listed here are shown with a capitalized folder name)
LABELS = {"bunny": "Bunny", "heart": "Heart", "benchy": "Benchy", "duck": "Duck", "cowboy": "Cowboy"}
COLORS = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
MARKERS = ["o", "s", "^", "D"]
PRINTHEAD_COUNTS = [1, 2, 4, 6, 8, 14]
FIGSIZE = (4.8, 5)
MARKERSIZE = 9            # was 15
LINEWIDTH = 1
XLABEL = "Number of printheads"
YLABEL_PRINT_TIME = "Relative\nprint time (%)"   # \n = line break, as in the paper figure
YLABEL_IDLE = "Idle\nefficiency (%)"

# Timing model (same as gcode_generator.py)
PRINT_SPEED = 5
JOG_SPEED = 20
ROT_SPEED = 200


# ---------------------------------------------------------------------------
# Print-time computation (from gcode_generator.py, without saving/animation)
# ---------------------------------------------------------------------------
def angle_around_axis(center, axis, p1, p2):
    """Unsigned angle (degrees) from p1 to p2 seen along the axis through center."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    v1 = np.asarray(p1) - np.asarray(center)
    v2 = np.asarray(p2) - np.asarray(center)
    v1_proj = v1 - np.dot(v1, axis) * axis
    v2_proj = v2 - np.dot(v2, axis) * axis
    e1 = v1_proj / np.linalg.norm(v1_proj)
    e2 = np.cross(axis, e1)
    x = np.dot(v2_proj, e1)
    y = np.dot(v2_proj, e2)
    return np.rad2deg(np.abs(np.arctan2(y, x)))


def find_consecutive_print_nodes(inds):
    conseq_nodes = []
    for _, g in groupby(enumerate(inds), lambda i_x: i_x[0] - i_x[1]):
        nodes = list(map(itemgetter(1), g))
        nodes.append(nodes[-1] + 1)
        conseq_nodes.append(nodes)
    return conseq_nodes


def import_cylinder_fit():
    """Import cylinder_fitting.fit without its plotting module.

    cylinder_fitting/__init__.py imports its own visualize.py, which calls
    matplotlib.use('TkAgg') and would hijack the plot windows. The fit itself
    doesn't need it, so a stand-in module is registered first.
    """
    import types
    if "cylinder_fitting.visualize" not in sys.modules:
        stub = types.ModuleType("cylinder_fitting.visualize")
        stub.show_fit = stub.show_G_distribution = lambda *a, **k: None
        sys.modules["cylinder_fitting.visualize"] = stub
    try:
        from cylinder_fitting import fit
    except ImportError:
        sys.exit("Error: the cylinder_fitting package is required: pip install cylinder_fitting")
    return fit


class PrintTimer:
    """Computes the total print time of one printer file.

    Mirrors gcode_generator.py exactly, including that the last cylinder fit
    (axis, centre, radius) carries over to the next file if a file's first
    printed level has no segment that can be fitted.
    """

    def __init__(self):
        self.fit = import_cylinder_fit()
        self.w_local = self.C_local = self.r_local = None

    def total_time(self, data):
        t_global = []
        flag = 0
        axis_center = axis_direction = None
        r_prev = None
        prev_level_end = None  # last point of the previous printed level

        for n in data["levels"]:
            max_length = 0
            print_paths_local = []
            path = np.array(data["levels"][n]["path"])
            mask = np.array(data["levels"][n]["mask"])
            inds = np.where(mask)[0]
            for nodes in find_consecutive_print_nodes(inds):
                pts = path[nodes, :]
                dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                length = np.cumsum(dists)[-1]
                if length < 1:
                    continue
                print_paths_local.append(pts)
                if flag == 0:
                    try:
                        w_fit, C_fit, r_fit, _ = self.fit(pts)
                    except Exception:
                        continue
                    if length > max_length and pts.shape[0] > 3:
                        max_length = length
                        self.w_local, self.C_local, self.r_local = w_fit, C_fit, r_fit

            if len(print_paths_local) == 0:
                continue

            if flag == 0:
                if self.C_local is None:  # nothing fitted yet: skip this level (original behaviour)
                    continue
                axis_center = self.C_local
                axis_direction = self.w_local
                r_prev = self.r_local
                flag = 1
            else:
                dth = angle_around_axis(axis_center, axis_direction,
                                        prev_level_end, print_paths_local[0][0, :])
                t_global.append(5 / JOG_SPEED + dth / ROT_SPEED
                                + (5 + (self.r_local - r_prev)) / JOG_SPEED)
                r_prev = self.r_local

            for j, pts in enumerate(print_paths_local):
                if j > 0:
                    dth = angle_around_axis(axis_center, axis_direction,
                                            print_paths_local[j - 1][-1, :], pts[0, :])
                    t_global.append(2 * 5 / JOG_SPEED + dth / ROT_SPEED)
                length = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
                t_global.append(length / PRINT_SPEED)
            prev_level_end = print_paths_local[-1][-1, :]

        return float(np.sum(t_global))


# ---------------------------------------------------------------------------
# Finding the data
# ---------------------------------------------------------------------------
def app_dir():
    """Directory of the running script, or of the .exe when packaged."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def find_runs(root, job):
    """Return {n_printheads: folder} for one job, in either folder layout."""
    runs = {}
    for sub in glob.glob(os.path.join(root, glob.escape(job) + "_*_printers*")):
        m = re.match(re.escape(job) + r"_(\d+)_printers", os.path.basename(sub))
        if m and os.path.isdir(sub):
            runs[int(m.group(1))] = sub
    for sub in glob.glob(os.path.join(root, glob.escape(job), "*")):
        m = re.search(r"n(\d+)$", os.path.basename(sub))
        if m and os.path.isdir(sub):
            runs.setdefault(int(m.group(1)), sub)
    return {n: f for n, f in runs.items() if glob.glob(os.path.join(f, "*.json"))}


def pick_data_dir(data_dir, jobs):
    if data_dir:
        if not os.path.isdir(data_dir):
            sys.exit(f"Error: folder not found: {os.path.abspath(data_dir)}")
        return data_dir
    candidates = []
    for base in (os.getcwd(), app_dir()):
        candidates += [base, os.path.join(base, "..", "results", "shells"),
                       os.path.join(base, "results", "shells")]
    best, best_count = None, 0
    for c in candidates:
        if os.path.isdir(c):
            count = sum(bool(find_runs(c, j)) for j in jobs)
            if count > best_count:
                best, best_count = c, count
    if best is None:
        sys.exit(f"Error: couldn't find printer JSON files for {', '.join(jobs)}. "
                 "Use --data-dir to point at the folder that holds the jobs.")
    return best


def compute_times(data_dir, jobs, counts):
    """Return (t_max, t_min) in seconds, shape (jobs, counts); NaN where data is missing."""
    timer = PrintTimer()
    t_max = np.full((len(jobs), len(counts)), np.nan)
    t_min = np.full((len(jobs), len(counts)), np.nan)
    for i, job in enumerate(jobs):
        runs = find_runs(data_dir, job)
        if not runs:
            print(f"Warning: no data for '{job}'")
            continue
        for j, n in enumerate(counts):
            folder = runs.get(n)
            if folder is None:
                print(f"Warning: no {n}-printhead run for '{job}'")
                continue
            files = sorted(glob.glob(os.path.join(folder, "*.json")))
            times = []
            for k, file in enumerate(files, 1):
                print(f"\r  {job:<10} {n:>2} printheads: file {k}/{len(files)}", end="", flush=True)
                with open(file, "r") as f:
                    data = json.load(f)
                times.append(timer.total_time(data))
            t_max[i, j], t_min[i, j] = max(times), min(times)
            print(f"\r  {job:<10} {n:>2} printheads: max {t_max[i, j] / 60:8.1f} min, "
                  f"min {t_min[i, j] / 60:8.1f} min  ({len(files)} files)")
    return t_max, t_min


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def style_axes(ax, ylabel):
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(-10, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xticks(range(2, int(ax.get_xlim()[1]) + 1, 2))
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(ylabel)
    ax.figure.tight_layout()  # keep the two-line y label inside the window
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_series(ax, x, ys, jobs):
    for i, y in enumerate(ys):
        ok = ~np.isnan(y)
        ax.plot(x[ok], y[ok],
                c=COLORS[i % len(COLORS)],
                linewidth=LINEWIDTH,
                linestyle="-",
                marker=MARKERS[i % len(MARKERS)],
                markersize=MARKERSIZE,
                fillstyle="none",
                zorder=len(jobs) + 1 - i,
                label=LABELS.get(jobs[i], jobs[i].capitalize()))


def print_stats(title, matrix, counts):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # columns that are all NaN
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
    print(f"\n{title}")
    print("  printheads: " + "  ".join(f"{n:>6}" for n in counts))
    print("  mean:       " + "  ".join(f"{m:6.3f}" for m in mean))
    print("  std:        " + "  ".join(f"{s:6.3f}" for s in std))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", nargs="+", default=JOBS, help=f"Jobs to plot (default: {' '.join(JOBS)})")
    p.add_argument("--printheads", nargs="+", type=int, default=PRINTHEAD_COUNTS,
                   help="Printhead counts on the x-axis (default: 1 2 4 6 8 14)")
    p.add_argument("--data-dir", help="Folder containing the job folders")
    p.add_argument("--no-legend", action="store_true", help="Hide the legend")
    p.add_argument("--save", action="store_true",
                   help="Also save print_times.pdf and print_time_var.pdf in the current folder")
    args = p.parse_args()

    counts = np.array(sorted(args.printheads))
    data_dir = pick_data_dir(args.data_dir, args.jobs)
    print(f"Data folder: {os.path.abspath(data_dir)}")
    print("Computing print times...")
    t_max, t_min = compute_times(data_dir, args.jobs, counts)

    # --- Figure 1: max print time normalized to the smallest printhead count ---
    normalized = t_max / t_max[:, [0]]
    fig1, ax1 = plt.subplots(figsize=FIGSIZE, num="Relative print time")
    plot_series(ax1, counts, 100 * normalized, args.jobs)
    ax1.plot(counts, 100 * counts[0] / counts, "k--", label="1/N")
    style_axes(ax1, YLABEL_PRINT_TIME)
    print_stats(f"Max print time / {counts[0]}-printhead time", normalized[:, 1:], counts[1:])

    # --- Figure 2: min / max print-time ratio ---
    ratio = t_min / t_max
    fig2, ax2 = plt.subplots(figsize=FIGSIZE, num="Idle efficiency")
    plot_series(ax2, counts, 100 * ratio, args.jobs)
    style_axes(ax2, YLABEL_IDLE)
    print_stats("Min / max print time", ratio, counts)

    if not args.no_legend:
        # each figure is shown on its own, so each gets its own legend (top right)
        ax1.legend(loc="upper right", frameon=False, handlelength=2.5)
        ax2.legend(loc="upper right", frameon=False, handlelength=2.5)

    if args.save:
        fig1.savefig("print_times.pdf", format="pdf")
        fig2.savefig("print_time_var.pdf", format="pdf")
        print("\nSaved print_times.pdf and print_time_var.pdf")

    plt.show(block=True)


if __name__ == "__main__":
    main()
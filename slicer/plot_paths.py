#!/usr/bin/env python3
r"""
Visualize multi-printhead print paths with PyVista (print paths only - no STL,
no matplotlib). Nothing is written to disk.

All jobs live in one folder, ../results/shells by default. Both layouts are found there:
    <shells>/<job>_<N>_printers_cylindrical_volume/printer_*.json   (main.py output)
    <shells>/<job>/<anything>n<N>/*.json                           (benchy, bunny, ...)

Examples:
    python view_printheads.py                      # lists the jobs and asks
    python view_printheads.py bunny 4
    python view_printheads.py hollow_cube 1 --show-travel --show-printheads --printer-id 0
    python view_printheads.py bunny 4 --results D:/other/shells

Or reuse your main.py command - just swap main.py for view_printheads.py:
    python view_printheads.py --job_names duck
        --printer_profiles ../printer_profiles/6_printer_profile_cylindrical_contour_1mm.json
        --load_data_path ../results/shells
  The printer count comes from the profile name (6_printer_profile... -> 6), like main.py.
  Several --job_names / --printer_profiles open one window after another.
  Options only main.py uses (--stl_names, --obj_scale, --load_data, --visualize_result,
  --velocity ...) are ignored, so a full main.py command can be pasted as is.

If ../results/shells doesn't exist, results/shells, the current folder and the
script's folder are searched.
"""

import argparse
import glob
import json
import os
import re
import sys
from itertools import groupby
from operator import itemgetter

import numpy as np
import pyvista as pv

VERSION = "2026-09-22 (main.py-style commands, print paths only)"

# ---------------------------------------------------------------------------
# Settings - edit these to change how the paths are drawn
# ---------------------------------------------------------------------------

DEFAULT_LINEWIDTH = 2          # line width for every job, unless --line-width is given
TRAVEL_LINEWIDTH_SCALE = 0.5   # travel moves (--show-travel) = this x the print line width

# Palettes keyed by number of printheads (number of JSON files)
PALETTES = {
    1: ["#5E4B8B"],
    2: ["#5E4B8B", "#ED9271"],
    4: ["#E3C84B", "#62A860", "#CE5A57", "#3F6DB5"],
    6: ["#5E4B8B", "#ED9271", "#E3C84B", "#62A860", "#CE5A57", "#3F6DB5"],
    8: ["#011959", "#D9381E", "#4F734C", "#114160", "#6FBDB4", "#A3B3E3", "#226061", "#FCC3DF"],
    14: ["#5E4B8B", "#ED9271", "#6FBDB4", "#A3B3E3", "#226061", "#FCC3DF", "#E3C84B",
         "#62A860", "#CE5A57", "#3F6DB5", "#011959", "#D9381E", "#4F734C", "#114160"],
}


def get_colors(n):
    """Return n colors: the fixed palette if one exists, otherwise sample a colormap."""
    if n in PALETTES:
        return PALETTES[n]
    if n <= len(PALETTES[14]):
        return PALETTES[14][:n]
    import colorsys  # evenly spaced hues for very large printer counts
    return ["#%02x%02x%02x" % tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / n, 0.65, 0.85))
            for i in range(n)]


def find_consecutive_print_nodes(inds):
    """Group mask indices into runs of consecutive printed nodes."""
    conseq_nodes = []
    for _, g in groupby(enumerate(inds), lambda i_x: i_x[0] - i_x[1]):
        nodes = list(map(itemgetter(1), g))
        nodes.append(nodes[-1] + 1)
        conseq_nodes.append(nodes)
    return conseq_nodes


DEFAULT_RESULTS = os.path.join("..", "results", "shells")


def app_dir():
    """Directory of the running script, or of the .exe when packaged."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def candidate_roots(results):
    """Folders to search for jobs, in order of preference."""
    if results:
        return [results]
    # works whether run from the repo root or from a subfolder next to results/
    roots = [os.path.join(base, rel)
             for base in (os.getcwd(), app_dir())
             for rel in (DEFAULT_RESULTS, os.path.join("results", "shells"))]
    roots += [os.getcwd(), app_dir()]
    seen, out = set(), []
    for r in roots:
        r = os.path.normpath(os.path.abspath(r))
        if r not in seen and os.path.isdir(r):
            seen.add(r)
            out.append(r)
    return out


def scan_jobs(root):
    """Return {job_name: {n_printers: folder}} for both folder layouts in root:
         <root>/<job>_<N>_printers_cylindrical_volume/printer_*.json   (main.py output)
         <root>/<job>/<anything>n<N>/*.json                           (original datasets)
    """
    jobs = {}
    for sub in glob.glob(os.path.join(root, "*")):
        if not os.path.isdir(sub):
            continue
        base = os.path.basename(sub)
        m = re.match(r"(.+)_(\d+)_printers", base)
        if m and glob.glob(os.path.join(sub, "*.json")):
            jobs.setdefault(m.group(1), {})[int(m.group(2))] = sub
            continue
        for run in glob.glob(os.path.join(sub, "*")):
            m = re.search(r"n(\d+)$", os.path.basename(run))
            if m and os.path.isdir(run) and glob.glob(os.path.join(run, "*.json")):
                jobs.setdefault(base, {})[int(m.group(1))] = run
    return jobs


def describe(jobs):
    return "\n".join(f"  {name:<25} printers: {', '.join(str(n) for n in sorted(runs))}"
                     for name, runs in sorted(jobs.items()))


def find_job_folder(results, job, n_printers):
    """Resolve (job, n_printers) to the folder holding its printer JSON files."""
    # --results may point straight at one job folder
    if results and os.path.isdir(results) and glob.glob(os.path.join(results, "*.json")):
        return results, os.path.basename(os.path.normpath(results))

    roots = candidate_roots(results)
    if not roots:
        where = results or DEFAULT_RESULTS
        sys.exit(f"Error: folder not found: {os.path.abspath(where)}")

    # merge jobs across roots; earlier roots win
    jobs = {}
    for root in reversed(roots):
        for name, runs in scan_jobs(root).items():
            jobs.setdefault(name, {}).update(runs)
    if not jobs:
        sys.exit("Error: no print jobs found in: " + ", ".join(roots))

    if job is None:
        print("Available jobs:\n" + describe(jobs))
        while job not in jobs:
            job = input("Job: ").strip()
            if job not in jobs:
                matches = [j for j in jobs if j.lower() == job.lower()]
                job = matches[0] if matches else job
    elif job not in jobs:
        matches = [j for j in jobs if j.lower() == job.lower()]
        if not matches:
            sys.exit(f"Error: job '{job}' not found. Available jobs:\n" + describe(jobs))
        job = matches[0]

    runs = jobs[job]
    opts = ", ".join(str(k) for k in sorted(runs))
    if n_printers is None:
        if len(runs) == 1:
            return next(iter(runs.values())), job
        while n_printers not in runs:
            try:
                n_printers = int(input(f"Number of printers for '{job}' ({opts}): ").strip())
            except ValueError:
                pass
    if n_printers not in runs:
        sys.exit(f"Error: no {n_printers}-printer run for '{job}'. Available: {opts}")
    return runs[n_printers], job


def sorted_json_files(folder):
    files = glob.glob(os.path.join(folder, "*.json"))
    if all(re.search(r"\d+\.json$", os.path.basename(f)) for f in files):
        return sorted(files, key=printer_index)
    return sorted(files)


def build_printhead_mesh(data, printing=True):
    """Combine all printed (or, with printing=False, travel) segments of one
    printhead into a single PolyData. mask[i] refers to the edge path[i] -> path[i+1]."""
    all_pts, lines = [], []
    offset = 0
    path_length = 0.0

    for level in data["levels"].values():
        path = np.asarray(level["path"])
        mask = np.asarray(level["mask"]).astype(bool)
        inds = np.where(mask if printing else ~mask)[0]
        if inds.size == 0:
            continue
        for nodes in find_consecutive_print_nodes(inds):
            nodes = [n for n in nodes if n < len(path)]
            if len(nodes) < 2:
                continue
            pts = path[nodes, :]
            path_length += np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
            all_pts.append(pts)
            lines.extend([len(pts), *range(offset, offset + len(pts))])
            offset += len(pts)

    if not all_pts:
        return None, 0.0
    # pass lines= directly: PolyData(points) alone adds a vertex (dot) at every point
    mesh = pv.PolyData(np.vstack(all_pts), lines=np.array(lines))
    return mesh, path_length


def printer_index(path):
    m = re.search(r"(\d+)(?=\.json$)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def add_printhead_marker(plotter, info, color, size):
    """Draw a printer's origin and the direction of its normal."""
    origin = np.asarray(info.get("origin", [0, 0, 0]), dtype=float)
    normal = np.asarray(info.get("normal", [0, 0, -1]), dtype=float)
    plotter.add_mesh(pv.Sphere(radius=size * 0.03, center=origin), color=color)
    plotter.add_mesh(pv.Arrow(start=origin, direction=normal, scale=size * 0.25), color=color)


def render(json_files, title, line_width, printer_ids=None, show_travel=False,
           show_printheads=False):
    if printer_ids:
        json_files = [f for f in json_files if printer_index(f) in printer_ids]
        if not json_files:
            sys.exit(f"Error: none of printer id(s) {printer_ids} found.")

    colors = get_colors(len(json_files))
    plotter = pv.Plotter(title=title)

    print(f"\n{title}")
    total = 0.0
    markers = []
    for k, file in enumerate(json_files):
        with open(file, "r") as f:
            data = json.load(f)
        data.pop("shell info", None)  # large and unused here

        mesh, length = build_printhead_mesh(data)
        total += length
        name = os.path.splitext(os.path.basename(file))[0]
        print(f"  {name:<30} {colors[k]}  path length: {length:.2f}")
        if mesh is not None:
            plotter.add_mesh(mesh, color=colors[k], line_width=line_width, label=name)
        if show_travel:
            travel, _ = build_printhead_mesh(data, printing=False)
            if travel is not None:
                plotter.add_mesh(travel, color="grey", opacity=0.5,
                                 line_width=max(line_width * TRAVEL_LINEWIDTH_SCALE, 1))
        if show_printheads and "printer info" in data:
            markers.append((data["printer info"], colors[k]))
    print(f"  {'Total':<30} {'':7}  path length: {total:.2f}\n")

    if markers:
        size = plotter.length if plotter.length > 0 else 1.0
        for info, color in markers:
            add_printhead_marker(plotter, info, color, size)

    plotter.add_legend(bcolor=None, face="line")
    plotter.show_axes()
    plotter.show()


def visualize(job=None, n_printers=None, results=None,
              line_width=None, **opts):
    folder, job = find_job_folder(results, job, n_printers)
    json_files = sorted_json_files(folder)
    if not json_files:
        sys.exit(f"Error: no JSON files in {folder}")
    n = len(json_files)
    title = f"{job} - {n} printer{'s' if n != 1 else ''}  ({os.path.basename(os.path.normpath(folder))})"
    render(json_files, title, line_width or DEFAULT_LINEWIDTH, **opts)


def strip_unknown_options(argv, known):
    """Drop options this viewer doesn't use (e.g. main.py's --load_data, --visualize_result
    matplotlib, --stl_names x.stl) together with their values, so a main.py command can be reused."""
    out, skipping = [], False
    for tok in argv:
        if tok.startswith("--"):
            skipping = tok.split("=", 1)[0] not in known
            if skipping:
                continue
        elif skipping:
            continue  # value belonging to an ignored option
        out.append(tok)
    return out


def printers_from_profile(profile):
    """main.py reads the printer count from the profile file name, e.g. 4_printer_profile_... -> 4."""
    try:
        name = re.split(r"[\\/]", profile)[-1]  # handles / and \ paths on any OS
        return int(name.split("_")[0])
    except ValueError:
        sys.exit(f"Error: can't read a printer count from profile name: {profile}")


def main():
    parser = argparse.ArgumentParser(description="Visualize multi-printhead print paths.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("job", nargs="?",
                        help="Job/shape name, e.g. hollow_cube or bunny (prompted if omitted)")
    parser.add_argument("n_printers", nargs="?", type=int,
                        help="Number of printers (prompted if the job has several)")
    parser.add_argument("--results", "--load_data_path", "--data-dir", dest="results",
                        help=f"Folder holding the jobs (default: {DEFAULT_RESULTS}), "
                             "or a single job folder")
    parser.add_argument("--job", "--job_names", dest="jobs", nargs="+",
                        help="Job name(s); several open one after another")
    parser.add_argument("--printers", dest="printers_opt", type=int,
                        help="Same as the n_printers argument")
    parser.add_argument("--printer_profiles", nargs="+",
                        help="main.py printer profile(s); the printer count is read from the name")
    parser.add_argument("--show-printheads", action="store_true",
                        help="Draw each printer's origin and normal")
    parser.add_argument("--printer-id", "--vis_printer_id", dest="printer_id", nargs="+", type=int,
                        help="Only show these printer ids")
    parser.add_argument("--show-travel", action="store_true", help="Also draw travel moves")
    parser.add_argument("--line-width", type=float, help="Line width")

    print(f"view_printheads.py version {VERSION}")
    known = {s for a in parser._actions for s in a.option_strings}
    argv = sys.argv[1:]
    cleaned = strip_unknown_options(argv, known)
    ignored = [t for t in argv if t.startswith("--") and t.split("=", 1)[0] not in known]
    if ignored:
        print("Ignoring main.py-only options: " + " ".join(ignored))
    args, leftover = parser.parse_known_args(cleaned)  # never fail on an unknown flag
    if leftover:
        print("Ignoring: " + " ".join(leftover))

    # Printer counts: --printers / positional, else one per --printer_profiles entry (like main.py)
    if args.printers_opt is not None:
        counts = [args.printers_opt]
    elif args.n_printers is not None:
        counts = [args.n_printers]
    elif args.printer_profiles:
        counts = [printers_from_profile(p) for p in args.printer_profiles]
    else:
        counts = [None]

    jobs = args.jobs or [args.job]
    opts = dict(results=args.results, line_width=args.line_width,
                printer_ids=args.printer_id, show_travel=args.show_travel,
                show_printheads=args.show_printheads)

    # Same loop order as main.py: every profile x every job; windows open one after another
    for n in counts:
        for job in jobs:
            visualize(job=job, n_printers=n, **opts)


if __name__ == "__main__":
    main()
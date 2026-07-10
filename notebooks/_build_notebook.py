"""
Build dataset_analysis.ipynb programmatically.

Run once to (re)generate the notebook. The resulting .ipynb is a self-contained
analysis tool that works on the project's `data/<scenario>/{images,labels}` layout
*and* the original Vibe Sim Output directory layout (auto-detected).
"""
import json
import os
import uuid

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src.splitlines(keepends=True), "outputs": [], "execution_count": None})

# ---------------------------------------------------------------------------
# Title and intro
# ---------------------------------------------------------------------------
md("""# Dataset Analyzer — CUAS Drone Detection Challenge

This notebook is the **post-generation QA tool** for the Vibe Sim outputs. After
you run any prompt sequence in `vibesim_prompts/`, point this notebook at the
resulting folder and it tells you:

1. How many images / labels were actually produced and whether they pair up.
2. The **pixels-on-target distribution** for the drone class (and the clutter
   classes if they were enabled) — this is the *single most important* statistic
   for sim-to-real transfer (see `strategy/DATA_STRATEGY.md` §1).
3. How your generated distribution compares to the **real IEEE reference
   distribution** (`ieee_pixels_on_target.npy`) — exactly the question the
   custom-distribution workflow in Section 5.2 of the problem statement is
   designed to answer.
4. Drone placement (where in the image frame the drone appears) — catches
   spatial bias where the model would otherwise fail to detect edge cases.
5. Background brightness / contrast statistics — catches "too clean" sim outputs
   (a classic sim-to-real failure mode called out in §3 of the strategy doc).
6. Class balance (drone vs bird vs aircraft) — validates the clutter strategy.

It supports two layouts automatically:

* **This project's layout** (recommended): `data/<scenario>/{images,labels}/...`
* **Raw Vibe Sim Output layout** (for when you export directly from the Vibe Sim
  file browser): the `CaptureSensor_BP_C_0_Color` / `labels` / `metadata` tree.

Run all cells top-to-bottom. Edit the `DATASETS` config block in cell 3 to point
at the folders you want to analyze.
""")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
code("""import json
import os
import warnings
from glob import glob
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# Plotting: in a headless server use Agg, otherwise inline in the notebook.
try:
    import matplotlib
    if not os.environ.get("MPLBACKEND"):
        try:
            %matplotlib inline
        except Exception:
            matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib required: pip install matplotlib")

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow required: pip install Pillow")

print("Imports OK. Matplotlib backend:", matplotlib.get_backend())""")

# ---------------------------------------------------------------------------
# Configuration cell
# ---------------------------------------------------------------------------
md("""## 1. Configuration — point at the folders you want to analyze

`DATASETS` is a list of dicts. Each dict can be either:

* `{ "name": "...", "layout": "project", "root": "data/rural" }`  — analyzes everything
  under `data/rural/{images,labels}/`, with optional nested batch subfolders.
* `{ "name": "...", "layout": "vibesim", "root": "/path/to/vibesim/Output/.../<run_name>" }`
  — analyzes a raw Vibe Sim export (auto-finds `*_Color/`, `labels/`, `metadata/`).

If you have `ieee_pixels_on_target.npy` from the Vibe Sim starter files, set
`IEEE_NPY` to its path to get a side-by-side comparison plot. Otherwise leave it
as `None` and the comparison plot is skipped.
""")

code("""# === EDIT THIS BLOCK ===
DATASETS = [
    {"name": "rural_main",   "layout": "project", "root": "data/rural"},
    {"name": "urban_main",   "layout": "project", "root": "data/urban"},
    # {"name": "ir_main",    "layout": "project", "root": "data/ir"},
    # {"name": "vibesim_raw","layout": "vibesim", "root": "/home/ubuntu/Output/Hackathon_RuralCUAS/.../2-19-2026-22-36-32"},
]

IEEE_NPY = "data/real_reference/ieee_pixels_on_target.npy"  # set None to skip
# ======================

# Style niceties
plt.rcParams.update({
    "figure.figsize": (7, 4),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

print(f"Configured {len(DATASETS)} dataset(s).")
for d in DATASETS:
    print(f"  - {d['name']:<16} layout={d['layout']:<8} root={d['root']}")
print(f"IEEE reference: {IEEE_NPY or '(skipped)'}")""")

# ---------------------------------------------------------------------------
# The DataAnalyzer class
# ---------------------------------------------------------------------------
md("""## 2. `DataAnalyzer` — does the actual work

This is the only class you need. The class auto-detects which layout you're
using, finds all image/label pairs, and exposes everything as numpy arrays
ready for plotting.
""")

code("""class DataAnalyzer:
    # Analyze a Vibe Sim export for the CUAS challenge.
    #
    # Supports two layouts:
    #   * 'project': <root>/images/*.png + <root>/labels/*.txt (optionally nested
    #     in batch subfolders).
    #   * 'vibesim': a raw Vibe Sim Output run dir, which has
    #     <root>/CaptureSensor_BP_C_0_Color/*.png,
    #     <root>/labels/*.txt, and optionally <root>/metadata/*.npy.
    #
    # Class ID convention (matches the rest of the project):
    #   0 = drone, 1 = bird, 2 = manned aircraft.

    CLASS_NAMES = {0: "drone", 1: "bird", 2: "manned_aircraft"}
    CLASS_COLORS = {0: "#1f77b4", 1: "#d62728", 2: "#9467bd"}

    def __init__(self, name, layout, root):
        self.name = name
        self.layout = layout
        self.root = root
        self.image_paths = []
        self.label_paths = []
        self.metadata = {}  # class_id -> np.ndarray (N,5) or similar, if present

        if layout == "project":
            self._load_project_layout()
        elif layout == "vibesim":
            self._load_vibesim_layout()
        else:
            raise ValueError(f"Unknown layout: {layout!r}")

        self._parse_labels()

    # ------------------------------------------------------------------ I/O
    def _load_project_layout(self):
        if not os.path.isdir(self.root):
            print(f"  [{self.name}] WARNING: {self.root} does not exist — skipping.")
            return
        # images/ dir at any depth
        for images_dir in glob(os.path.join(self.root, "**", "images"), recursive=True):
            labels_dir = os.path.join(os.path.dirname(images_dir), "labels")
            for img in glob(os.path.join(images_dir, "*")):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    stem = os.path.splitext(os.path.basename(img))[0]
                    lbl = os.path.join(labels_dir, stem + ".txt")
                    if os.path.exists(lbl):
                        self.image_paths.append(img)
                        self.label_paths.append(lbl)
                    else:
                        # orphan image
                        self.image_paths.append(img)

    def _load_vibesim_layout(self):
        if not os.path.isdir(self.root):
            print(f"  [{self.name}] WARNING: {self.root} does not exist — skipping.")
            return
        # Find the *_Color dir (the official starter notebook hardcodes one name;
        # we accept any suffix to be robust to Falcon version drift).
        color_dirs = glob(os.path.join(self.root, "*_Color"))
        if not color_dirs:
            print(f"  [{self.name}] WARNING: no *_Color dir under {self.root}")
            return
        color_dir = color_dirs[0]
        labels_dir = os.path.join(self.root, "labels")
        metadata_dir = os.path.join(self.root, "metadata")
        for img in sorted(glob(os.path.join(color_dir, "*.png"))):
            stem = os.path.splitext(os.path.basename(img))[0]
            lbl = os.path.join(labels_dir, stem + ".txt")
            if os.path.exists(lbl):
                self.image_paths.append(img)
                self.label_paths.append(lbl)
        # load metadata .npy files if present
        meta_files = {
            0: "drone_area_pixels_width_height_distance_dataset.npy",
            1: "bird_area_pixels_width_height_distance_dataset.npy",
            2: "manned_aircraft_area_pixels_width_height_distance_dataset.npy",
        }
        for cls, fname in meta_files.items():
            fpath = os.path.join(metadata_dir, fname)
            if os.path.isfile(fpath):
                self.metadata[cls] = np.load(fpath)

    def _parse_labels(self):
        # Read every YOLO .txt label and accumulate per-class statistics.
        self.boxes_per_image = []          # list of [class, xc, yc, w, h] rows
        self.pot_by_class = {0: [], 1: [], 2: []}    # pixels-on-target per class
        self.placement_by_class = {0: [], 1: [], 2: []}  # (xc, yc) per class
        self.image_count = 0
        self.label_count = 0
        self.unpaired = 0

        for img_path, lbl_path in zip(self.image_paths, self.label_paths):
            self.image_count += 1
            try:
                with Image.open(img_path) as im:
                    iw, ih = im.size
            except Exception:
                self.unpaired += 1
                continue
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    self.boxes_per_image.append([cls, xc, yc, bw, bh])
                    self.pot_by_class.setdefault(cls, []).append(bw * iw * bh * ih)
                    self.placement_by_class.setdefault(cls, []).append((xc, yc))
                    self.label_count += 1

        # convert to numpy for convenience
        for cls in self.pot_by_class:
            self.pot_by_class[cls] = np.array(self.pot_by_class[cls], dtype=np.float64)
            self.placement_by_class[cls] = (
                np.array(self.placement_by_class[cls], dtype=np.float64)
                if self.placement_by_class[cls] else np.zeros((0, 2))
            )

    # ------------------------------------------------------------- Reporting
    def summary_table(self):
        print(f"\\n[{self.name}]  layout={self.layout}  root={self.root}")
        print(f"  paired image/label files : {self.image_count}")
        print(f"  total label rows         : {self.label_count}")
        for cls, name in self.CLASS_NAMES.items():
            pot = self.pot_by_class.get(cls, np.array([]))
            print(f"  class {cls} ({name:<16}): n={pot.size:>5}"
                  + (f"  pot median={np.median(pot):.0f}  p5={np.percentile(pot,5):.0f}  p95={np.percentile(pot,95):.0f}"
                     if pot.size else "  (none)"))

    def plot_pot_histograms(self, bins=40, ax=None):
        for cls, name in self.CLASS_NAMES.items():
            pot = self.pot_by_class.get(cls, np.array([]))
            if pot.size == 0:
                continue
            ax.hist(pot, bins=bins, alpha=0.55,
                    color=self.CLASS_COLORS[cls], label=f"{name} (n={pot.size})",
                    edgecolor="black", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_xlabel("pixels on target (px²)")
        ax.set_ylabel("count (log scale)")
        ax.set_title(f"Pixels-on-target distribution — {self.name}")
        ax.legend()

    def plot_placement_heatmap(self, ax=None):
        # 2D histogram of where the drone appears in the image frame.
        pts = self.placement_by_class.get(0, np.zeros((0, 2)))
        if pts.size == 0:
            ax.text(0.5, 0.5, "no drone boxes", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"Drone placement — {self.name} (empty)")
            return
        # bins in normalized image coords [0,1]
        h, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=20, range=[[0, 1], [0, 1]])
        im = ax.imshow(h, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
                       cmap="magma")
        ax.set_xlabel("image x (normalized)")
        ax.set_ylabel("image y (normalized)")
        ax.set_title(f"Drone placement heatmap — {self.name}  (n={pts.size})")
        plt.colorbar(im, ax=ax, label="count")

    def plot_background_stats(self, max_images=200):
        # Mean / std brightness — catches 'too clean' synthetic data.
        paths = self.image_paths[:max_images]
        if not paths:
            print(f"  [{self.name}] no images for background stats")
            return
        means, stds = [], []
        for p in paths:
            try:
                with Image.open(p) as im:
                    arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
                means.append(arr.mean())
                stds.append(arr.std())
            except Exception:
                continue
        if not means:
            return
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].hist(means, bins=25, color="#2ca02c", edgecolor="black", linewidth=0.4)
        ax[0].set_title(f"Background mean brightness — {self.name}")
        ax[0].set_xlabel("mean (0=black, 1=white)")
        ax[1].hist(stds, bins=25, color="#ff7f0e", edgecolor="black", linewidth=0.4)
        ax[1].set_title(f"Background contrast (std) — {self.name}")
        ax[1].set_xlabel("pixel std")
        fig.tight_layout()
        plt.show()
        print(f"  [{self.name}] mean brightness median={np.median(means):.2f}, "
              f"std median={np.median(stds):.2f}")
        print("    (real-world imagery typically has std > 0.15; if yours is much lower, the sim output is 'too clean'. See DATA_STRATEGY.md §3.)")

print("DataAnalyzer class loaded.")""")

# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------
md("""## 3. Load datasets

Runs the `DataAnalyzer` over everything in `DATASETS`. Any folder that doesn't
exist is silently skipped (so you can run this with empty `data/rural/` while
you wait for Vibe Sim exports).
""")

code("""analyzers = []
for d in DATASETS:
    a = DataAnalyzer(d["name"], d["layout"], d["root"])
    analyzers.append(a)
    a.summary_table()

# Load IEEE reference if available
ieee_areas = None
if IEEE_NPY and os.path.isfile(IEEE_NPY):
    ieee_areas = np.load(IEEE_NPY)
    print(f"\\nIEEE reference distribution loaded: n={ieee_areas.size}, "
          f"mean={ieee_areas.mean():.0f}, median={np.median(ieee_areas):.0f}")
else:
    print("\\nIEEE reference distribution not provided — skipping sim-to-real comparison plot.")""")

# ---------------------------------------------------------------------------
# Pixels-on-target histograms
# ---------------------------------------------------------------------------
md("""## 4. Pixels-on-target histograms

**This is the single most important diagnostic** for the sim-to-real transfer
question (problem statement §5.2). If your generated distribution doesn't look
*roughly* like the IEEE reference distribution, your model will be brittle on
real-world small drones — even if it has perfect in-sim mAP.
""")

code("""# One panel per dataset, plus an overlay panel that includes IEEE if available
n = len([a for a in analyzers if a.image_count > 0])
if n == 0:
    print("No datasets to plot — populate data/<scenario>/images and labels first.")
else:
    fig, axes = plt.subplots(1, n + (1 if ieee_areas is not None else 0),
                             figsize=(5 * (n + (1 if ieee_areas is not None else 0)), 4))
    if axes.ndim == 0:
        axes = [axes]
    idx = 0
    for a in analyzers:
        if a.image_count == 0:
            continue
        a.plot_pot_histograms(ax=axes[idx])
        idx += 1
    if ieee_areas is not None:
        ax = axes[idx]
        ax.hist(ieee_areas, bins=40, color="black", alpha=0.55,
                edgecolor="black", linewidth=0.4, label=f"IEEE reference (n={ieee_areas.size})")
        ax.set_yscale("log")
        ax.set_xlabel("pixels on target (px²)")
        ax.set_ylabel("count (log scale)")
        ax.set_title("IEEE EO reference (real-world target)")
        ax.legend()
    fig.tight_layout()
    plt.show()""")

# ---------------------------------------------------------------------------
# Quantitative comparison with IEEE
# ---------------------------------------------------------------------------
md("""## 5. Quantitative comparison against the IEEE reference

Per-dataset Wasserstein distance and percentile match. Wasserstein is a proper
distance between two 1-D distributions; smaller = closer to the real
distribution. We also print what the matching Vibe Sim custom-distribution
prompt should look like.
""")

code("""def percentile_table(areas):
    return {p: float(np.percentile(areas, p)) for p in (5, 25, 50, 75, 95)}

if ieee_areas is None:
    print("IEEE reference not loaded — set IEEE_NPY above to enable comparison.")
else:
    print(f"{'dataset':<16} {'n':>6} {'median':>10} {'p5':>10} {'p95':>10} {'W_dist':>10}")
    ref = np.sort(ieee_areas)
    for a in analyzers:
        pot = a.pot_by_class.get(0, np.array([]))
        if pot.size == 0:
            continue
        # simple 1-D Wasserstein (no scipy dep)
        a_sorted = np.sort(pot)
        n1, n2 = len(a_sorted), len(ref)
        # quantile-based approximation
        qs = np.linspace(0, 1, max(n1, n2, 50))
        q1 = np.quantile(a_sorted, qs)
        q2 = np.quantile(ref, qs)
        w = float(np.mean(np.abs(q1 - q2)))
        med = float(np.median(pot))
        p5 = float(np.percentile(pot, 5))
        p95 = float(np.percentile(pot, 95))
        print(f"{a.name:<16} {pot.size:>6} {med:>10.0f} {p5:>10.0f} {p95:>10.0f} {w:>10.0f}")
    print("\\nReference (IEEE): median={:.0f}  p5={:.0f}  p95={:.0f}".format(
        np.median(ieee_areas), np.percentile(ieee_areas, 5), np.percentile(ieee_areas, 95)))
    print("\\nFor the lowest-Wasserstein dataset, the matching Vibe Sim prompt is:")
    best = min(
        (a for a in analyzers if a.pot_by_class.get(0, np.array([])).size),
        key=lambda a: float(np.mean(np.abs(
            np.quantile(np.sort(a.pot_by_class[0]), np.linspace(0, 1, 50)) -
            np.quantile(ieee_areas, np.linspace(0, 1, 50))
        ))),
        default=None,
    )
    if best is not None:
        pot = best.pot_by_class[0]
        print(f'  "Make sure the drones are uniformly distributed between {np.percentile(pot, 5):.0f} '
              f'and {np.percentile(pot, 95):.0f} square pixels"')
        print("  (Or, preferred, load the .npy directly as a Custom distribution — preserves the real shape.)")""")

# ---------------------------------------------------------------------------
# Placement heatmaps
# ---------------------------------------------------------------------------
md("""## 6. Drone placement heatmaps

Catches spatial bias. If the drone is consistently in the center 30% of the
frame, the model will under-perform near the edges — which is where real CUAS
systems often see threats (edge of FOV, near the horizon).
""")

code("""nonempty = [a for a in analyzers if a.image_count > 0 and a.placement_by_class.get(0, np.zeros((0,2))).size > 0]
if not nonempty:
    print("No drone boxes found in any loaded dataset.")
else:
    fig, axes = plt.subplots(1, len(nonempty), figsize=(5 * len(nonempty), 4.5))
    if len(nonempty) == 1:
        axes = [axes]
    for ax, a in zip(axes, nonempty):
        a.plot_placement_heatmap(ax=ax)
    fig.tight_layout()
    plt.show()""")

# ---------------------------------------------------------------------------
# Background stats
# ---------------------------------------------------------------------------
md("""## 7. Background brightness / contrast

Catches the "too clean" failure mode (synthetic images that lack the noise/
texture of real-world imagery). Real outdoor imagery typically has pixel std
> 0.15; if your sim outputs cluster near 0.05–0.10, enable film grain in
Vibe Sim (see `vibesim_prompts/phase1_rural.md` Batch D).
""")

code("""for a in analyzers:
    if a.image_count == 0:
        continue
    print(f"\\n--- {a.name} ---")
    a.plot_background_stats()""")

# ---------------------------------------------------------------------------
# Class balance
# ---------------------------------------------------------------------------
md("""## 8. Class balance (drone vs clutter)

Validates the clutter strategy. If you enabled birds / manned aircraft in
Vibe Sim (recommended — see DATA_STRATEGY.md §5), you should see *some* class 1
or class 2 labels. If you didn't enable clutter at all, your model will likely
false-positive on birds in the real eval set.
""")

code("""if not analyzers:
    print("No analyzers loaded.")
else:
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    names, drone_counts, bird_counts, aircraft_counts = [], [], [], []
    for a in analyzers:
        if a.image_count == 0:
            continue
        names.append(a.name)
        drone_counts.append(int(a.pot_by_class.get(0, np.array([])).size))
        bird_counts.append(int(a.pot_by_class.get(1, np.array([])).size))
        aircraft_counts.append(int(a.pot_by_class.get(2, np.array([])).size))
    if not names:
        print("No datasets to plot.")
    else:
        x = np.arange(len(names))
        w = 0.27
        ax.bar(x - w, drone_counts,     width=w, color=DataAnalyzer.CLASS_COLORS[0], label="drone")
        ax.bar(x,     bird_counts,      width=w, color=DataAnalyzer.CLASS_COLORS[1], label="bird")
        ax.bar(x + w, aircraft_counts,  width=w, color=DataAnalyzer.CLASS_COLORS[2], label="manned aircraft")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("box count (log)")
        ax.set_title("Class balance per dataset")
        ax.legend()
        fig.tight_layout()
        plt.show()
        for n, d, b, ac in zip(names, drone_counts, bird_counts, aircraft_counts):
            ratio = (b + ac) / max(d, 1)
            print(f"  {n:<16} drone={d:<5} bird={b:<5} aircraft={ac:<5}  clutter/drone ratio = {ratio:.2f}")
            print(f"  {'':<16} (target: 0.2–0.5 — enough clutter to learn negatives, not so much the model confuses them)")""")

# ---------------------------------------------------------------------------
# Sample images with overlays
# ---------------------------------------------------------------------------
md("""## 9. Spot-check — sample images with label overlays

Always look at your data. Plots won't catch a corrupted label, a misaligned
camera, or a drone that's been cut off at the image edge.
""")

code("""import random

def show_samples(analyzer, n=6, seed=0):
    if analyzer.image_count == 0:
        print(f"[{analyzer.name}] no images to show.")
        return
    rng = random.Random(seed)
    idxs = rng.sample(range(analyzer.image_count), min(n, analyzer.image_count))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for ax, i in zip(axes, idxs):
        img_path = analyzer.image_paths[i]
        lbl_path = analyzer.label_paths[i] if i < len(analyzer.label_paths) else None
        try:
            with Image.open(img_path) as im:
                iw, ih = im.size
                arr = np.asarray(im.convert("RGB"))
        except Exception as e:
            ax.text(0.5, 0.5, f"failed to load: {e}", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.imshow(arr)
        if lbl_path and os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw / 2) * iw; y1 = (yc - bh / 2) * ih
                    x2 = (xc + bw / 2) * iw; y2 = (yc + bh / 2) * ih
                    color = DataAnalyzer.CLASS_COLORS.get(cls, "white")
                    name = DataAnalyzer.CLASS_NAMES.get(cls, f"cls{cls}")
                    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                               fill=False, edgecolor=color, linewidth=1.5))
                    ax.text(x1, max(y1 - 3, 0), name, color=color, fontsize=8,
                            bbox=dict(facecolor="white", alpha=0.6, pad=1, edgecolor="none"))
        ax.set_title(os.path.basename(img_path), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(idxs), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Sample images — {analyzer.name}")
    fig.tight_layout()
    plt.show()

# Edit this to focus on a specific dataset
FOCUS = analyzers[0].name if analyzers else None
for a in analyzers:
    if a.name == FOCUS:
        show_samples(a, n=6, seed=0)
        break""")

# ---------------------------------------------------------------------------
# Decision helper
# ---------------------------------------------------------------------------
md("""## 10. Decision helper — is this batch ready to train on?

Go / no-go check based on the diagnostics above. This is a rough heuristic —
override it with your own judgement.
""")

code("""def go_nogo(analyzer, ieee_areas):
    if analyzer.image_count == 0:
        return "NO-GO", "no images"
    pot = analyzer.pot_by_class.get(0, np.array([]))
    if pot.size < 50:
        return "NO-GO", f"only {pot.size} drone boxes — generate at least ~200 images before training"
    # check coverage: drone should appear across the full image frame
    pts = analyzer.placement_by_class.get(0, np.zeros((0, 2)))
    if pts.size > 0:
        std_x, std_y = float(np.std(pts[:, 0])), float(np.std(pts[:, 1]))
        if std_x < 0.05 or std_y < 0.05:
            return "NO-GO", f"spatial bias detected (std_x={std_x:.3f}, std_y={std_y:.3f}) — vary placement range"
    if ieee_areas is not None and pot.size > 0:
        # rough shape match: median within 2x and p95 within 5x
        med_sim, med_ref = float(np.median(pot)), float(np.median(ieee_areas))
        p95_sim, p95_ref = float(np.percentile(pot, 95)), float(np.percentile(ieee_areas, 95))
        if not (0.5 * med_ref <= med_sim <= 2.0 * med_ref):
            return "WARN", f"median POT ({med_sim:.0f}) far from IEEE ({med_ref:.0f}) — consider re-running with custom distribution"
        if p95_sim > 5 * p95_ref:
            return "WARN", f"p95 POT ({p95_sim:.0f}) much larger than IEEE ({p95_ref:.0f}) — too many close-up drones"
    return "GO", "looks reasonable; train and check val mAP"

print(f"{'dataset':<16} {'verdict':<8} reason")
for a in analyzers:
    verdict, reason = go_nogo(a, ieee_areas)
    print(f"{a.name:<16} {verdict:<8} {reason}")""")

# ---------------------------------------------------------------------------
# Wrap-up
# ---------------------------------------------------------------------------
md("""## What's next?

* **If everything is GO / WARN**: run `python scripts/prepare_dataset.py --scenarios rural urban --out data/merged` then `python scripts/train_yolo.py`.
* **If NO-GO**: re-run the matching Vibe Sim prompt sequence with the indicated axis varied (placement, distribution, or sample size).
* **Between batches**: come back to this notebook — re-run with a new `DATASETS` entry pointing at the latest batch, and diff the histograms to see whether the change actually moved the distribution in the direction you wanted.

The histogram that moves the most is the one that matters. The one that doesn't move — drop that axis from the next batch, you have no signal there.
""")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(os.path.dirname(__file__), "dataset_analysis.ipynb")
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Wrote {out_path}  ({len(cells)} cells)")

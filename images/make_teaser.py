"""Generate the README teaser figure.

Renders the benchmark as a detailed pipeline:
    Configure (inputs)  ->  Inside each run (internal mechanics)  ->  Measure (outputs)

The middle column expands what actually happens during a run, including the
per-epoch training loop with its method-specific loss and oversmoothing probe.

Run:  python3 images/make_teaser.py
Output:  images/teaser.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ----------------------------------------------------------------------------- palette
INK = "#1f2933"        # near-black text
MUTED = "#52606d"      # secondary text
EDGE = "#cbd2d9"       # light borders
BG = "#ffffff"

BLUE = {"band": "#2563eb", "fill": "#eef4ff"}    # inputs
TEAL = {"band": "#0f766e", "fill": "#e6f4f2"}    # engine
ORANGE = {"band": "#c2410c", "fill": "#fdf0e8"}  # outputs
LOOP_FILL = "#fff7e6"                            # training-loop highlight
LOOP_EDGE = "#b45309"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "savefig.facecolor": BG,
    "figure.facecolor": BG,
})


def rbox(ax, x, y, w, h, fill, edge=EDGE, lw=1.2, r=0.06, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=z,
        mutation_aspect=1.0,
    ))


def band(ax, x, y, w, h, color, title):
    rbox(ax, x, y, w, h, color["band"], edge=color["band"], r=0.08, z=4)
    ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
            color="white", fontsize=15, fontweight="bold", zorder=5)


def step(ax, x, y, w, h, color, lines, lead=None, fs=12.0):
    """A content box. `lead` is an optional bold accent word at the front."""
    rbox(ax, x, y, w, h, color["fill"], r=0.06, z=2)
    if lead:
        ax.text(x + 0.34, y + h / 2, lead, ha="left", va="center",
                color=color["band"], fontsize=18, fontweight="bold", zorder=4)
        ax.text(x + 1.5, y + h / 2, lines, ha="left", va="center",
                color=INK, fontsize=fs, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, lines, ha="center", va="center",
                color=INK, fontsize=fs, zorder=4, linespacing=1.35)


def down(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                 mutation_scale=20, linewidth=2.2, color=MUTED, zorder=5,
                 shrinkA=0, shrinkB=0))


def right(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=22, linewidth=2.3, color=MUTED, zorder=6,
                 shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(15.5, 8.6))
ax.set_xlim(0, 23)
ax.set_ylim(-0.5, 13)
ax.axis("off")

TOP = 12.3

# ============================================================ LEFT — Configure (inputs)
lx, lw = 0.4, 4.7
band(ax, lx, TOP - 0.95, lw, 0.95, BLUE, "Configure")
axes_in = [("24", "graph datasets"), ("10", "label-noise types"),
           ("5", "GNN backbones"), ("13", "robustness methods")]
chip_h, gap = 1.5, 0.42
cy = TOP - 0.95 - 0.5
for big, small in axes_in:
    cy -= chip_h
    step(ax, lx, cy, lw, chip_h, BLUE, small, lead=big, fs=12.5)
    cy -= gap
ax.text(lx + lw / 2, cy + 0.05, "config.yaml  ·  £[...] sweep → Cartesian grid of runs",
        ha="center", va="center", color=MUTED, fontsize=10.5, style="italic")

right(ax, lx + lw + 0.1, lx + lw + 1.0, 6.4)

# ===================================================== MIDDLE — Inside each run (engine)
mx, mw = lx + lw + 1.15, 11.3
band(ax, mx, TOP - 0.95, mw, 0.95, TEAL, "Inside each run")

# vertical flow of steps
sy = TOP - 0.95 - 0.45
SH = 1.18   # standard step height


def mstep(text, h=SH, fill=TEAL):
    global sy
    sy -= h
    step(ax, mx, sy, mw, h, fill, text, fs=12.0)
    box_top = sy
    sy -= 0.0
    return box_top


# 1. data + split
top1 = mstep("Load graph  →  train / val / test split")
sy -= 0.42
down(ax, mx + mw / 2, top1, sy + 0.0)

# 2. noise
top2 = mstep("Inject label noise via transition matrix  P\ncorrupt train + val labels  ·  test labels stay clean", h=1.45)
sy -= 0.42
down(ax, mx + mw / 2, top2, sy)

# 3. assemble
top3 = mstep("Assemble model  =  backbone (GCN / GIN / GAT / GATv2 / GPS)  +  robustness method")
sy -= 0.42
down(ax, mx + mw / 2, top3, sy)

# 4. TRAINING LOOP (expanded)
loop_h = 2.9
sy -= loop_h
lyx, lyy = mx, sy
rbox(ax, lyx, lyy, mw, loop_h, LOOP_FILL, edge=LOOP_EDGE, lw=1.8, r=0.05, z=2)
ax.text(lyx + 0.35, lyy + loop_h - 0.32, "Training loop · per epoch",
        ha="left", va="center", color=LOOP_EDGE, fontsize=12, fontweight="bold", zorder=4)

# inner step chips
inner = ["forward\npass", "method-specific\nloss", "backprop\n+ step",
         "val loss\n→ early stop", "checkpoint\nbest"]
n = len(inner)
pad = 0.45
row_y = lyy + 0.4
ich_w = (mw - 2 * pad - (n - 1) * 0.5) / n
ich_h = 1.05
ix = lyx + pad
centers = []
for i, t in enumerate(inner):
    rbox(ax, ix, row_y, ich_w, ich_h, "white", edge=LOOP_EDGE, lw=1.3, r=0.08, z=4)
    ax.text(ix + ich_w / 2, row_y + ich_h / 2, t, ha="center", va="center",
            color=INK, fontsize=10.5, zorder=5, linespacing=1.15)
    centers.append((ix, ix + ich_w))
    ix += ich_w + 0.5
# arrows between inner chips
for i in range(n - 1):
    ax.add_patch(FancyArrowPatch((centers[i][1], row_y + ich_h / 2),
                 (centers[i + 1][0], row_y + ich_h / 2), arrowstyle="-|>",
                 mutation_scale=15, linewidth=1.8, color=LOOP_EDGE, zorder=5,
                 shrinkA=0, shrinkB=0))
# "next epoch" return arrow arcing cleanly above the chip row
ax.add_patch(FancyArrowPatch(
    (centers[-1][0] + ich_w / 2, row_y + ich_h),
    (centers[0][0] + ich_w / 2, row_y + ich_h),
    connectionstyle="arc3,rad=0.34", arrowstyle="-|>", mutation_scale=15,
    linewidth=1.8, color=LOOP_EDGE, linestyle=(0, (4, 2)), zorder=5,
    shrinkA=6, shrinkB=6))
ax.text(lyx + mw / 2, lyy + loop_h - 0.72, "next epoch  ·  probe oversmoothing every N epochs",
        ha="center", va="center", color=LOOP_EDGE, fontsize=10, style="italic", zorder=6)

sy -= 0.42
down(ax, mx + mw / 2, lyy, sy)

# 5. restore + evaluate
top5 = mstep("Restore best checkpoint  →  evaluate on clean test split")

# arrow out to outputs
right(ax, mx + mw + 0.1, mx + mw + 1.0, 6.4)

# aggregation footnote spanning the middle
ax.text(mx + mw / 2, sy - 0.65, "repeat over runs  →  report  mean ± std",
        ha="center", va="center", color=MUTED, fontsize=11.5, fontweight="bold", style="italic")

# ============================================================ RIGHT — Measure (outputs)
rx, rw = mx + mw + 1.15, 4.7
band(ax, rx, TOP - 0.95, rw, 0.95, ORANGE, "Measure")
outs = [
    "Classification\nAcc · F1 · Precision · Recall\n+ noise-split analysis",
    "Oversmoothing\nNumRank · Erank · EDir\nEProj · MAD",
    "Compute & carbon\nFLOPs · wall-clock · CO₂",
]
oh, ogap = 2.0, 0.42
oy = TOP - 0.95 - 0.5
for t in outs:
    oy -= oh
    step(ax, rx, oy, rw, oh, ORANGE, t, fs=11.5)
    oy -= ogap

# ------------------------------------------------------------------------- footer line
ax.text(11.5, -0.25,
        "A unified, reproducible pipeline for comparing GNN robustness strategies under label noise",
        ha="center", va="center", color=MUTED, fontsize=12.5, style="italic")

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.01)
fig.savefig("images/teaser.png", dpi=200)
print("wrote images/teaser.png")

"""
Draw the flowcharts used in the Task 1 and Task 2 chapter.

Run with:   ./tf-env/bin/python make_flowcharts.py

They are drawn rather than screenshotted so that a change to the pipeline can be
followed by a change to the diagram, and so that every one of them comes out in
the same style as the ones in the earlier report: a faint grid, white boxes with
a coloured border, diamonds for the decisions, and the title set in bold at the
top left.

Everything is written into report_figures/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

OUT = "report_figures"

# The draw.io palette the earlier report used, so the two sets of diagrams sit
# beside each other without looking like they came from different documents.
GREEN = ("#82b366", "#d5e8d4")     # start and end
RED = ("#b85450", "#f8cecc")       # a numbered stage
BLUE = ("#6c8ebf", "#dae8fc")      # an ordinary step
YELLOW = ("#d6b656", "#fff2cc")    # a decision
GREY = ("#666666", "#f5f5f5")      # a note
INK = "#1f2a44"


def canvas(width, height, title):
    figure, axes = plt.subplots(figsize=(width, height))
    axes.set_xlim(0, 100)
    # a band above 100 so the title has the diagram to itself rather than
    # sitting across the first box
    axes.set_ylim(0, 108)
    axes.axis("off")

    # the faint square grid the earlier diagrams are drawn on
    for x in range(0, 101, 2):
        axes.plot([x, x], [0, 100], color="#e8e8e8", linewidth=0.4, zorder=0)
    for y in range(0, 101, 2):
        axes.plot([0, 100], [y, y], color="#e8e8e8", linewidth=0.4, zorder=0)

    axes.text(1, 106, title, fontsize=10.5, fontweight="bold", color="#000000",
              va="top", ha="left", zorder=5)
    return figure, axes


def box(axes, x, y, w, h, text, colour=BLUE, stage=False, size=7.6):
    """A process box.  `stage` draws the two inset rules draw.io puts on its
    predefined-process shape, which is what the earlier report used for a
    numbered stage."""
    edge, fill = colour
    axes.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fill,
                             edgecolor=edge, linewidth=1.1, zorder=3))
    if stage:
        for offset in (-w / 2 + 1.6, w / 2 - 1.6):
            axes.plot([x + offset, x + offset], [y - h / 2, y + h / 2],
                      color=edge, linewidth=1.0, zorder=4)
    axes.text(x, y, text, ha="center", va="center", fontsize=size, color=INK,
              zorder=5, linespacing=1.45)


def rounded(axes, x, y, w, h, text, colour=GREEN, size=7.6):
    edge, fill = colour
    axes.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fill,
                             edgecolor=edge, linewidth=1.1, zorder=3,
                             joinstyle="round"))
    axes.text(x, y, text, ha="center", va="center", fontsize=size, color=INK,
              zorder=5, linespacing=1.45)


def diamond(axes, x, y, w, h, text, size=7.2):
    edge, fill = YELLOW
    points = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    axes.add_patch(Polygon(points, closed=True, facecolor=fill, edgecolor=edge,
                           linewidth=1.1, zorder=3))
    axes.text(x, y, text, ha="center", va="center", fontsize=size, color=INK,
              zorder=5, linespacing=1.4)


def arrow(axes, start, end, label=None, label_offset=(0, 1.6), elbow=None):
    """A black connector.  `elbow` routes it through one corner point."""
    points = [start] + ([elbow] if elbow else []) + [end]
    for a, b in zip(points, points[1:]):
        axes.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                       color="#000000", linewidth=0.9,
                                       shrinkA=0, shrinkB=0, zorder=4))
    if label:
        mid = ((points[0][0] + points[-1][0]) / 2 + label_offset[0],
               (points[0][1] + points[-1][1]) / 2 + label_offset[1])
        axes.text(mid[0], mid[1], label, fontsize=6.8, color=INK, ha="center",
                  va="center", zorder=5,
                  bbox=dict(facecolor="white", edgecolor="none", pad=0.6))


def save(figure, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print("wrote", path)


# --------------------------------------------------------------- Task 1 chain
def task1_chain():
    figure, axes = canvas(13.5, 3.0, "Task 1 - the pipeline from the user's box to the mask")
    y = 52
    steps = [
        ("Input scan\n+ one box the\nuser drags", GREEN),
        ("Preprocess\nCLAHE + z-score,\ncrop box + 35% pad,\nresample to 128 x 128", BLUE),
        ("Describe\n41 features per pixel\n(32 appearance,\n4 geometry, 5 context)", BLUE),
        ("Classify\nRandom Forest,\n120 trees ->\nprobability map", BLUE),
        ("Post-process\nsmooth 1.5, threshold\n0.5, close, fill holes,\nlargest component", BLUE),
        ("Output\nbinary mask\npasted back into\nthe 512 x 512 frame", GREEN),
    ]
    width, gap = 14.5, 2.2
    x = 9
    centres = []
    for text, colour in steps:
        box(axes, x, y, width, 30, text, colour=colour, size=7.0)
        centres.append(x)
        x += width + gap
    for a, b in zip(centres, centres[1:]):
        arrow(axes, (a + width / 2, y), (b - width / 2, y))

    axes.text(1, 12, "The geometry and context features are the only route the user's box takes into the model,\n"
                     "which is what makes the method semi-automated rather than automatic.",
              fontsize=7.4, color="#444444", va="top", ha="left")
    save(figure, "fig_task1_chain.png")


# ----------------------------------------------------------- Task 1 flowchart
def task1_flowchart():
    figure, axes = canvas(8.6, 11.0,
                          "task1/segmenter.py - segment(image, box): four stages")
    rounded(axes, 34, 95, 24, 6, "segment(image, box)")
    box(axes, 34, 85, 40, 8.5,
        "STAGE 1  preprocess\nCLAHE + z-score over the head, crop the box\nwith 35% padding, resample to 128 x 128",
        colour=RED, stage=True, size=7.0)
    box(axes, 34, 73.5, 40, 8.5,
        "STAGE 2  features\n41 per pixel: appearance, geometry relative\nto the box, context against the box core",
        colour=RED, stage=True, size=7.0)
    diamond(axes, 34, 62, 26, 9, "forest loaded\nin the worker?")
    box(axes, 80, 62, 26, 7.5, "start mvi-env worker,\nunpickle the forest\n(once per session)", colour=GREY, size=6.8)
    box(axes, 34, 50, 40, 8.5,
        "STAGE 3  classify\nRandom Forest -> per-pixel tumour probability,\nreturned to the server over the pipe",
        colour=RED, stage=True, size=7.0)
    diamond(axes, 34, 37.5, 26, 9, "any pixel\nabove 0.5?")
    box(axes, 80, 37.5, 26, 7.5, "return an empty mask\nand say so, rather than\nan invented contour", colour=GREY, size=6.8)
    box(axes, 34, 25, 40, 8.5,
        "STAGE 4  post-process\nsmooth, threshold, close, fill holes,\nkeep the component the box contains",
        colour=RED, stage=True, size=7.0)
    box(axes, 34, 14, 34, 6.5, "paste back into the full 512 x 512 frame", colour=BLUE, size=7.0)
    rounded(axes, 34, 5, 26, 6, "return mask, box, latency")

    arrow(axes, (34, 92), (34, 89.3))
    arrow(axes, (34, 80.7), (34, 77.8))
    arrow(axes, (34, 69.2), (34, 66.6))
    arrow(axes, (47, 62), (67, 62), label="no", label_offset=(0, 1.8))
    arrow(axes, (80, 58.2), (80, 54.5), elbow=None)
    arrow(axes, (80, 54.5), (54, 54.5))
    arrow(axes, (34, 57.4), (34, 54.3), label="yes", label_offset=(3.2, 0))
    arrow(axes, (34, 45.7), (34, 42.1))
    arrow(axes, (47, 37.5), (67, 37.5), label="no", label_offset=(0, 1.8))
    arrow(axes, (34, 33), (34, 29.3), label="yes", label_offset=(3.2, 0))
    arrow(axes, (34, 20.7), (34, 17.3))
    arrow(axes, (34, 10.7), (34, 8.1))
    save(figure, "fig_task1_flowchart.png")


# ------------------------------------------------------- Task 2 architecture
def task2_architecture():
    figure, axes = canvas(12.5, 7.6,
                          "task2/models.py - build_mha_resunet(): encoder, bottleneck, decoder")
    encoder = [("enc1\n32 -> 8", 84), ("enc2\n64 -> 32", 70), ("enc3\n128 -> 60", 56), ("enc4\n256 -> 116", 42)]
    for text, y in encoder:
        box(axes, 16, y, 20, 9, "residual block\n" + text, colour=BLUE, size=7.0)
    for a, b in zip([y for _, y in encoder], [y for _, y in encoder][1:]):
        arrow(axes, (16, a - 4.5), (16, b + 4.5), label="max pool 2", label_offset=(-9, 0))

    box(axes, 50, 27, 30, 10,
        "bottleneck  512 -> 232\nresidual block + multi-head self-attention\n(4 heads, key dim 64, 12 x 12 grid)",
        colour=RED, stage=True, size=7.0)
    arrow(axes, (16, 37.5), (16, 27), elbow=None)
    arrow(axes, (16, 27), (35, 27))

    decoder = [("dec4\n256 -> 116", 42), ("dec3\n128 -> 60", 56), ("dec2\n64 -> 32", 70), ("dec1\n32 -> 8", 84)]
    for text, y in decoder:
        box(axes, 84, y, 20, 9, "residual block\n" + text, colour=BLUE, size=7.0)
    arrow(axes, (65, 27), (84, 27))
    arrow(axes, (84, 27), (84, 37.5))
    for a, b in zip([y for _, y in decoder], [y for _, y in decoder][1:]):
        arrow(axes, (84, a + 4.5), (84, b - 4.5), label="upsample 2", label_offset=(10, 0))

    for y in (84, 70, 56, 42):
        box(axes, 50, y, 24, 7.5, "dual attention gate\nspatial x channel", colour=YELLOW, size=6.8)
        arrow(axes, (26, y), (38, y))
        arrow(axes, (62, y), (74, y))

    rounded(axes, 84, 94, 22, 6, "1 x 1 conv, sigmoid\n-> mask")
    arrow(axes, (84, 88.5), (84, 91))
    rounded(axes, 16, 94, 20, 6, "scan 192 x 192")
    arrow(axes, (16, 91), (16, 88.5))

    axes.text(1, 14, "Each width is given as trained -> pruned.  The key dimension is held at 64 rather than scaled with the\n"
                     "channel count, because a key dimension that moves with the width cannot be sliced when the\n"
                     "bottleneck is pruned, and the narrow model would then not be able to inherit the attention weights.",
              fontsize=7.2, color="#444444", va="top", ha="left")
    save(figure, "fig_task2_architecture.png")


# ----------------------------------------------------------- Task 2 pipeline
def task2_pipeline():
    figure, axes = canvas(13.5, 4.4, "Task 2 - training, tuning, pruning and what was submitted")
    y = 60
    steps = [
        ("Train the full\nMHA-ResUNet\n40 epochs, 49.8 min\n8.84 M parameters", BLUE),
        ("PSO tunes the\nbenchmark U-Net\n6 x 4 = 24 evaluations,\n30.4 min", BLUE),
        ("Train the benchmark\nat the chosen settings\n53.0 min\n4.37 M parameters", BLUE),
        ("Prune 55% of the\nchannels over 6 rounds\n28.8 min\n-> 1.96 M parameters", RED),
        ("Retrain those widths\nfrom scratch\n23.6 min\nval Dice 0.8619", RED),
        ("Evaluate all of them\non the same\n200 test scans", GREEN),
    ]
    width, gap = 14.5, 2.2
    x = 9
    centres = []
    for text, colour in steps:
        box(axes, x, y, width, 34, text, colour=colour, size=6.9,
            stage=(colour is RED))
        centres.append(x)
        x += width + gap
    for a, b in zip(centres, centres[1:]):
        arrow(axes, (a + width / 2, y), (b - width / 2, y))

    axes.text(1, 26, "The two red stages are the ones the two million parameter limit forced.  Pruning alone does not survive at this\n"
                     "depth - it ends at 0.5916 on the training metric - so the widths it selected are retrained from a fresh\n"
                     "initialisation instead, which is the route Liu et al. (2019) argue for and the model that is submitted.",
              fontsize=7.2, color="#444444", va="top", ha="left")
    save(figure, "fig_task2_pipeline.png")


# --------------------------------------------------------------- PSO search
def pso_flowchart():
    figure, axes = canvas(8.2, 9.2, "task2/pso.py - the search that tunes the benchmark U-Net")
    rounded(axes, 40, 94, 30, 6, "6 particles, random positions\nin the four ranges")
    box(axes, 40, 82, 44, 9,
        "build the U-Net at this particle's\nbase filters, dropout, activation\nand learning rate",
        colour=BLUE, size=7.0)
    box(axes, 40, 69, 44, 9,
        "fitness: train 6 epochs on 1,000 scans\nat 128 x 128, score validation Dice",
        colour=BLUE, size=7.0)
    box(axes, 40, 56, 44, 8,
        "update personal best and global best",
        colour=BLUE, size=7.0)
    box(axes, 40, 44, 44, 9,
        "velocity  v <- w.v + c1.r1(p - x) + c2.r2(g - x)\nposition  x <- x + v      (w 0.7, c1 = c2 = 1.5)",
        colour=RED, stage=True, size=6.8)
    diamond(axes, 40, 30, 30, 10, "4 iterations\ndone?")
    box(axes, 86, 30, 22, 7, "next iteration", colour=GREY, size=6.8)
    rounded(axes, 40, 15, 40, 8,
            "best: 24 filters, dropout 0.215,\nelu, lr 1.85e-3  (30.4 min)")

    arrow(axes, (40, 91), (40, 86.5))
    arrow(axes, (40, 77.5), (40, 73.5))
    arrow(axes, (40, 64.5), (40, 60))
    arrow(axes, (40, 52), (40, 48.5))
    arrow(axes, (40, 39.5), (40, 35))
    arrow(axes, (55, 30), (75, 30), label="no", label_offset=(0, 1.8))
    arrow(axes, (86, 33.5), (86, 82))
    arrow(axes, (86, 82), (62, 82))
    arrow(axes, (40, 25), (40, 19), label="yes", label_offset=(3.4, 0))

    axes.text(1, 8, "Searching at 128 x 128 on a subset is what keeps the search affordable.  What matters at this stage is the\n"
                    "ranking of the candidates rather than the Dice any of them reaches.",
              fontsize=7.2, color="#444444", va="top", ha="left")
    save(figure, "fig_task2_pso.png")


# ------------------------------------------------------------ prune flowchart
def prune_flowchart():
    figure, axes = canvas(8.8, 10.8,
                          "task2/prune.py - structured pruning, and where it stopped working")
    rounded(axes, 36, 95, 30, 6, "trained model\n8.84 M parameters")
    box(axes, 36, 85, 42, 8,
        "rank every output channel by the L1 norm\nof the filter that produces it (Li et al., 2017)",
        colour=BLUE, size=7.0)
    box(axes, 36, 74, 42, 8,
        "drop the weakest 12.5%, rebuild the network\nnarrower, slice every kernel on both axes",
        colour=RED, stage=True, size=7.0)
    box(axes, 36, 63, 42, 7.5,
        "recalibrate BatchNorm\n100 forward-only batches, no gradients",
        colour=BLUE, size=7.0)
    box(axes, 36, 53, 42, 7, "recover: 3 epochs", colour=BLUE, size=7.0)
    diamond(axes, 36, 41, 28, 10, "6 rounds\ndone?")
    box(axes, 84, 41, 20, 7, "next round", colour=GREY, size=6.8)
    box(axes, 36, 28, 42, 7.5, "final fine-tune, 12 epochs", colour=BLUE, size=7.0)
    diamond(axes, 36, 16, 30, 10, "recovered to\nthe full model?")
    box(axes, 84, 16, 22, 9, "no - 0.5916, and a mean\nmaximum probability of\n0.517: keep the widths,\nretrain from scratch",
        colour=RED, stage=True, size=6.6)
    rounded(axes, 36, 4, 34, 6, "1,963,474 parameters, 2.79 GFLOPs")

    arrow(axes, (36, 92), (36, 89))
    arrow(axes, (36, 81), (36, 78))
    arrow(axes, (36, 70), (36, 66.8))
    arrow(axes, (36, 59.2), (36, 56.5))
    arrow(axes, (36, 49.5), (36, 46))
    arrow(axes, (50, 41), (74, 41), label="no", label_offset=(0, 1.8))
    arrow(axes, (84, 44.5), (84, 74))
    arrow(axes, (84, 74), (57, 74))
    arrow(axes, (36, 36), (36, 31.8), label="yes", label_offset=(3.4, 0))
    arrow(axes, (36, 24.2), (36, 21))
    arrow(axes, (51, 16), (73, 16), label="no", label_offset=(0, 1.8))
    arrow(axes, (36, 11), (36, 7), label="yes", label_offset=(3.4, 0))
    arrow(axes, (84, 11.5), (84, 4))
    arrow(axes, (84, 4), (53, 4))
    save(figure, "fig_task2_prune.png")


if __name__ == "__main__":
    task1_chain()
    task1_flowchart()
    task2_architecture()
    task2_pipeline()
    pso_flowchart()
    prune_flowchart()

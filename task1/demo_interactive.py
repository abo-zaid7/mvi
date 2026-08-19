"""
Interactive demo - this is the method being used the way it is meant to be used.

Run with:   python demo_interactive.py
            python demo_interactive.py <path to an mri .jpg>

Drag a box around the lesion with the mouse and let go.  The segmentation
appears straight away, together with the metrics if a ground truth mask exists
for that image.  Press 'n' for the next random scan, 'r' to redo the box and
'q' to quit.

The point of this script is to show that no ground truth is involved at run
time: the only thing the algorithm receives is the rectangle drawn by the user.
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector

import config
import dataset
import metrics
import segmenter


class Demo:
    def __init__(self, model, paths):
        self.model = model
        self.paths = paths
        self.rng = np.random.default_rng()

        self.figure, self.axes = plt.subplots(1, 2, figsize=(11, 5.5))
        self.selector = RectangleSelector(self.axes[0], self.on_box, useblit=True,
                                          button=[1], interactive=False)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.load_random()

    # ------------------------------------------------------------------ views
    def load_random(self):
        self.path = str(self.rng.choice(self.paths))
        self.image, self.truth = dataset.load_pair(self.path)
        self.draw()

    def draw(self, pred=None, elapsed=None):
        for ax in self.axes:
            ax.clear()
            ax.axis("off")

        self.axes[0].imshow(self.image, cmap="gray")
        self.axes[0].set_title("drag a box around the tumour\n%s"
                               % os.path.basename(self.path), fontsize=9)

        self.axes[1].imshow(self.image, cmap="gray")
        if self.truth is not None:
            self.axes[1].contour(self.truth, levels=[0.5], colors="lime", linewidths=1.5)
        if pred is not None:
            self.axes[1].contour(pred, levels=[0.5], colors="red", linewidths=1.5)
            scores = metrics.all_metrics(pred, self.truth)
            self.axes[1].set_title(
                "green = ground truth, red = prediction\n"
                "Dice %.3f   IoU %.3f   HD95 %.1f px   %.0f ms"
                % (scores["dice"], scores["iou"], scores["hd95"], elapsed * 1000),
                fontsize=9)
        else:
            self.axes[1].set_title("result appears here\n"
                                   "keys:  n = next scan   r = reset   q = quit", fontsize=9)

        self.figure.tight_layout()
        self.figure.canvas.draw_idle()

    # ---------------------------------------------------------------- events
    def on_box(self, click, release):
        left, right = sorted([int(click.xdata), int(release.xdata)])
        top, bottom = sorted([int(click.ydata), int(release.ydata)])
        if bottom - top < 5 or right - left < 5:
            print("box too small, try again")
            return

        start = time.time()
        pred = self.model.segment(self.image, (top, left, bottom, right))
        elapsed = time.time() - start
        self.draw(pred, elapsed)

    def on_key(self, event):
        if event.key == "n":
            self.load_random()
        elif event.key == "r":
            self.draw()
        elif event.key == "q":
            plt.close(self.figure)


def main():
    model = segmenter.load_segmenter()

    if len(sys.argv) > 1:
        paths = [sys.argv[1]]
    else:
        paths = dataset.list_images("test")
        print("loaded %d test scans" % len(paths))

    print("drag a box around the tumour.  n = next, r = reset, q = quit")
    Demo(model, paths)
    plt.show()


if __name__ == "__main__":
    main()

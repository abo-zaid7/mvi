"""
Turning masks into the pictures the GUI shows.

The brief asks for the ground truth image and the segmented boundaries to be
visualised, so there is one function per view and they all return a PNG as
bytes.  Everything is drawn at 512 x 512, which is the resolution both tasks are
scored at.

Two palettes are provided.  The default one is the usual green/red pairing that
radiology tools use, and the second is the Okabe-Ito colour-blind safe set, for
the roughly 1 in 12 men with red-green colour vision deficiency who cannot
reliably separate the first pair.
"""

import cv2
import numpy as np

SIZE = 512

# Colours are written (blue, green, red) because that is the order OpenCV works
# in, which is worth remembering when editing these - they are easy to get
# backwards.
PALETTES = {
    # The green/red pairing radiology tools normally use.
    "clinical": {
        "truth": (80, 220, 80),        # green   - ground truth
        "prediction": (60, 60, 255),   # red     - the model's boundary
        "overlap": (60, 200, 240),     # amber   - where they agree
        "missed": (60, 150, 255),      # orange  - tumour the model missed
        "extra": (200, 90, 220),       # magenta - healthy tissue called tumour
    },
    # Okabe & Ito's palette, which stays separable under all three common types
    # of colour vision deficiency.  Green and red - the default pair above - are
    # the one combination roughly 1 in 12 men cannot reliably tell apart.
    "accessible": {
        "truth": (115, 158, 0),        # bluish green  #009E73
        "prediction": (0, 94, 213),    # vermillion    #D55E00
        "overlap": (66, 228, 240),     # yellow        #F0E442
        "missed": (233, 180, 86),      # sky blue      #56B4E9
        "extra": (167, 121, 204),      # reddish purple #CC79A7
    },
}


def palette(name):
    return PALETTES.get(name, PALETTES["clinical"])


def to_png(canvas):
    ok, buffer = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("could not encode the image as PNG")
    return buffer.tobytes()


def as_colour(image):
    """A greyscale scan as a 3 channel canvas we can draw coloured lines on."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def draw_outline(canvas, mask, colour, thickness=2):
    """Trace the boundary of a mask onto the canvas."""
    if mask is None or not mask.any():
        return canvas
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, colour, thickness)
    return canvas


def scan(image):
    """The MRI on its own."""
    return to_png(as_colour(image))


def truth_view(image, truth, colours, filled=True, opacity=0.35):
    """
    The ground truth view the brief asks for.

    It is shown as a translucent fill plus a solid outline: the fill answers
    "where is the lesion" at a glance and the outline keeps the exact border
    visible, which a fill on its own hides.
    """
    canvas = as_colour(image)
    if truth is None:
        return to_png(canvas)

    if filled:
        wash = canvas.copy()
        wash[truth] = colours["truth"]
        canvas = cv2.addWeighted(wash, opacity, canvas, 1 - opacity, 0)

    return to_png(draw_outline(canvas, truth, colours["truth"]))


def prediction_view(image, prediction, colours, filled=True, opacity=0.35):
    """The model's mask on its own, drawn the same way as the ground truth."""
    canvas = as_colour(image)
    if prediction is None:
        return to_png(canvas)

    if filled:
        wash = canvas.copy()
        wash[prediction] = colours["prediction"]
        canvas = cv2.addWeighted(wash, opacity, canvas, 1 - opacity, 0)

    return to_png(draw_outline(canvas, prediction, colours["prediction"]))


def boundary_view(image, truth, prediction, colours, box=None, thickness=2):
    """
    Both boundaries over the scan - the main view.

    Two outlines on one image is what makes the error visible: where the model
    is right the two lines sit on top of each other, and everywhere they
    separate is exactly the error the Dice score is summarising.
    """
    canvas = as_colour(image)

    if box is not None:
        top, left, bottom, right = box
        cv2.rectangle(canvas, (left, top), (right, bottom), (160, 160, 160), 1)

    draw_outline(canvas, truth, colours["truth"], thickness)
    draw_outline(canvas, prediction, colours["prediction"], thickness)
    return to_png(canvas)


def difference_view(image, truth, prediction, colours, opacity=0.55):
    """
    Where the two masks disagree, colour coded.

    Splitting the error into "missed" and "extra" says something Dice cannot:
    the same score can come from a model that under-segments and one that
    over-segments, and those two mistakes matter very differently when the mask
    is used to plan a radiotherapy margin.
    """
    canvas = as_colour(image)
    if truth is None or prediction is None:
        return to_png(canvas)

    overlap = np.logical_and(truth, prediction)
    missed = np.logical_and(truth, ~prediction)
    extra = np.logical_and(prediction, ~truth)

    wash = canvas.copy()
    wash[overlap] = colours["overlap"]
    wash[missed] = colours["missed"]
    wash[extra] = colours["extra"]

    blended = cv2.addWeighted(wash, opacity, canvas, 1 - opacity, 0)
    return to_png(draw_outline(blended, truth, colours["truth"], 1))


def heatmap_view(image, probabilities, opacity=0.5):
    """
    The raw probability map before it is thresholded.

    Worth showing because a confident wrong answer and an unsure one look
    identical once they have been cut at 0.5, and Task 2's failure cases are the
    confident kind.
    """
    canvas = as_colour(image)
    if probabilities is None:
        return to_png(canvas)

    scaled = np.clip(probabilities * 255, 0, 255).astype(np.uint8)
    coloured = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)

    # Leave the confidently-empty background as plain MRI, otherwise the whole
    # frame turns purple and the scan underneath is lost.
    weight = np.clip(probabilities / 0.15, 0, 1)[..., None] * opacity
    blended = coloured * weight + canvas * (1 - weight)
    return to_png(blended.astype(np.uint8))


def side_by_side(image, truth, prediction, colours):
    """
    Scan, ground truth and prediction in one strip.

    This is the view that gets exported for the report and for the group demo,
    where three separate screenshots would have to be lined up by hand.
    """
    panels = [
        as_colour(image),
        draw_outline(as_colour(image), truth, colours["truth"]),
        draw_outline(as_colour(image), prediction, colours["prediction"]),
    ]

    # "Scan" rather than "MRI": the group's ultrasound models go through the same
    # export, and labelling a breast ultrasound as an MRI in a report figure is
    # exactly the kind of mistake nobody notices until it is printed.
    labels = ["Scan", "Ground truth", "Prediction"]
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return to_png(np.hstack(panels))


VIEWS = ("scan", "truth", "prediction", "boundaries", "difference", "heatmap",
         "strip")


def build(view, image, truth=None, prediction=None, probabilities=None,
          box=None, palette_name="clinical", opacity=0.35, thickness=2):
    """One entry point so the web layer does not need a branch per view."""
    colours = palette(palette_name)

    if view == "scan":
        return scan(image)
    if view == "truth":
        return truth_view(image, truth, colours, opacity=opacity)
    if view == "prediction":
        return prediction_view(image, prediction, colours, opacity=opacity)
    if view == "difference":
        return difference_view(image, truth, prediction, colours)
    if view == "heatmap":
        return heatmap_view(image, probabilities)
    if view == "strip":
        return side_by_side(image, truth, prediction, colours)

    return boundary_view(image, truth, prediction, colours, box, thickness)

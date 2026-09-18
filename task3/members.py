"""
Who contributed which model, and where each one lives.

The brief asks for one Task 1 and one Task 2 model per student, so the GUI
lists exactly six entries named after their authors rather than after their
file names.  Everything that differs between the three sets of work - the
dataset, how the model is run, whether it is loaded live or read back from
recorded results - is declared here once, and backends.py does what this table
says instead of guessing from a file extension.

    ROSTER          the six entries, in the order the GUI shows them
    find(entry_id)  one of them by id, or None

Three "kinds" of entry exist because the three sets of work do not run the same
way:

    task1_forest   Salman's Random Forest, unpickled by the mvi-env worker
    keras          a network this process can load and run (Salman, Afnan)
    afnan_task1    Afnan's classical pipeline, ported from her notebook
    recorded       Aman's results, which are read from disk rather than run
"""

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Aman's two share folders.  The trailing " 2" is part of the folder name as it
# was handed over, so it is matched rather than tidied - renaming somebody
# else's folder is a good way to break their next hand-over.
AMMAN_TASK1_DIR = os.path.join(PROJECT_DIR, "amman", "Task1_GUI_share 2")
AMMAN_TASK2_DIR = os.path.join(PROJECT_DIR, "amman", "Task2_GUI_share 2")

AFNAN_DIR = os.path.join(PROJECT_DIR, "afnan")


ROSTER = [
    # ------------------------------------------------------------- Task 1
    {
        "id": os.path.join(PROJECT_DIR, "task1", "outputs", "rf_segmenter.joblib"),
        "name": "SALMAN (ME)",
        "task": 1,
        "dataset": "brisc",
        "kind": "task1_forest",
        "note": "BRISC 2025 brain MRI - Random Forest on 41 features per pixel, "
                "segmented from one box the user draws",
        "needs_box": True,
    },
    {
        "id": "afnan:task1",
        "name": "AFNAN",
        "task": 1,
        "dataset": "busi",
        "kind": "afnan_task1",
        "note": "BUSI breast ultrasound - multi-Otsu and blob proposals, scored "
                "and refined with a Chan-Vese contour",
        "needs_box": False,
    },
    {
        "id": "amman:task1",
        "name": "AMMAN",
        "task": 1,
        "dataset": "fives",
        "kind": "recorded",
        "results_dir": os.path.join(AMMAN_TASK1_DIR, "results"),
        "variant": None,
        "note": "FIVES retinal fundus - classical vessel segmentation, recorded "
                "results for all 200 test images",
        "needs_box": False,
    },

    # ------------------------------------------------------------- Task 2
    {
        "id": os.path.join(PROJECT_DIR, "task2", "outputs", "models",
                           "mha_resunet_pruned.h5"),
        "name": "SALMAN (ME)",
        "task": 2,
        "dataset": "brisc",
        "kind": "keras",
        "note": "BRISC 2025 brain MRI - MHA-ResUNet, structurally pruned: "
                "66% fewer FLOPs at the same Dice",
        "needs_box": False,
    },
    {
        "id": os.path.join(AFNAN_DIR, "innovation_seed42_pruned.keras"),
        "name": "AFNAN",
        "task": 2,
        "dataset": "busi",
        "kind": "keras",
        "note": "BUSI breast ultrasound - residual V-Net blocks, ASPP bottleneck "
                "and fused attention gates, pruned",
        "needs_box": False,
    },
    {
        "id": "amman:task2",
        "name": "AMMAN",
        "task": 2,
        "dataset": "fives",
        "kind": "recorded",
        "results_dir": os.path.join(AMMAN_TASK2_DIR, "results"),
        # Three variants were recorded; the pruned one is the final model.
        "variant": "pruned",
        "note": "FIVES retinal fundus - attention U-Net with clDice, structurally "
                "pruned and fine-tuned, recorded results",
        "needs_box": False,
    },
]

BY_ID = {entry["id"]: entry for entry in ROSTER}


def find(entry_id):
    """One roster entry by id, or None if this is a model the user typed in."""
    return BY_ID.get(entry_id)


def present(entry):
    """Whether the files this entry needs are actually on the machine."""
    if entry["kind"] == "recorded":
        return os.path.isdir(os.path.join(entry["results_dir"], "images"))
    if entry["kind"] == "afnan_task1":
        return True                      # pure code, nothing to find on disk
    return os.path.exists(entry["id"])

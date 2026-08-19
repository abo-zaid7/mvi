"""
Run the whole Task 2 pipeline end to end.

Run with:   python run_all.py

The stages have to happen in this order, because each one needs what the
previous produced:

    1. train the proposed MHA-ResUNet        (~80 min on an M1 Max GPU)
    2. PSO search for the U-Net baseline     (~30 min)
    3. train the U-Net with the PSO settings (~55 min)
    4. structurally prune and fine-tune      (~25 min)
    5. evaluate everything on 200 test scans (~5 min)

Individual stages can also be run on their own - see the README.
"""

import subprocess
import sys
import time

STAGES = [
    ("train the proposed MHA-ResUNet", [sys.executable, "train.py", "proposed"]),
    ("PSO hyperparameter search", [sys.executable, "pso.py"]),
    ("train the PSO-tuned U-Net", [sys.executable, "train.py", "unet"]),
    ("structural pruning + fine-tune", [sys.executable, "prune.py"]),
    ("evaluate on 200 test scans", [sys.executable, "evaluate.py"]),
]


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    total = time.time()

    for number, (title, command) in enumerate(STAGES, 1):
        if only and str(number) not in only:
            continue

        print("\n" + "=" * 70)
        print("stage %d/%d - %s" % (number, len(STAGES), title))
        print("=" * 70, flush=True)

        start = time.time()
        result = subprocess.run(command)
        if result.returncode != 0:
            print("\nstage %d failed with exit code %d, stopping"
                  % (number, result.returncode))
            return result.returncode
        print("stage %d finished in %.1f min" % (number, (time.time() - start) / 60))

    print("\nall stages done in %.1f min" % ((time.time() - total) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())

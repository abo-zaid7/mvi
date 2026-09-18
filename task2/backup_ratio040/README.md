# Task 2 at pruning ratio 0.40 - the version before the 2 M parameter limit

Kept because the lecturer's 2 M trainable-parameter limit arrived after this
version was finished and written up.  This is the state everything in
`Individual Report - Task 1 and Task 2.docx` and `Chapter 5` was measured from:

| model | trainable params | GFLOPs | size | test Dice |
|---|---|---|---|---|
| MHA-ResUNet, unpruned | 8,838,017 | 14.93 | 34.06 MB | 0.8692 |
| MHA-ResUNet, pruned 0.40 | 3,283,538 | 5.08 | 12.86 MB | 0.8699 |
| U-Net, PSO-tuned | 4,367,641 | 7.62 | 16.84 MB | 0.8352 |

The pruned model is over the limit at 3.28 M, which is why a deeper prune was
run.  Nothing here has been changed by that run.

## Restoring it

    cd task2
    cp backup_ratio040/models/*.h5* outputs/models/
    cp backup_ratio040/results_summary.csv backup_ratio040/results_per_image.csv \
       backup_ratio040/complexity.json backup_ratio040/pruning_summary.json outputs/

Then set `PRUNE_RATIO = 0.4` and `PRUNE_ROUNDS = 4` back in `config.py` if the
pruning stage is to be re-run at the old setting.

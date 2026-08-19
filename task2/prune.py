"""
Structural (channel) pruning of the trained MHA-ResUNet.

Why structural and not magnitude pruning:  the usual "set small weights to
zero" approach produces a sparse matrix that still has the same shape, so on
ordinary hardware it costs exactly as many FLOPs as before.  The brief asks for
pruning that *reduces FLOPs*, which means whole channels have to disappear and
the network has to be physically rebuilt narrower.  That is what happens here.

Method (L1-norm channel pruning, Li et al., ICLR 2017):

  1. for every block, score each output channel by the L1 norm of the filter
     that produces it
  2. keep the highest scoring (1 - PRUNE_RATIO) of them, rounded to a multiple
     of 8 so the tensor shapes stay hardware friendly
  3. rebuild the identical architecture at the smaller widths
  4. copy the surviving weights across, slicing every kernel on both its input
     and its output channel axis
  5. fine-tune briefly to recover the small amount of accuracy that was lost

Step 4 is what makes this worth doing: the pruned network starts from the
trained weights instead of from scratch, so fine-tuning takes a fraction of the
original training budget.

Run with:   python prune.py
"""

import json
import os
import time

import numpy as np

import complexity
import config
import data
import models
import train


# --------------------------------------------------------------- importance
def channel_scores(kernel, transposed=False):
    """
    L1 norm of each output filter.

    Conv2D kernels are (kh, kw, in, out) so we sum over everything but the last
    axis.  Conv2DTranspose kernels are (kh, kw, out, in), so the output axis is
    the third one instead.
    """
    axis = (0, 1, 3) if transposed else (0, 1, 2)
    return np.abs(kernel).sum(axis=axis)


def top_indices(scores, keep_count):
    """Indices of the strongest channels, returned in ascending order."""
    keep_count = int(np.clip(keep_count, 1, len(scores)))
    chosen = np.argsort(scores)[::-1][:keep_count]
    return np.sort(chosen)


def round_width(width, ratio):
    """
    New width after pruning, kept to a multiple of 4 and at least 8.

    Multiples of 4 rather than 8 because the rounding has to be fine enough for
    a narrow layer to actually shrink: at 12% per round a 32 channel layer
    rounds straight back to 32 on an 8 grid and never gets any smaller.  The
    final guard makes sure every round removes something, so repeated rounds
    cannot silently stall.
    """
    if ratio <= 0:
        return width
    new_width = max(8, int(round(width * (1 - ratio) / 4.0)) * 4)
    if new_width >= width:
        new_width = max(8, width - 4)
    return new_width


# --------------------------------------------------------------- weight copying
def slice_conv(old_layer, new_layer, in_idx, out_idx, transposed=False):
    weights = old_layer.get_weights()
    kernel = weights[0]

    if transposed:                        # (kh, kw, out, in)
        kernel = kernel[:, :, :, in_idx][:, :, out_idx, :]
    else:                                 # (kh, kw, in, out)
        kernel = kernel[:, :, in_idx, :][:, :, :, out_idx]

    new_weights = [kernel]
    if len(weights) > 1:                  # bias
        new_weights.append(weights[1][out_idx])
    new_layer.set_weights(new_weights)


def slice_batchnorm(old_layer, new_layer, idx):
    new_layer.set_weights([w[idx] for w in old_layer.get_weights()])


def copy_residual_block(old, new, prefix, in_idx, mid_idx, out_idx):
    """Transfer one residual block, which is 2 convs, 2 BNs and maybe a shortcut."""
    slice_conv(old.get_layer(prefix + "_conv1"), new.get_layer(prefix + "_conv1"),
               in_idx, mid_idx)
    slice_batchnorm(old.get_layer(prefix + "_bn1"), new.get_layer(prefix + "_bn1"), mid_idx)

    slice_conv(old.get_layer(prefix + "_conv2"), new.get_layer(prefix + "_conv2"),
               mid_idx, out_idx)
    slice_batchnorm(old.get_layer(prefix + "_bn2"), new.get_layer(prefix + "_bn2"), out_idx)

    # The shortcut is a 1x1 conv only when the block changes the channel count.
    # Pruning can make the input and output widths equal, in which case the
    # rebuilt block uses a plain identity shortcut and there is nothing to copy.
    try:
        old_short = old.get_layer(prefix + "_short")
        new_short = new.get_layer(prefix + "_short")
    except ValueError:
        return

    slice_conv(old_short, new_short, in_idx, out_idx)
    slice_batchnorm(old.get_layer(prefix + "_short_bn"),
                    new.get_layer(prefix + "_short_bn"), out_idx)


def copy_attention_gate(old, new, prefix, skip_idx, gate_idx, mid_idx, se_idx):
    slice_conv(old.get_layer(prefix + "_theta"), new.get_layer(prefix + "_theta"),
               skip_idx, mid_idx)
    slice_conv(old.get_layer(prefix + "_phi"), new.get_layer(prefix + "_phi"),
               gate_idx, mid_idx)
    slice_conv(old.get_layer(prefix + "_psi"), new.get_layer(prefix + "_psi"),
               mid_idx, np.array([0]))
    slice_conv(old.get_layer(prefix + "_se1"), new.get_layer(prefix + "_se1"),
               gate_idx, se_idx)
    slice_conv(old.get_layer(prefix + "_se2"), new.get_layer(prefix + "_se2"),
               se_idx, skip_idx)


def recalibrate_batchnorm(model, batches, n_batches=100):
    """
    Re-estimate the BatchNorm running statistics after pruning.

    This step matters far more than it looks.  Removing 40% of a convolution's
    *input* channels means every surviving output channel loses roughly 40% of
    the contributions that formed it, so its pre-activation distribution shifts.
    The inherited moving_mean and moving_var describe the old distribution, and
    once that error has cascaded through nine blocks the network output
    collapses - the transferred model scores near zero Dice even though the
    convolution weights themselves came across perfectly.

    The fix is cheap: push a few hundred batches through in training mode, which
    makes BatchNorm recompute its statistics, and apply no gradients at all.
    Dropout is switched off first so the statistics are estimated from clean
    activations.
    """
    from tensorflow import keras

    dropouts = [l for l in model.layers if isinstance(l, keras.layers.Dropout)]
    saved_rates = [l.rate for l in dropouts]
    for layer in dropouts:
        layer.rate = 0.0

    # start from a clean slate so the new statistics are purely data driven
    for layer in model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            gamma, beta, mean, variance = layer.get_weights()
            layer.set_weights([gamma, beta, np.zeros_like(mean), np.ones_like(variance)])

    used = min(n_batches, len(batches))
    for i in range(used):
        images, _ = batches[i]
        model(images, training=True)          # forward only - no optimiser step

    for layer, rate in zip(dropouts, saved_rates):
        layer.rate = rate

    return used


def copy_attention_bottleneck(old, new, idx):
    """LayerNorm and the MHA projections, all sliced on the channel axis."""
    slice_batchnorm(old.get_layer("attn_ln"), new.get_layer("attn_ln"), idx)

    old_mha = old.get_layer("attn_mha")
    new_mha = new.get_layer("attn_mha")
    transferred = []

    for old_w, new_w in zip(old_mha.get_weights(), new_mha.get_weights()):
        if old_w.shape == new_w.shape:
            transferred.append(old_w)                     # bias terms, unchanged
        elif old_w.ndim == 3 and old_w.shape[0] != new_w.shape[0]:
            transferred.append(old_w[idx])                # query/key/value: (channels, heads, key_dim)
        elif old_w.ndim == 3 and old_w.shape[2] != new_w.shape[2]:
            transferred.append(old_w[:, :, idx])          # output proj: (heads, key_dim, channels)
        elif old_w.ndim == 1:
            transferred.append(old_w[idx])
        else:
            transferred.append(new_w)                     # shape we cannot map, keep the fresh one
    new_mha.set_weights(transferred)


# --------------------------------------------------------------- the pruner
def prune_model(old_model, old_filters, ratio, depth=4):
    """Build the narrower network and move the surviving weights into it."""
    new_filters = [round_width(w, ratio) for w in old_filters]
    print("widths %s  ->  %s" % (old_filters, new_filters))

    input_size = old_model.inputs[0].shape[1]
    new_model = models.build_mha_resunet(input_size=input_size,
                                         filters=new_filters,
                                         heads=config.ATTENTION_HEADS,
                                         dropout=0.1)

    stages = ["enc%d" % (i + 1) for i in range(depth)] + ["bottleneck"] + \
             ["dec%d" % (i + 1) for i in range(depth)]

    # ---- which channels survive in every block
    # Widths run enc1..enc4, bottleneck, then dec1..dec4.  Decoder level i is
    # paired with encoder level i, so dec1 is the *narrowest* decoder block, not
    # the widest - hence new_filters[:-1] rather than a reversed list.
    stage_widths = new_filters + new_filters[:-1]
    keep_out, keep_mid = {}, {}
    for stage, new_width in zip(stages, stage_widths):
        keep_out[stage] = top_indices(
            channel_scores(old_model.get_layer(stage + "_conv2").get_weights()[0]), new_width)
        keep_mid[stage] = top_indices(
            channel_scores(old_model.get_layer(stage + "_conv1").get_weights()[0]), new_width)

    # ---- encoder
    previous = np.array([0])                       # the input image has one channel
    for level in range(depth):
        stage = "enc%d" % (level + 1)
        copy_residual_block(old_model, new_model, stage, previous,
                            keep_mid[stage], keep_out[stage])
        previous = keep_out[stage]

    # ---- bottleneck and its attention
    copy_residual_block(old_model, new_model, "bottleneck", previous,
                        keep_mid["bottleneck"], keep_out["bottleneck"])
    copy_attention_bottleneck(old_model, new_model, keep_out["bottleneck"])
    previous = keep_out["bottleneck"]

    # ---- decoder
    for level in reversed(range(depth)):
        stage = "dec%d" % (level + 1)
        skip_stage = "enc%d" % (level + 1)
        width = new_filters[level]

        up_old = old_model.get_layer("up%d" % (level + 1))
        keep_up = top_indices(channel_scores(up_old.get_weights()[0], transposed=True), width)
        slice_conv(up_old, new_model.get_layer("up%d" % (level + 1)),
                   previous, keep_up, transposed=True)

        gate = "gate%d" % (level + 1)
        keep_gate_mid = top_indices(
            channel_scores(old_model.get_layer(gate + "_theta").get_weights()[0]), width)
        se_width = max(width // 8, 4)
        keep_se = top_indices(
            channel_scores(old_model.get_layer(gate + "_se1").get_weights()[0]), se_width)
        copy_attention_gate(old_model, new_model, gate, keep_out[skip_stage],
                            keep_up, keep_gate_mid, keep_se)

        # the concatenation puts the upsampled tensor first, then the gated skip
        old_up_width = old_model.get_layer("up%d" % (level + 1)).get_weights()[0].shape[2]
        concat_idx = np.concatenate([keep_up, keep_out[skip_stage] + old_up_width])

        copy_residual_block(old_model, new_model, stage, concat_idx,
                            keep_mid[stage], keep_out[stage])
        previous = keep_out[stage]

    # ---- output layer
    slice_conv(old_model.get_layer("output"), new_model.get_layer("output"),
               previous, np.array([0]))

    return new_model, new_filters


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    print("loading the trained model ...")
    old_model, architecture = train.load_model(config.PROPOSED_MODEL)
    before = complexity.summarise(old_model, config.PROPOSED_MODEL)
    complexity.print_summary("before pruning", before)

    train_batches, val_batches = train.prepare_data()

    # Take the channels out over several rounds instead of all at once.  Each
    # round removes a slice, re-estimates the BatchNorm statistics and gives the
    # network a couple of epochs to settle before the next cut.
    rounds = config.PRUNE_ROUNDS
    per_round = 1 - (1 - config.PRUNE_RATIO) ** (1.0 / rounds)
    print("\npruning %.0f%% of the channels over %d rounds (%.1f%% each) ..."
          % (100 * config.PRUNE_RATIO, rounds, 100 * per_round))

    new_model = old_model
    new_filters = architecture["filters"]
    journey = []
    start = time.time()

    for round_number in range(1, rounds + 1):
        print("\n--- round %d / %d ---" % (round_number, rounds))
        new_model, new_filters = prune_model(new_model, new_filters, per_round)
        train.compile_model(new_model, config.LEARNING_RATE / 5)

        transfer_only = new_model.evaluate(val_batches, verbose=0,
                                           return_dict=True)["dice_coefficient"]
        used = recalibrate_batchnorm(new_model, train_batches,
                                     config.BN_RECALIBRATION_BATCHES)
        recalibrated = new_model.evaluate(val_batches, verbose=0,
                                          return_dict=True)["dice_coefficient"]

        history = train.fit(new_model, train_batches, val_batches,
                            epochs=config.PRUNE_ROUND_EPOCHS, patience=config.PRUNE_ROUND_EPOCHS,
                            verbose=0)
        recovered = max(history.history["val_dice_coefficient"])

        print("  widths %s" % new_filters)
        print("  transfer %.4f  ->  BN recalibration %.4f  ->  %d epochs %.4f"
              % (transfer_only, recalibrated, config.PRUNE_ROUND_EPOCHS, recovered))
        journey.append({"round": round_number, "filters": list(new_filters),
                        "dice_after_transfer": float(transfer_only),
                        "dice_after_bn_recalibration": float(recalibrated),
                        "dice_after_recovery": float(recovered)})

    print("\nfinal fine-tune for %d epochs ..." % config.PRUNE_FINETUNE_EPOCHS)
    history = train.fit(new_model, train_batches, val_batches,
                        epochs=config.PRUNE_FINETUNE_EPOCHS, patience=6)
    minutes = (time.time() - start) / 60

    transfer_only = {"dice_coefficient": journey[-1]["dice_after_transfer"]}
    recalibrated = {"dice_coefficient": journey[-1]["dice_after_bn_recalibration"]}
    used = config.BN_RECALIBRATION_BATCHES

    pruned_architecture = {"kind": "mha_resunet", "input_size": config.IMAGE_SIZE,
                           "filters": new_filters,
                           "heads": config.ATTENTION_HEADS, "dropout": 0.1}
    train.save_model(new_model, config.PRUNED_MODEL, pruned_architecture)

    after = complexity.summarise(new_model, config.PRUNED_MODEL)
    print("\n================ pruning summary ================")
    complexity.print_summary("before pruning", before)
    complexity.print_summary("after pruning", after)
    print("  FLOPs      %.2f -> %.2f GFLOPs  (%.0f%% fewer)"
          % (before["gflops"], after["gflops"],
             100 * (1 - after["gflops"] / before["gflops"])))
    print("  parameters %s -> %s  (%.0f%% fewer)"
          % ("{:,}".format(before["trainable_params"]),
             "{:,}".format(after["trainable_params"]),
             100 * (1 - after["trainable_params"] / before["trainable_params"])))
    print("  size       %.2f -> %.2f MB" % (before["size_mb"], after["size_mb"]))
    print("  latency    %.1f -> %.1f ms" % (1000 * before["latency_s"],
                                            1000 * after["latency_s"]))
    print("  fine-tuned in %.1f min, best validation Dice %.4f"
          % (minutes, max(history.history["val_dice_coefficient"])))
    print("\n  round by round:")
    for step in journey:
        print("    %d. %-28s transfer %.4f -> BN recal %.4f -> recovered %.4f"
              % (step["round"], str(step["filters"]), step["dice_after_transfer"],
                 step["dice_after_bn_recalibration"], step["dice_after_recovery"]))

    with open(os.path.join(config.OUTPUT_DIR, "pruning_summary.json"), "w") as handle:
        json.dump({"ratio": config.PRUNE_RATIO,
                   "old_filters": architecture["filters"], "new_filters": new_filters,
                   "before": before, "after": after,
                   "rounds": config.PRUNE_ROUNDS,
                   "per_round_ratio": round(1 - (1 - config.PRUNE_RATIO) ** (1.0 / config.PRUNE_ROUNDS), 4),
                   "journey": journey,
                   "recalibration_batches": used,
                   "val_dice_after_finetune": float(max(history.history["val_dice_coefficient"])),
                   "finetune_minutes": round(minutes, 1)}, handle, indent=2)


if __name__ == "__main__":
    main()

"""
The two networks compared in Task 2.

  build_unet()        - a conventional U-Net.  Its hyperparameters (width,
                        dropout, activation) are the ones the PSO search tunes,
                        so this is the baseline the proposed model has to beat.

  build_mha_resunet() - the proposed model.  Three things make it different
                        from the plain U-Net:

     1. residual encoder / decoder blocks, so gradients reach the deep layers
     2. multi-head self attention at the bottleneck.  A convolution only ever
        sees its own receptive field; MHA lets every position in the 12x12
        bottleneck grid look at every other position, which helps when a tumour
        has to be told apart from a symmetric structure on the other side of the
        brain.
     3. a dual attention gate on every skip connection (see below).

Every layer is given an explicit name, because the pruning code in prune.py
rebuilds the same graph at a smaller width and copies weights across by name.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import config


# --------------------------------------------------------------- building blocks
def residual_block(x, filters, name, activation="relu", dropout=0.0):
    """Conv-BN-act twice with a shortcut added back on."""
    shortcut = x
    if keras.backend.int_shape(x)[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                                 name=name + "_short")(shortcut)
        shortcut = layers.BatchNormalization(momentum=config.BN_MOMENTUM, name=name + "_short_bn")(shortcut)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=name + "_conv1")(x)
    x = layers.BatchNormalization(momentum=config.BN_MOMENTUM, name=name + "_bn1")(x)
    x = layers.Activation(activation, name=name + "_act1")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=name + "_drop")(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=name + "_conv2")(x)
    x = layers.BatchNormalization(momentum=config.BN_MOMENTUM, name=name + "_bn2")(x)

    x = layers.Add(name=name + "_add")([x, shortcut])
    return layers.Activation(activation, name=name + "_act2")(x)


def plain_block(x, filters, name, activation="relu", dropout=0.0):
    """The ordinary two convolution block used by the baseline U-Net."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=name + "_conv1")(x)
    x = layers.BatchNormalization(momentum=config.BN_MOMENTUM, name=name + "_bn1")(x)
    x = layers.Activation(activation, name=name + "_act1")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=name + "_drop")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=name + "_conv2")(x)
    x = layers.BatchNormalization(momentum=config.BN_MOMENTUM, name=name + "_bn2")(x)
    return layers.Activation(activation, name=name + "_act2")(x)


def dual_attention_gate(skip, gating, filters, name):
    """
    The novel skip gate.

    A standard Attention U-Net gate (Oktay et al., 2018) produces one spatial map
    that says *where* to look.  That is useful but it treats every feature
    channel as equally relevant, which is wasteful in a skip connection carrying
    256 channels of mostly healthy tissue.

    This gate adds a second, channel-wise branch (squeeze and excitation), so the
    skip is re-weighted by *where* to look and by *which features* matter:

        skip_out = skip * spatial_attention * channel_attention

    Both branches are driven by the decoder's gating signal, so the deeper,
    more semantic features decide what the shallow features are allowed to pass.
    """
    # ---- spatial branch: where to look
    theta = layers.Conv2D(filters, 1, padding="same", name=name + "_theta")(skip)
    phi = layers.Conv2D(filters, 1, padding="same", name=name + "_phi")(gating)
    joined = layers.Add(name=name + "_add")([theta, phi])
    joined = layers.Activation("relu", name=name + "_relu")(joined)
    spatial = layers.Conv2D(1, 1, padding="same", activation="sigmoid",
                            name=name + "_psi")(joined)

    # ---- channel branch: which features matter
    pooled = layers.GlobalAveragePooling2D(keepdims=True, name=name + "_gap")(gating)
    squeezed = layers.Conv2D(max(filters // 8, 4), 1, activation="relu",
                             name=name + "_se1")(pooled)
    channel = layers.Conv2D(keras.backend.int_shape(skip)[-1], 1, activation="sigmoid",
                            name=name + "_se2")(squeezed)

    gated = layers.Multiply(name=name + "_mul_spatial")([skip, spatial])
    return layers.Multiply(name=name + "_mul_channel")([gated, channel])


def attention_bottleneck(x, heads, name, key_dim=None):
    """
    Multi-head self attention over the bottleneck feature grid.

    key_dim is deliberately a fixed number rather than channels/heads.  If it
    scaled with the channel count then pruning the bottleneck would change the
    shape of the attention weights in a way that cannot be sliced, and the
    pruned model could not inherit them.  Holding it fixed means only the input
    and output projections shrink, and those slice cleanly.
    """
    _, height, width, channels = keras.backend.int_shape(x)
    key_dim = key_dim or config.ATTENTION_KEY_DIM

    sequence = layers.Reshape((height * width, channels), name=name + "_flat")(x)
    normed = layers.LayerNormalization(name=name + "_ln")(sequence)

    attended = layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim,
                                         name=name + "_mha")(normed, normed)
    sequence = layers.Add(name=name + "_res")([sequence, attended])
    return layers.Reshape((height, width, channels), name=name + "_unflat")(sequence)


# --------------------------------------------------------------- the models
def build_unet(input_size=None, base_filters=32, depth=4, dropout=0.1,
               activation="relu", name="unet"):
    """A conventional U-Net - the baseline whose hyperparameters PSO tunes."""
    size = input_size or config.IMAGE_SIZE
    inputs = keras.Input((size, size, 1), name="input")

    x = inputs
    skips = []
    for level in range(depth):
        filters = base_filters * (2 ** level)
        x = plain_block(x, filters, "enc%d" % (level + 1), activation, dropout)
        skips.append(x)
        x = layers.MaxPooling2D(name="pool%d" % (level + 1))(x)

    x = plain_block(x, base_filters * (2 ** depth), "bottleneck", activation, dropout)

    for level in reversed(range(depth)):
        filters = base_filters * (2 ** level)
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same",
                                   name="up%d" % (level + 1))(x)
        x = layers.Concatenate(name="cat%d" % (level + 1))([x, skips[level]])
        x = plain_block(x, filters, "dec%d" % (level + 1), activation, dropout)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="output")(x)
    return keras.Model(inputs, outputs, name=name)


def build_mha_resunet(input_size=None, filters=None, heads=None, dropout=0.1,
                      activation="relu", name="mha_resunet"):
    """
    The proposed model.

    `filters` is a list of five widths (four encoder levels plus the
    bottleneck).  Passing a smaller list is how prune.py builds the slimmed
    down version of this same architecture.
    """
    size = input_size or config.IMAGE_SIZE
    heads = heads or config.ATTENTION_HEADS
    if filters is None:
        base = config.BASE_FILTERS
        filters = [base, base * 2, base * 4, base * 8, base * 16]

    inputs = keras.Input((size, size, 1), name="input")

    # ---- encoder
    x = inputs
    skips = []
    for level, width in enumerate(filters[:-1]):
        x = residual_block(x, width, "enc%d" % (level + 1), activation, dropout)
        skips.append(x)
        x = layers.MaxPooling2D(name="pool%d" % (level + 1))(x)

    # ---- bottleneck with global context
    x = residual_block(x, filters[-1], "bottleneck", activation, dropout)
    x = attention_bottleneck(x, heads, "attn")

    # ---- decoder with the dual attention gates
    for level in reversed(range(len(filters) - 1)):
        width = filters[level]
        x = layers.Conv2DTranspose(width, 2, strides=2, padding="same",
                                   name="up%d" % (level + 1))(x)
        gated = dual_attention_gate(skips[level], x, width, "gate%d" % (level + 1))
        x = layers.Concatenate(name="cat%d" % (level + 1))([x, gated])
        x = residual_block(x, width, "dec%d" % (level + 1), activation, dropout)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="output")(x)
    return keras.Model(inputs, outputs, name=name)


def stage_names(depth=4):
    """The blocks whose width the pruner is allowed to change."""
    return (["enc%d" % (i + 1) for i in range(depth)] + ["bottleneck"] +
            ["dec%d" % (i + 1) for i in range(depth)])

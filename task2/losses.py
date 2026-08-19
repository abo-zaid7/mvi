"""
Loss functions and the Keras training metrics.

Tumours cover about 1-2% of a slice, so plain binary cross-entropy is happy to
predict "background everywhere" and still score 98% accuracy.  Adding a Dice
term fixes that, because Dice only looks at the overlap of the positive class.
The combination of the two is the usual choice in medical segmentation.
"""

import tensorflow as tf
from tensorflow import keras

SMOOTH = 1.0


def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    overlap = tf.reduce_sum(y_true * y_pred)
    return (2.0 * overlap + SMOOTH) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + SMOOTH)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """Half cross-entropy, half Dice."""
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.5 * tf.reduce_mean(bce) + 0.5 * dice_loss(y_true, y_pred)


def iou_metric(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]) > 0.5, tf.float32)
    overlap = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - overlap
    return (overlap + SMOOTH) / (union + SMOOTH)

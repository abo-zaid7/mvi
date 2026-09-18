
from __future__ import annotations

from typing import Tuple

import tensorflow as tf

VESSEL_PIXEL_VALUE = 255

MASK_THRESHOLD = 127.0

GREEN_CHANNEL_INDEX = 1

INPUT_MODES = ("green", "rgb")

def decode_image(path: tf.Tensor, input_mode: str = "green") -> tf.Tensor:

    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unknown input_mode '{input_mode}', expected one of {INPUT_MODES}")

    raw = tf.io.read_file(path)
    image = tf.io.decode_png(raw, channels=3)
    image = tf.cast(image, tf.float32)

    if input_mode == "green":
        image = image[..., GREEN_CHANNEL_INDEX : GREEN_CHANNEL_INDEX + 1]

    return image

def decode_mask(path: tf.Tensor) -> tf.Tensor:

    raw = tf.io.read_file(path)
    mask = tf.io.decode_png(raw, channels=1)
    return tf.cast(tf.cast(mask, tf.float32) > MASK_THRESHOLD, tf.float32)

def resize_image(image: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    return tf.image.resize(image, image_size, method="bilinear")

def resize_mask(mask: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:

    return tf.image.resize(mask, image_size, method="nearest")

def normalize_image(image: tf.Tensor) -> tf.Tensor:
    return image / 255.0

def preprocess_sample(
    image_path: tf.Tensor,
    mask_path: tf.Tensor,
    image_size: Tuple[int, int],
    input_mode: str = "green",
) -> Tuple[tf.Tensor, tf.Tensor]:

    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unknown input_mode '{input_mode}', expected one of {INPUT_MODES}")

    channels = 1 if input_mode == "green" else 3

    image = decode_image(image_path, input_mode)
    image = resize_image(image, image_size)
    image = normalize_image(image)

    mask = decode_mask(mask_path)
    mask = resize_mask(mask, image_size)

    image.set_shape((*image_size, channels))
    mask.set_shape((*image_size, 1))

    return image, mask

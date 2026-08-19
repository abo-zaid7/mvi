"""
Data loading and augmentation for the BRISC 2025 segmentation task.

Unlike Task 1 this is fully automatic - the network gets the whole slice and no
user input at all.  The scans are small enough that the entire dataset is held
in memory as uint8, which is far quicker than reading JPEGs every epoch.
"""

import glob
import os

import cv2
import numpy as np
from tensorflow import keras

import config


def list_images(split="train"):
    folder = config.TRAIN_IMAGES if split == "train" else config.TEST_IMAGES
    return sorted(glob.glob(os.path.join(folder, "*.jpg")))


def mask_path_for(image_path):
    return image_path.replace(os.sep + "images" + os.sep,
                              os.sep + "masks" + os.sep).replace(".jpg", ".png")


def preprocess(image):
    """CLAHE for contrast, same idea as Task 1 so the two are comparable."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)


def load_arrays(paths, size, with_masks=True):
    """Read every scan into one uint8 array of shape (n, size, size)."""
    images = np.zeros((len(paths), size, size), np.uint8)
    masks = np.zeros((len(paths), size, size), np.uint8) if with_masks else None

    for i, path in enumerate(paths):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        images[i] = preprocess(cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR))
        if with_masks:
            mask = cv2.imread(mask_path_for(path), cv2.IMREAD_GRAYSCALE)
            masks[i] = (cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) > 127)

    return (images, masks) if with_masks else images


def train_val_split(paths, val_fraction=None, seed=None):
    """Split the training folder into a training and a validation part."""
    val_fraction = config.VAL_FRACTION if val_fraction is None else val_fraction
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    order = rng.permutation(len(paths))
    cut = int(len(paths) * (1 - val_fraction))
    return [paths[i] for i in order[:cut]], [paths[i] for i in order[cut:]]


class Batches(keras.utils.Sequence):
    """
    Feeds batches to Keras, with light augmentation on the training set.

    The augmentations are the ones that make sense for brain MRI: flips and
    small rotations (the head is not always centred the same way), a little
    zoom, and a brightness/contrast jitter to mimic different scanners.
    Anything stronger tends to create anatomy that does not exist.
    """

    def __init__(self, images, masks, batch_size=None, augment=False, shuffle=None):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size or config.BATCH_SIZE
        self.augment = augment
        self.shuffle = augment if shuffle is None else shuffle
        self.rng = np.random.default_rng(config.SEED)
        self.order = np.arange(len(images))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __getitem__(self, index):
        picked = self.order[index * self.batch_size:(index + 1) * self.batch_size]
        x = self.images[picked].astype(np.float32) / 255.0
        y = self.masks[picked].astype(np.float32)

        if self.augment:
            x, y = zip(*[self._augment(a, b) for a, b in zip(x, y)])
            x, y = np.stack(x), np.stack(y)

        return x[..., None], y[..., None]

    def _augment(self, image, mask):
        # Left-right only.  The brain is roughly symmetric across the midline so
        # a mirrored slice is still a plausible head, but flipping top to bottom
        # would put the skull base above the vertex and produce anatomy that
        # cannot occur.
        if self.rng.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]

        if self.rng.random() < 0.7:
            angle = self.rng.uniform(-15, 15)
            zoom = self.rng.uniform(0.9, 1.1)
            centre = (image.shape[1] / 2, image.shape[0] / 2)
            matrix = cv2.getRotationMatrix2D(centre, angle, zoom)
            size = (image.shape[1], image.shape[0])
            image = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST)

        if self.rng.random() < 0.5:
            image = np.clip(image * self.rng.uniform(0.85, 1.15)
                            + self.rng.uniform(-0.08, 0.08), 0, 1)

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

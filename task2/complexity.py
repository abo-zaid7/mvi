"""
Model complexity: FLOPs, tunable parameters, size on disk and latency.

The brief asks for all of these for Task 2, and they are also the numbers that
back up the carbon footprint argument - a model that needs fewer FLOPs draws
less power for every scan it ever segments.

FLOPs are read out of the TensorFlow graph profiler, which counts a
multiply-add as two operations, so the figures reported here are the usual
"2 x MACs" convention.
"""

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


def count_flops(model):
    """Forward pass FLOPs for a single image."""
    spec = tf.TensorSpec([1] + list(model.inputs[0].shape[1:]), model.inputs[0].dtype)
    concrete = tf.function(lambda x: model(x)).get_concrete_function(spec)
    frozen = convert_variables_to_constants_v2(concrete)

    options = ProfileOptionBuilder.float_operation()
    options["output"] = "none"          # keep the profiler quiet
    info = profile(frozen.graph, options=options)
    return int(info.total_float_ops)


def count_parameters(model):
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(model.count_params())
    return trainable, total


def size_on_disk_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def measure_latency(model, runs=30, warmup=5):
    """Seconds per image for a batch of one, which is how it runs in a clinic."""
    shape = [1] + list(model.inputs[0].shape[1:])
    dummy = np.random.rand(*shape).astype("float32")

    for _ in range(warmup):
        model.predict(dummy, verbose=0)

    start = time.time()
    for _ in range(runs):
        model.predict(dummy, verbose=0)
    return (time.time() - start) / runs


def summarise(model, path=None, measure_time=True):
    """Every complexity number for one model, as a dictionary."""
    trainable, total = count_parameters(model)
    row = {
        "flops": count_flops(model),
        "gflops": round(count_flops(model) / 1e9, 3),
        "trainable_params": trainable,
        "total_params": total,
    }
    if path and os.path.exists(path):
        row["size_mb"] = round(size_on_disk_mb(path), 2)
    if measure_time:
        row["latency_s"] = round(measure_latency(model), 4)
    return row


def print_summary(name, row):
    print("%-22s  %7.3f GFLOPs | %8s params | %6s MB | %6.1f ms"
          % (name, row["gflops"], "{:,}".format(row["trainable_params"]),
             row.get("size_mb", "-"), 1000 * row.get("latency_s", 0)))

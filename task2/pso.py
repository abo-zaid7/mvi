"""
Particle Swarm Optimisation for the baseline U-Net's hyperparameters.

The brief asks for the proposed model to be compared against "a conventional
U-Net with hyperparameters tuned by Particle Swarm Optimization", so the
baseline is given every chance rather than being left on default settings.

Four hyperparameters are searched:

    base_filters   8 - 48     width of the first encoder level
    dropout        0.0 - 0.4
    activation     relu / elu (a continuous value that gets rounded)
    learning_rate  1e-4 - 5e-3 (searched on a log scale)

PSO itself is only about twenty lines: every particle remembers its own best
position, the swarm remembers the global best, and each step the velocity is
pulled towards both.

    v <- w*v + c1*r1*(personal_best - x) + c2*r2*(global_best - x)
    x <- x + v

Fitness is the validation Dice after a short training run on a subset of the
data at a smaller resolution - full trainings for every particle would take
days, and the ranking of candidates is what matters here, not the absolute
score.

Run with:   python pso.py
"""

import json
import os
import time

import numpy as np

import config
import data
import models
import train

# name, lower bound, upper bound
SEARCH_SPACE = [
    ("base_filters", 8, 48),
    ("dropout", 0.0, 0.4),
    ("activation", 0.0, 1.99),      # < 1 -> relu, >= 1 -> elu
    ("log_learning_rate", np.log10(1e-4), np.log10(5e-3)),
]

INERTIA = 0.7
COGNITIVE = 1.5
SOCIAL = 1.5


def decode(position):
    """Turn a raw particle position into usable hyperparameters."""
    return {
        "base_filters": int(round(position[0] / 4.0) * 4) or 4,   # multiples of 4
        "dropout": float(np.clip(position[1], 0.0, 0.4)),
        "activation": "relu" if position[2] < 1.0 else "elu",
        "learning_rate": float(10 ** position[3]),
    }


class Fitness:
    """Trains a small U-Net for a few epochs and reports the validation Dice."""

    def __init__(self):
        print("loading the PSO subset (%d scans at %d px) ..."
              % (config.PSO_SUBSET, config.PSO_SIZE))
        self.train_batches, self.val_batches = train.prepare_data(
            size=config.PSO_SIZE, subset=config.PSO_SUBSET)
        self.cache = {}

    def __call__(self, params):
        key = (params["base_filters"], round(params["dropout"], 3),
               params["activation"], round(params["learning_rate"], 6))
        if key in self.cache:
            return self.cache[key]

        model = models.build_unet(input_size=config.PSO_SIZE,
                                  base_filters=params["base_filters"],
                                  depth=4,
                                  dropout=params["dropout"],
                                  activation=params["activation"])
        train.compile_model(model, params["learning_rate"])

        history = model.fit(self.train_batches, validation_data=self.val_batches,
                            epochs=config.PSO_EPOCHS, verbose=0)
        score = float(max(history.history["val_dice_coefficient"]))

        del model
        self.cache[key] = score
        return score


def run_pso():
    rng = np.random.default_rng(config.SEED)
    n_particles = config.PSO_PARTICLES
    n_dimensions = len(SEARCH_SPACE)

    lower = np.array([low for _, low, _ in SEARCH_SPACE])
    upper = np.array([high for _, _, high in SEARCH_SPACE])
    span = upper - lower

    positions = rng.uniform(lower, upper, (n_particles, n_dimensions))
    velocities = rng.uniform(-0.2 * span, 0.2 * span, (n_particles, n_dimensions))

    fitness = Fitness()
    personal_best = positions.copy()
    personal_scores = np.full(n_particles, -np.inf)
    global_best, global_score = None, -np.inf

    history = []
    start = time.time()

    for iteration in range(config.PSO_ITERATIONS):
        print("\n--- PSO iteration %d / %d ---" % (iteration + 1, config.PSO_ITERATIONS))

        for i in range(n_particles):
            params = decode(positions[i])
            score = fitness(params)
            print("  particle %d  filters=%2d dropout=%.2f %-4s lr=%.1e  ->  Dice %.4f"
                  % (i, params["base_filters"], params["dropout"],
                     params["activation"], params["learning_rate"], score))

            history.append({"iteration": iteration, "particle": i,
                            "score": score, **params})

            if score > personal_scores[i]:
                personal_scores[i] = score
                personal_best[i] = positions[i].copy()
            if score > global_score:
                global_score = score
                global_best = positions[i].copy()

        # move the swarm
        r1 = rng.random((n_particles, n_dimensions))
        r2 = rng.random((n_particles, n_dimensions))
        velocities = (INERTIA * velocities
                      + COGNITIVE * r1 * (personal_best - positions)
                      + SOCIAL * r2 * (global_best - positions))
        velocities = np.clip(velocities, -0.3 * span, 0.3 * span)
        positions = np.clip(positions + velocities, lower, upper)

        print("  best so far: Dice %.4f  %s" % (global_score, decode(global_best)))

    minutes = (time.time() - start) / 60
    best_params = decode(global_best)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(config.PSO_RESULT, "w") as handle:
        json.dump({"best_params": best_params,
                   "best_score": global_score,
                   "search_minutes": round(minutes, 1),
                   "evaluations": len(history),
                   "settings": {"particles": n_particles,
                                "iterations": config.PSO_ITERATIONS,
                                "epochs_per_eval": config.PSO_EPOCHS,
                                "subset": config.PSO_SUBSET,
                                "image_size": config.PSO_SIZE},
                   "history": history}, handle, indent=2)

    print("\n============ PSO finished in %.1f min ============" % minutes)
    print("best validation Dice : %.4f" % global_score)
    print("best hyperparameters : %s" % best_params)
    print("written to", config.PSO_RESULT)
    return best_params


if __name__ == "__main__":
    run_pso()

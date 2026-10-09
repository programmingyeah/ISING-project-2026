import math

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

# h
#opties

L = 512
T = 2.269
h = 0.0
J = 1.0

EQUILIBRATION_SWEEPS = 50
MEASUREMENT_SWEEPS = 200
SAMPLE_EVERY = 2
BLOCK_SIZE = 20
SEED = 1234

#functies
@njit(cache=True)
def set_seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def wolff_step(spins, p_neighbour, p_ghost, h, marks, sites_i, sites_j, tag):
    L = spins.shape[0]
    row = np.random.randint(L)
    col = np.random.randint(L)
    spin = spins[row, col]

    marks[row, col] = tag
    sites_i[0] = row
    sites_j[0] = col
    cluster_size = 1
    current = 0
    touches_ghost = False
    ghost_spin = 1 if h >= 0 else -1

    while current < cluster_size:
        i = sites_i[current]
        j = sites_j[current]
        current += 1

        if h != 0 and spin == ghost_spin and np.random.random() < p_ghost:
            touches_ghost = True

        for direction in range(4):
            if direction == 0:
                ni, nj = (i - 1) % L, j
            elif direction == 1:
                ni, nj = (i + 1) % L, j
            elif direction == 2:
                ni, nj = i, (j - 1) % L
            else:
                ni, nj = i, (j + 1) % L

            if marks[ni, nj] != tag and spins[ni, nj] == spin:
                if np.random.random() < p_neighbour:
                    marks[ni, nj] = tag
                    sites_i[cluster_size] = ni
                    sites_j[cluster_size] = nj
                    cluster_size += 1

    if not touches_ghost:
        for k in range(cluster_size):
            spins[sites_i[k], sites_j[k]] *= -1

    return cluster_size, tag + 1


@njit(cache=True)
def sweep(spins, p_neighbour, p_ghost, h, marks, sites_i, sites_j, tag):
    visited = 0
    while visited < spins.size:
        cluster_size, tag = wolff_step(
            spins, p_neighbour, p_ghost, h, marks, sites_i, sites_j, tag
        )
        visited += cluster_size
    return tag


def mean_and_error(values):
    values = np.asarray(values)
    number_of_blocks = len(values) // BLOCK_SIZE
    blocks = values[:number_of_blocks * BLOCK_SIZE].reshape(number_of_blocks, BLOCK_SIZE)
    block_means = blocks.mean(axis=1)
    return block_means.mean(), block_means.std(ddof=1) / math.sqrt(number_of_blocks)


def energy_per_spin(spins):
    bonds = np.sum(spins * np.roll(spins, -1, axis=0))
    bonds += np.sum(spins * np.roll(spins, -1, axis=1))
    return (-J * bonds - h * spins.sum()) / spins.size

#main
rng = np.random.default_rng(SEED)
p_neighbour = 1 - math.exp(-2 * J / T)
p_ghost = 1 - math.exp(-2 * abs(h) / T)

#als T dichtbij 0 is kan p_neighbour afronden naar 1.0 door floating point errors
if p_neighbour == 1.0:
    starting_spin = 1 if h >= 0 else -1
    spins = np.full((L, L), starting_spin, dtype=np.int8)
else:
    spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)

marks = np.zeros((L, L), dtype=np.int32)
sites_i = np.empty(L * L, dtype=np.int32)
sites_j = np.empty(L * L, dtype=np.int32)
set_seed(SEED)
tag = 1

print("equilibration: 0%", flush=True)
next_progress = 10
for update in range(1, EQUILIBRATION_SWEEPS + 1):
    tag = sweep(spins, p_neighbour, p_ghost, h, marks, sites_i, sites_j, tag)
    percent = 100 * update // EQUILIBRATION_SWEEPS
    if percent >= next_progress:
        print(f"equilibration: {percent}%", flush=True)
        next_progress += 10

absolute_magnetization = []
energy = []
print("measurement: 0%", flush=True)
next_progress = 10
for update in range(1, MEASUREMENT_SWEEPS + 1):
    tag = sweep(spins, p_neighbour, p_ghost, h, marks, sites_i, sites_j, tag)
    if update % SAMPLE_EVERY == 0:
        m = spins.mean()
        absolute_magnetization.append(abs(m))
        energy.append(energy_per_spin(spins))
    percent = 100 * update // MEASUREMENT_SWEEPS
    if percent >= next_progress:
        print(f"measurement: {percent}%", flush=True)
        next_progress += 10

mean_abs_m, error_abs_m = mean_and_error(absolute_magnetization)
mean_e, error_e = mean_and_error(energy)

#output
print(f"mean |magnetization|:   {mean_abs_m:.4f} ± {error_abs_m:.4f}")
print(f"mean energy per spin:   {mean_e:+.4f} ± {error_e:.4f}")

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(spins, cmap="binary", vmin=-1, vmax=1, interpolation="nearest")
ax.set(title=f"2D Ising model at T = {T}, h = {h}", xticks=[], yticks=[])
fig.tight_layout()
plt.show()

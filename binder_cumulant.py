import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


#opties

#resoluties
L_values = [8,16,32,64,128,256,512]

#temperatuur domein
T_CENTER = 2.27
T_UNCERTAINTY = 0.006
RANGE_SIGMAS = 5
N_TEMPERATURES = 21
T_values = np.linspace(
    T_CENTER - RANGE_SIGMAS * T_UNCERTAINTY,
    T_CENTER + RANGE_SIGMAS * T_UNCERTAINTY,
    N_TEMPERATURES,
)
SLOPE_POINTS = 5

#modelparameters
h = 0.0
J = 1.0


EQUILIBRATION_UPDATES = 50
MEASUREMENT_UPDATES = 200
SAMPLE_EVERY = 1
N_RUNS = 5
N_WORKERS = os.cpu_count() or 1
SEED = 1234

#functies

@njit(cache=True)
def set_seed(seed):
    np.random.seed(seed)


@njit(cache=True, nogil=True)
def wolff_step(spins, p_neighbour, marks, sites_i, sites_j, tag):
    L = spins.shape[0]
    row = np.random.randint(L)
    col = np.random.randint(L)
    spin = spins[row, col]

    marks[row, col] = tag
    sites_i[0] = row
    sites_j[0] = col
    cluster_size = 1
    current = 0

    while current < cluster_size:
        i = sites_i[current]
        j = sites_j[current]
        current += 1

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

    for k in range(cluster_size):
        spins[sites_i[k], sites_j[k]] *= -1

    return cluster_size, tag + 1


@njit(cache=True, nogil=True)
def sweep(spins, p_neighbour, marks, sites_i, sites_j, tag):
    visited = 0
    while visited < spins.size:
        cluster_size, tag = wolff_step(spins, p_neighbour, marks, sites_i, sites_j, tag)
        visited += cluster_size
    return tag


def run(L, T, seed):
    rng = np.random.default_rng(seed)
    spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)
    marks = np.zeros((L, L), dtype=np.int32)
    sites_i = np.empty(L * L, dtype=np.int32)
    sites_j = np.empty(L * L, dtype=np.int32)
    p_neighbour = 1 - math.exp(-2 * J / T)

    set_seed(seed)
    tag = 1
    for _ in range(EQUILIBRATION_UPDATES):
        tag = sweep(spins, p_neighbour, marks, sites_i, sites_j, tag)

    m2 = []
    m4 = []
    for update in range(1, MEASUREMENT_UPDATES + 1):
        tag = sweep(spins, p_neighbour, marks, sites_i, sites_j, tag)
        if update % SAMPLE_EVERY == 0:
            m = spins.mean()
            m2.append(m ** 2)
            m4.append(m ** 4)
    return m2, m4


def binder(m2, m4):
    return 1 - np.mean(m4) / (3 * np.mean(m2) ** 2)


def crossing(T_values, U1, U2):
    difference = np.asarray(U1) - np.asarray(U2)
    possible_crossings = []

    for i in range(len(T_values) - 1):
        if difference[i] * difference[i + 1] <= 0:
            fraction = difference[i] / (difference[i] - difference[i + 1])
            possible_crossings.append(T_values[i] + fraction * (T_values[i + 1] - T_values[i]))

    if not possible_crossings:
        return None
    return min(possible_crossings, key=lambda value: abs(value - T_CENTER))


def binder_slope(T_values, U_values):
    middle = np.argmin(abs(T_values - T_CENTER))
    start = max(0, middle - SLOPE_POINTS // 2)
    stop = min(len(T_values), start + SLOPE_POINTS)
    start = stop - SLOPE_POINTS
    return abs(np.polyfit(T_values[start:stop], U_values[start:stop], 1)[0])


def run_task(L, T, repeat):
    seed = SEED + 1000 * L + 100 * repeat + int(round(T * 1_000_000))
    m2, m4 = run(L, T, seed)
    return L, T, repeat, m2, m4


tasks = [(L, float(T), repeat) for L in L_values for T in T_values for repeat in range(N_RUNS)]
results = {}

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = [executor.submit(run_task, L, T, repeat) for L, T, repeat in tasks]
    for done, future in enumerate(as_completed(futures), 1):
        L, T, repeat, m2, m4 = future.result()
        results[L, T, repeat] = m2, m4
        print(f"[{done}/{len(tasks)}] L={L}, T={T:.3f}", flush=True)

curves = {}
run_curves = {}
for L in L_values:
    curves[L] = []
    run_curves[L] = [[] for _ in range(N_RUNS)]
    for T in T_values:
        m2 = []
        m4 = []
        for repeat in range(N_RUNS):
            m2_run, m4_run = results[L, float(T), repeat]
            m2.extend(m2_run)
            m4.extend(m4_run)
            run_curves[L][repeat].append(binder(m2_run, m4_run))
        curves[L].append(binder(m2, m4))

fig, ax = plt.subplots(figsize=(8, 5))
for L in L_values:
    ax.plot(T_values, curves[L], "o-", ms=3, label=f"L = {L}")

ax.set(
    title="Binder cumulants",
    xlabel="temperature T",
    ylabel="Binder cumulant U",
)
ax.legend()
fig.tight_layout()
fig.savefig("binder_cumulants.png", dpi=180)
plt.show()

for L1, L2 in zip(L_values[:-1], L_values[1:]):
    Tc_values = []
    for repeat in range(N_RUNS):
        Tc = crossing(T_values, run_curves[L1][repeat], run_curves[L2][repeat])
        if Tc is not None:
            Tc_values.append(Tc)

    if not Tc_values:
        print(f"No crossing found for L = {L1} and L = {L2}")
    else:
        Tc = np.mean(Tc_values)
        if len(Tc_values) > 1:
            error = np.std(Tc_values, ddof=1) / math.sqrt(len(Tc_values))
            print(f"Tc from L = {L1} and L = {L2}: {Tc:.5f} ± {error:.5f}")
        else:
            print(f"Tc from L = {L1} and L = {L2}: {Tc:.5f}")


slopes = {}
for L in L_values:
    slopes[L] = np.array([binder_slope(T_values, values) for values in run_curves[L]])

mean_slopes = np.array([slopes[L].mean() for L in L_values])
fit = np.polyfit(np.log(L_values), np.log(mean_slopes), 1)
nu = 1 / fit[0]

rng = np.random.default_rng(SEED)
nu_values = []
for _ in range(1000):
    bootstrap_slopes = []
    for L in L_values:
        values = slopes[L]
        bootstrap_slopes.append(rng.choice(values, len(values)).mean())
    bootstrap_fit = np.polyfit(np.log(L_values), np.log(bootstrap_slopes), 1)
    nu_values.append(1 / bootstrap_fit[0])

nu_error = np.std(nu_values, ddof=1)
print(f"nu from Binder slopes: {nu:.4f} ± {nu_error:.4f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(L_values, mean_slopes, "o", label="measured slopes")
ax.loglog(L_values, np.exp(fit[1]) * np.asarray(L_values) ** fit[0], "-", label=f"1/nu = {fit[0]:.3f}")
ax.set(xlabel="lattice size L", ylabel=r"$|dU/dT|$")
ax.legend()
fig.tight_layout()
fig.savefig("binder_slopes.png", dpi=180)
plt.show()

import math
import os  #om te zien hoeveel cpu cores er zijn
from concurrent.futures import ThreadPoolExecutor, as_completed  #voor meerdere runs tegelijk

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


#opties

#resoluties
L_values = [64, 128, 256, 512]  #alle rooster groottes die we vergelijken

#temperatuur domein
T_CENTER = 2.269185  #beste huidige schatting voor Tc
T_UNCERTAINTY = 0.001  #onzekerheid van die schatting
RANGE_SIGMAS = 3  #scan van center min 5 sigma tot center plus 5 sigma
N_TEMPERATURES = 31  #zoveel temperaturen in de scan
T_values = np.linspace(  #maak een gelijk verdeelde temperatuur grid
    T_CENTER - RANGE_SIGMAS * T_UNCERTAINTY,  #onderkant van het domein
    T_CENTER + RANGE_SIGMAS * T_UNCERTAINTY,  #bovenkant van het domein
    N_TEMPERATURES,  #aantal punten in dat domein
)
SLOPE_POINTS = 15  #aantal punten rond Tc voor een binder slope fit

#model parameters
J = 1.0  #standaard koppeling

EQUILIBRATION_UPDATES = 100 #om wat sneller te gaan hebben we minder sweeps dan ising.py
MEASUREMENT_UPDATES = 600
SAMPLE_EVERY = 1
N_RUNS = 5  #onafhankelijke random runs per combinatie van L en T, dit is eigenlijk equivalent aan gwn meer metingen nemen
#maar: repeats laten ons een grote run opsplitsen in meerdere kleine, die kunnen dan over meerdere CPU cores verspreid worden
#voor performance, elke repeat heeft ook zijn eigen RNG dus vermindert het een beetje de correlatie
N_WORKERS = os.cpu_count() or 1  #laat zoveel taken tegelijk op cpu cores draaien
SEED = 12346


#functies

@njit(cache=True)
def set_seed(seed):
    np.random.seed(seed)


#zelfde als ising.py
@njit(cache=True, nogil=True)  #nogil laat verschillende python threads tegelijk werken
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


#zelfde als in ising.py
@njit(cache=True, nogil=True)
def sweep(spins, p_neighbour, marks, sites_i, sites_j, tag):
    visited = 0
    while visited < spins.size:
        cluster_size, tag = wolff_step(spins, p_neighbour, marks, sites_i, sites_j, tag)
        visited += cluster_size
    return tag


def run(L, T, seed):  #een volledige simulatie voor een (L,T), vooral zelfde als ising.py, alleen andere meetwaarden
    rng = np.random.default_rng(seed)
    spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)  #willekeurige beginstoestand
    marks = np.zeros((L, L), dtype=np.int32)
    sites_i = np.empty(L * L, dtype=np.int32)
    sites_j = np.empty(L * L, dtype=np.int32)
    p_neighbour = 1 - math.exp(-2 * J / T)

    set_seed(seed)
    tag = 1
    for _ in range(EQUILIBRATION_UPDATES): #evenwichtstoestand bereiken
        tag = sweep(spins, p_neighbour, marks, sites_i, sites_j, tag)

    m2_total = 0.0  #tel alle m^2 metingen op
    m4_total = 0.0  #tel alle m^4 metingen op
    samples = 0  #houd bij hoeveel metingen we echt nemen

    for update in range(1, MEASUREMENT_UPDATES + 1):  #neem de metingen
        tag = sweep(spins, p_neighbour, marks, sites_i, sites_j, tag)  #doe een sweep
        if update % SAMPLE_EVERY == 0:
            m = spins.mean()  #magnetizatie epr spin
            m2_total += m ** 2  #voeg m^2 toe aan de som
            m4_total += m ** 4  #voeg m^4 toe aan de som
            samples += 1  #er is nu 1 sample meer
    return m2_total / samples, m4_total / samples  #geef alleen <m2> en <m4> terug


def binder(mean_m2, mean_m4):  #bereken U = 1 - <m4> / (3 <m2>^2)
    return 1 - mean_m4 / (3 * mean_m2 ** 2)


def crossing(T_values, U1, U2):  #zoek waar 2 binder curves elkaar snijden
    difference = np.asarray(U1) - np.asarray(U2)  #verschil tussen beide curves bij elke T
    possible_crossings = []  #alle kandidaten voor een snijpunt doen we hier

    for i in range(len(T_values) - 1):  #neem elke paar opeenvolgende temperaturen
        if difference[i] * difference[i + 1] <= 0:  #veranderd de temperatuur van teken? dan zit hier een snijpunt tussen
            fraction = difference[i] / (difference[i] - difference[i + 1])  #lineaire interpolatie om de nulpunt te schatten
            possible_crossings.append(T_values[i] + fraction * (T_values[i + 1] - T_values[i]))  #bereken nu de bijbehorende T

    if not possible_crossings:  #geen snijpunten in de domein?
        return None  #geef niets terug zodat we dit kunnen melden
    return min(possible_crossings, key=lambda value: abs(value - T_CENTER)) #neem de snijpunt die het dichts bij T_c ligt


def binder_slope(T_values, U_values):  #schat |dU/dT| rondom Tc
    middle = np.argmin(abs(T_values - T_CENTER))  #zoek het T punt het dichtst bij Tc
    half = SLOPE_POINTS // 2  #neem evenveel punten links als rechts
    T_fit = T_values[middle - half:middle + half + 1]  #kleine T window rond Tc
    U_fit = U_values[middle - half:middle + half + 1]  #bijbehorende binder waarden
    return abs(np.polyfit(T_fit, U_fit, 1)[0])  #lineaire fit en pak de helling


def run_task(L, T, repeat):  #elke thread runt deze functie
    seed = SEED + 1000 * L + 100 * repeat + int(round(T * 1_000_000))  #unieke seed voor L,T,repeat
    mean_m2, mean_m4 = run(L, T, seed)  #doe de simulatie
    return L, T, repeat, mean_m2, mean_m4  #stuur de momenten terug naar de main thread

#voor elke combinatie van L,T en repeat willen we een simulatie hebben
tasks = [(L, float(T), repeat) for L in L_values for T in T_values for repeat in range(N_RUNS)]
results = {}  #hier bewaren we de resultaten

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:  #start de worker threads
    futures = [executor.submit(run_task, L, T, repeat) for L, T, repeat in tasks]  #geef elke taak aan een worker
    for done, future in enumerate(as_completed(futures), 1):  #pak taken op zodra ze klaar zijn
        L, T, repeat, mean_m2, mean_m4 = future.result()  #haal het resultaat van die run op
        results[L, T, repeat] = mean_m2, mean_m4  #bewaar de twee gemiddelden
        print(f"[{done}/{len(tasks)}] L={L}, T={T:.3f}", flush=True)

#nu binder cumulants berekenen uit de meetwaardes, waaruit we later de snijpunt berekenen

#voor elke L,T combineer de repeat runs
curves = {}  #de gemiddelde binder curve voor elke L
run_curves = {}  #binder curves apart per repeat,voor de errors later
for L in L_values:  #voor elke roostergrootte:
    curves[L] = [] #begin met een lege gemiddelde binder curve
    run_curves[L] = [[] for _ in range(N_RUNS)]  #geef iedere repeat zijn eigen curve lijst
    for T in T_values:  #neem nu elke temperatuur:
        m2_values = []  #gemiddelde m2 van elke repeat
        m4_values = []  #gemiddelde m4 van elke repeat
        for repeat in range(N_RUNS):  #en neem elke run voor deze L,T combinatie
            mean_m2, mean_m4 = results[L, float(T), repeat]  #pak <m2> en <m4> van deze repeat
            m2_values.append(mean_m2)  #bewaar <m2> van deze repeat
            m4_values.append(mean_m4)  #bewaar <m4> van deze repeat
            run_curves[L][repeat].append(binder(mean_m2, mean_m4))  #bewaar U per repeat voor de errors
        curves[L].append(binder(np.mean(m2_values), np.mean(m4_values)))  #U met alle repeats samen

fig, ax = plt.subplots(figsize=(8, 5))
for L in L_values:  #teken 1 curve voor iedere grootte
    ax.plot(T_values, curves[L], "o-", ms=3, label=f"L = {L}") #teken ook de lijnen tussen punten om snijpunten te laten zien

ax.set(
    title="Binder cumulants",
    xlabel="temperature T",
    ylabel="Binder cumulant U",
)
ax.legend()
fig.tight_layout()
fig.savefig("binder_cumulants.png", dpi=180)
plt.show()

#vergelijk steeds twee opeenvolgende L waarden om hun Tc snijpunt te vinden
for L1, L2 in zip(L_values[:-1], L_values[1:]):
    Tc_values = []  #elke repeat geeft een eigen snijpunt schatting
    for repeat in range(N_RUNS):  #loop door onafhankelijke repeats
        Tc = crossing(T_values, run_curves[L1][repeat], run_curves[L2][repeat])  #zoek die snijpunt
        if Tc is not None:  #gebruik alleen repeats die echt een snijpunt vinden
            Tc_values.append(Tc)  #bewaar de Tc sschatting

    if not Tc_values: #check of er wel snijpunten zijn gevonden
        print(f"No crossing found for L = {L1} and L = {L2}")
    else:  #anders kunnen we een gemiddelde maken
        Tc = np.mean(Tc_values)  #gemiddelde Tc van alle repeats
        if len(Tc_values) > 1:  #alleen dan bestaat er een spreiding
            error = np.std(Tc_values, ddof=1) / math.sqrt(len(Tc_values))  #standard error over repeats
            print(f"Tc from L = {L1} and L = {L2}: {Tc:.5f} ± {error:.5f}")  #print Tc plus error
        else:  #met 1 repeat kan je geen error berekenen
            print(f"Tc from L = {L1} and L = {L2}: {Tc:.5f}")  #print alleen Tc


#nu halen we de kritieke exponent 'nu' uit de schaal van de binder hellingen
slopes = {}  #bewaar de hellingen voor iedere L
for L in L_values:  #voor elke L:
    slopes[L] = np.array([binder_slope(T_values, values) for values in run_curves[L]])  #sla op 1 slope per repeat

mean_slopes = np.array([slopes[L].mean() for L in L_values])  #gemiddelde helling per L
fit = np.polyfit(np.log(L_values), np.log(mean_slopes), 1)  #log slope = (1/nu) log L + constante

#onzekerheid in nu uit de repeats
nu_values = []  #1 nu schatting per repeat
for repeat in range(N_RUNS):  #doe dezelfde exponent fit voor iedere repeat
    repeat_slopes = [binder_slope(T_values, run_curves[L][repeat]) for L in L_values]
    repeat_fit = np.polyfit(np.log(L_values), np.log(repeat_slopes), 1)
    nu_values.append(1 / repeat_fit[0])  #de nu is 1/helling

nu = 1 / fit[0]
nu_error = np.std(nu_values, ddof=1) / math.sqrt(N_RUNS)
print(f"nu from Binder slopes: {nu:.4f} ± {nu_error:.4f}")

fig, ax = plt.subplots(figsize=(6, 4))  #tweede plot voor de 'nu' fits
ax.loglog(L_values, mean_slopes, "o", label="measured slopes")  #meetpunten in log-log ruimte
ax.loglog(L_values, np.exp(fit[1]) * np.asarray(L_values) ** fit[0], "-", label=f"1/nu = {fit[0]:.3f}")  # fit lijn
ax.set(xlabel="lattice size L", ylabel=r"$|dU/dT|$")
ax.legend()
fig.tight_layout()
fig.savefig("binder_slopes.png", dpi=180)
plt.show()

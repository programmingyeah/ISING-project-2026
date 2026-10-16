import math

import matplotlib.pyplot as plt
import numpy as np
from numba import njit  #maakt de trage functies sneller
import scipy.odr as odr


#opties

L = 512 
T = 2.269185 
J = 1.0  #sterkte waarmee buren willen alignen

T_left = 0.01
T_right = 4
T_step_size = 0.01

EQUILIBRATION_SWEEPS = 50  #eerst zoveel sweeps om een evenwichtstoestand te bereiken (voor de metingen)
MEASUREMENT_SWEEPS = 20  #daarna zoveel sweeps voor de metingen zelf
SAMPLE_EVERY = 2  #elke SAMPLE_EVERY sweeps wordt een meting, dus totaal metingen = MEASUREMENT_SWEEPS/SAMPLE_EVERY
BLOCK_SIZE = 5  #aantal metingen in een blok voor bepalen van onzekerheden
SEED = 1234  #zelfde seed = zelfde run


#functies

@njit(cache=True)
def set_seed(seed): #numba kan niet de python RNG gebruiken, dus configureren we de seed van numba's eigen RNG met de oude SEED
    np.random.seed(seed)


@njit(cache=True)
def wolff_step(spins, p_neighbour, marks, sites_i, sites_j, tag): #maakt een cluster
    L = spins.shape[0]  #geen global parameters, dus L afleiden uit de grootte van de spins array
    row = np.random.randint(L)  #willekeurige rij
    col = np.random.randint(L)  #willekeurige kolom
    spin = spins[row, col]  #neem nu de spin die daar zit

    marks[row, col] = tag  #markeer deze spin als deel van de cluster, note: stel dat we geen marks array hadden, om te
    # weten of op een (i,j) in het raster een spin zit, zou je door elke elementen van sites_i and sites_j moeten kijken
    # dit zou heel traag zijn, zeker voor grote cluster_size, marks[i, j] is maar een lookup, puur voor efficientie

    #marks wordt herbruikt, en het getal tag is de index van deze cluster, na alle wolff steps is marks dan bevolkt met tag's 0,1,2...n_clusters
    #en is elke cluster een regio in marks waar de elementen gelijk staan aan de clusters tag

    sites_i[0] = row  #rij en kolom van de eerste spin opslaan, over het algemeen: de rij en kolom van de n'de spin is 
    # (sites_i[n],sites_j[n])
    sites_j[0] = col  

    cluster_size = 1  #er zit nu 1 spin in de cluster
    current = 0  #we zitten op de eerste spin, beginnen met 0

    while current < cluster_size:  #ga door tot elke cluster spin bekeken is
        i = sites_i[current]  #pak de rij en kolom van de huidige cluster spin
        j = sites_j[current] 
        current += 1  #volgende keer nemen we de volgende cluster spin (als er nog meer zijn)

        for direction in range(4):  #check de 4 buren
            if direction == 0: 
                ni, nj = (i - 1) % L, j  # modulo geeft periodieke randen
            elif direction == 1: 
                ni, nj = (i + 1) % L, j  
            elif direction == 2:
                ni, nj = i, (j - 1) % L 
            else: 
                ni, nj = i, (j + 1) % L

            if marks[ni, nj] != tag and spins[ni, nj] == spin:  #check voor een nieuwe buur, moet niet al in de cluster zitten
                # en moet dezelfde spin hebben al de cluster
                if np.random.random() < p_neighbour:  #maak met kans p een bond
                    marks[ni, nj] = tag  #dit behoort nu tot de cluster
                    sites_i[cluster_size] = ni  #rij en kolom opslaan
                    sites_j[cluster_size] = nj  
                    cluster_size += 1  #nu hebben we een spin toegevoegd

    for k in range(cluster_size):  #neem alle spins in de cluster
        spins[sites_i[k], sites_j[k]] *= -1  #flip de spins

    return cluster_size, tag + 1  #geef grootte terug plus een nieuwe tag


@njit(cache=True)
def sweep(spins, p_neighbour, marks, sites_i, sites_j, tag): #doe ongeveer LxL wolff steps (kan minder zijn met grote wolff steps)
    visited = 0  #totaal aantal spins die geflipt zijn
    while visited < spins.size:  #stop zodra we ongeveer 1 volledig rooster bezocht hebben
        cluster_size, tag = wolff_step(  #flip een cluster
            spins, p_neighbour, marks, sites_i, sites_j, tag
        )

        visited += cluster_size 
    return tag  #geef de nieuwste tag terug


def mean_and_error(values):  #gemiddelde en blocked standard error
    values = np.asarray(values)  #maak er een numpy array van
    number_of_blocks = len(values) // BLOCK_SIZE  #hoeveel blocks kunnen erin passen? // is afgeronde deling
    blocks = values[:number_of_blocks * BLOCK_SIZE].reshape(number_of_blocks, BLOCK_SIZE)  #knip in blocks
    block_means = blocks.mean(axis=1)  #neem voor elk het gemiddelde, er is correlatie dus geen error hier
    return block_means.mean(), block_means.std(ddof=1) / math.sqrt(number_of_blocks)  #mean en error tussen de minder gecorreleerde blocks


def energy_per_spin(spins):
    bonds = np.sum(spins * np.roll(spins, -1, axis=0))  #alle vetricale bond energieen
    bonds += np.sum(spins * np.roll(spins, -1, axis=1))  #alle horizontale bond energieen
    return -J * bonds / spins.size  #deel de totale energie door het aantal spins


def heat_capacity_and_error(energy, temp):  
    energy = np.asarray(energy)  
    heat_capacity = L * L * np.var(energy, ddof=1) / temp**2  #c per spin is N var(e) / T^2

    number_of_blocks = len(energy) // BLOCK_SIZE 
    blocks = energy[:number_of_blocks * BLOCK_SIZE].reshape(number_of_blocks, BLOCK_SIZE) 
    block_heat_capacities = L * L * np.var(blocks, axis=1, ddof=1) / temp**2  #c voor ieder block
    error = block_heat_capacities.std(ddof=1) / math.sqrt(number_of_blocks)  # spreiding van block c's
    return heat_capacity, error  #geef c en de error terug


#main

rng = np.random.default_rng(SEED)  #normale numpy RNG voor de beginswaarden

def main(temp):

    p_neighbour = 1 - math.exp(-2 * J / temp)  #kans om een cluster uit te breiden

    #als T dichtbij 0 is kan p_neighbour afronden naar 1.0 door floating point errors
    if p_neighbour == 1.0:  #dan is een uniforme grondtoestand het veiligst
        spins = np.full((L, L), 1, dtype=np.int8)  #kies 1 van de 2 gelijke grondtoestanden
        #spin is of +1 of -1 dus een standaard int64 zou een verspilling van geheugen zijn
    else:  #bij gewone temperaturen starten we random
        spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)

    marks = np.zeros((L, L), dtype=np.int32)  #onthoudt later welke spin al in de huidige cluster zit
    sites_i = np.empty(L * L, dtype=np.int32)  #rijen van cluster spins
    sites_j = np.empty(L * L, dtype=np.int32)  #kolommen van cluster spins
    set_seed(SEED)  #zet ook numba's random seed
    tag = 1  #eerste unieke label voor clusters

    #print("equilibration: 0%", flush=True)
    #next_progress = 10
    for update in range(1, EQUILIBRATION_SWEEPS + 1):  #begin de burn-in sweeps
        tag = sweep(spins, p_neighbour, marks, sites_i, sites_j, tag)  #standaard update
        """
        percent = 100 * update // EQUILIBRATION_SWEEPS
        if percent >= next_progress:  #print alleen elke 10 procent
            print(f"equilibration: {percent}%", flush=True)  #laat de voortgang zien
            next_progress += 10
        """

    absolute_magnetization = []  #|m| metingen
    energy = []  #energie per spin metingen
    #print("measurement: 0%", flush=True)
    #next_progress = 10
    for update in range(1, MEASUREMENT_SWEEPS + 1):  #begin de meet sweeps
        tag = sweep(spins, p_neighbour, marks, sites_i, sites_j, tag)  #standaard update
        if update % SAMPLE_EVERY == 0:  #niet elke sweep meten om correlatie te verminderen
            m = spins.mean()  #magnetisatie per spin van deze configuratie
            absolute_magnetization.append(abs(m))  #|m| opslaan
            energy.append(energy_per_spin(spins))  #energie per spin opslaan
        """
        percent = 100 * update // MEASUREMENT_SWEEPS
        if percent >= next_progress:
            print(f"measurement: {percent}%", flush=True)
            next_progress += 10
        """

    mean_abs_m, error_abs_m = mean_and_error(absolute_magnetization)  #gemiddelde |m| plus error
    mean_e, error_e = mean_and_error(energy)  #gemiddelde energie plus error
    heat_capacity_c, heat_capacity_error = heat_capacity_and_error(energy, temp)  #c plus error

    return [mean_abs_m, error_abs_m, mean_e, error_e, heat_capacity_c, heat_capacity_error]


T = T_left
data = []

print("measurement: 0%", flush=True)
next_progress = 5

while T <= T_right:
    values = main(T)
    values.append(T)
    data.append(values)
    
    
    percent = 100 * (T - T_left) // (T_right - T_left)
    if percent >= next_progress:
        print(f"measurement: {percent}%", flush=True)
        next_progress += 5

    T += T_step_size


data = np.array(data)

T_values = data[:,-1]

# plot absolute magnetisation
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(T_values, data[:,0], "o-", ms=3) #, label=f"L = {L}") 

ax.set(
    title="Magnetisation",
    xlabel="temperature T",
    ylabel="<m>",
)
#ax.legend()
fig.tight_layout()
fig.savefig("graph_magnetisation.png", dpi=180)
plt.show()


# plot energy
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(T_values, data[:,2], "o-", ms=3) #, label=f"L = {L}") 

ax.set(
    title="energy",
    xlabel="temperature T",
    ylabel="<E>",
)
#ax.legend()
fig.tight_layout()
fig.savefig("graph_energy.png", dpi=180)
plt.show()


# plot heat capacity
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(T_values, data[:,4], "o", ms=3) #, label=f"L = {L}") 

ax.set(
    title="heat capacity",
    xlabel="temperature T",
    ylabel="<C>",
)
#ax.legend()
fig.tight_layout()
fig.savefig("graph_heat_capacity.png", dpi=180)
plt.show()




# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:33:01 2017

@author: capel102
Versie 2024

Met dit script kun je een fit aan een set data maken. Fitten is: parameters in 
een model zo aanpassen dat de modelwaarden het best overeenkomen met de gemeten data.
De wiskundige achtergrond van fitten en de precieze werking van de fitroutine 
wordt behandeld in het onderdeel DATA-V.

Als gebruiker geef je als input:
    (1) De data. We nemen aan dat de dataset bestaat uit waarden voor de onafhankelijke 
    grootheid (in het script x genoemd) met onzekerheden (sig_x), en bijbehorende
    waarden voor de afhankelijke grootheid (deze noemen we y) met onzekerheden (sig_y)
    (2) Het model
    (3) Startwaarden voor de parameters, zodat de fit vlakbij de beste parameterwaardes start
    
(4)-(7) zijn de commando's waarmee de fit wordt gedefinieerd en uitgevoerd. 
    
Als output krijg je:
    (8) De beste schatters voor parameterwaarden met onzekerheden en overige 
    statistische informatie (dit wordt in de loop van DATA-V behandeld)
    (9) Een (onopgemaakte) plot met data en het model met de beste schatters voor 
    de parameters ingevuld. 

De standaardopzet van het script doet een fit volgens het model y = A + Bx, en in de 
comments hebben we een kwadratisch model voorgeprogrammeerd. Je kunt zelf een ander
model definiÃ«ren dat goed bij jouw situatie past.
"""


##### Gebruikersinput

### (1) Fictieve dataset x- en y-waarden, beide met onzekerheid
x = np.array(T_values) # Onafhankelijke grootheid
#sig_x = np.array([0.1, 0.3, 0.2, 0.5, 0.1]) # Onzekerheid onafhankelijke grootheid - mag niet nul zijn!
y = np.array(data[:,4]) # Afhankelijke grootheid
sig_y = np.array(data[:,5]) # Onzekerheid afhankelijke grootheid - mag niet nul zijn!

### (2) Modelfunctie

# Rechte lijn y = A + Bx
# B is een vector met de parameters, in dit geval twee (A = B[0], B = B[1])
# x is de array met de x waarden
# def f(B, x):
#     return B[0] + B[1]*x

# Uitgecomment: parabool y = A + Bx + C x^2
# In dit geval heeft de vector B dus drie elementen (A = B[0], B = B[1], C = B[2])
# def f(B, x):
#     return B[0] + B[1]*x + B[2]*x**2

def f(B, x):
    return B[0] + B[1] * np.exp(
        -(x - B[2])**2 / (2 * B[3]**2)
    )

B_start = [0.5, 1.0, 2.3, 0.3]

### (3) Startwaarden voor parameters
# B_start = [0.5, 2.0] # Voor de rechte lijn
# B_start = [0.5, 2.0, 0.1] # Voor de parabool

# (3a) Het is goed om de dataset te plotten samen met het model met de startwaarden 
# ingevuld. Zo kun je inschatten of de startwaarden voor parameters goed zijn. 
plt.figure()
plt.errorbar(x,y,yerr=sig_y,fmt='ko')
plt.plot(x, f(B_start,x), 'b-')
plt.show()

#%%

##### Fit

###  (4) Definieer het model-object om te gebruiken in odr
odr_model = odr.Model(f)

### (5) Definieer een RealData object
## Een RealData-object vraagt om de onzekerheden in beide richtingen. 
## !! De onzekerheid in de x-richting mag ook nul zijn (dan mag je sx=0 weglaten), 
## maar dan moet bij onderdeel (6)/(6a) wel gekozen worden voor een
## kleinste-kwadratenaanpassing !!
odr_data  = odr.RealData(x,y,sy=sig_y)

### (6) Maak een ODR object met data, model en startwaarden
## Je geeft startwaarden voor parameters mee bij keyword beta0
odr_obj   = odr.ODR(odr_data,odr_model,beta0=B_start)
### (6a) Stel in op kleinste kwadraten (optioneel)
## Als de onzekerheden in de x-richting gelijk zijn aan nul, dan faalt de 
## default-aanpak van ODR. Je moet dan als methode een kleinste-kwadraten- 
## aanpassing instellen. Dit gaat met het volgende commando (nu uit):
odr_obj.set_job(fit_type=2)

### (7) Voer de fit uit
## Dit gebeurt expliciet door functie .run() aan te roepen
odr_res   = odr_obj.run()

#%%

##### Output

### (8) Haal resultaten uit het resultaten-object:
# (8a) De beste schatters voor de parameters
par_best = odr_res.beta
# (8b) De (EXTERNE!) onzekerheden voor deze parameters
par_sig_ext = odr_res.sd_beta
# (8c) De (INTERNE!) covariantiematrix
par_cov = odr_res.cov_beta 
print(" De (INTERNE!) covariantiematrix  = \n", par_cov)
# (8d) De chi-kwadraat en de gereduceerde chi-kwadraat van deze aanpassing
chi2 = odr_res.sum_square
print("\n Chi-squared         = ", chi2)
chi2red = odr_res.res_var
print(" Reduced chi-squared = ", chi2red, "\n")
# (8e) Een compacte weergave van de belangrijkste resultaten als output
odr_res.pprint()

### (9) Plot van de fit met de dataset. 
# De plot is onopgemaakt, je zult zelf asbijschriften moeten toevoegen, 
# plotbereik kiezen etc. om hem mooi te maken. 
# Om het model wat meer punten te geven, maken we daarvoor een aparte lijst x-waarden 
xplot = np.linspace(np.min(x),np.max(x),num=1000)
plt.figure()
plt.errorbar(x,y,yerr=sig_y,fmt='ko')
plt.plot(xplot,f(par_best,xplot),'r-')
plt.show()
# Naar keuze: exporteren als png of pdf in een directory naar keuze
#plt.savefig('fit.png', dpi=300, bbox_inches='tight')
#plt.savefig('fit.pdf', bbox_inches='tight')


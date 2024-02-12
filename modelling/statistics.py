import numpy as np
from scipy import stats
from attacker import Attacker


def simulate(attacker):
    nb_tries = 20000
    min, max = 0, 0
    tries = np.zeros(nb_tries)
    for i in range(0, nb_tries):
        if ((i / nb_tries) * 100) % 10 == 0:
            print(f"Progression de la simulation d'attaque : {(i / nb_tries) * 100}%")
        tries[i] = attacker.attack()
        if tries[i] < min:
            min = tries[i]
        if tries[i] > max:
            max = tries[i]
    return tries


def calculate_stats(attacker, all_p=None, all_K=None):
    if all_p is None and all_K is None :
        tries = simulate(attacker)
        print_stats(attacker, tries)
    elif all_K is None:
        means = np.zeros(len(all_p))
        stds = np.zeros(len(all_p))
        for i, p in enumerate(all_p):
            attacker.p = p
            print(f"Attaques en cours pour p = {p}")
            tries = simulate(attacker)
            means[i] = tries.mean()
            stds[i] = tries.std()
        return means, stds
    else:
        means = np.zeros(len(all_K))
        stds = np.zeros(len(all_K))
        for i, k in enumerate(all_K):
            attacker = Attacker(M=attacker.model.M, K=k, know_M=False, is_uniform=True)
            print(f"Attaques en cours pour K = {k}")
            tries = simulate(attacker)
            means[i] = tries.mean()
            stds[i] = tries.std()
        return means, stds


def calculate_confidence_interval(means, stds):
    all_confidence_intervals = []
    for i in range(len(means)):
        # Calcul de l'intervalle de confiance à 95%
        confidence_interval = stats.norm.interval(0.95, loc=means[i], scale=stds[i] / np.sqrt(20000))

        all_confidence_intervals.append(confidence_interval)

    return all_confidence_intervals
        #print("Moyenne des déplacements de l'attaquant:", means[i])
        #print("Intervalle de confiance (95%):", confidence_interval)


def print_stats(attacker, tries):
    str = f"Pour M = {attacker.model.M}, K = {attacker.model.n*(attacker.model.M-1)} et une probabilité de retour "
    if attacker.is_uniform:
        str += f"de {attacker.p} :"
    else:
        str += "variable selon la taille du chemin parcouru (1 - e^(0.1*k))"
    print(str)
    print(f"Nombre moyen de déplacement : {tries.mean()}")
    print(f"Nombre médian de déplacement : {np.median(tries)}")
    print(f"Écart type : {tries.std()}")
    print(f"Nombre minimal de déplacement : {np.min(tries)}")
    print(f"Nombre maximal de déplacement : {np.max(tries)}")
    print(f"Mode : {stats.mode(tries, keepdims=True).mode[0]}")
    print(f"Fréquence du mode : {stats.mode(tries, keepdims=True).count[0]}")




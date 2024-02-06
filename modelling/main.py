from modelling import Modelling
from attacker import Attacker
import numpy as np
#from scipy import stats
import matplotlib.pyplot as plt


if __name__ == '__main__':

    M = 9
    K = 8

    attacker = Attacker(M=M, K=K, know_M=False, is_uniform=True)

    attacker.model.plot_graph()

    all_p = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]

    # Pour voir l'animation d'une attaque
    # attacker.animate_attack(interval=100)

    means = np.zeros(len(all_p))

    print(f"Pour M = {M}, K = {K}: ")

    for p in all_p:
        # On lance nb_tries attaques
        print(f"Attaques en cours pour p = {p}")
        attacker.p = p
        nb_tries = 10000
        tries = np.zeros(nb_tries)
        for i in range(0, nb_tries):
            if ((i / nb_tries) * 100) % 10 == 0:
                print(f"Progression : {(i / nb_tries) * 100}%")
            tries[i] = attacker.attack()
        means = np.append(means, tries.mean())

    # Création du graphique
    plt.plot(all_p, means[len(all_p):])  # Exclure les zéros initiaux ajoutés à means
    plt.title(f'Moyennes des déplacement d\'un attaquant en fonction de p (M = {M}, K = {K})')
    plt.xlabel('p')
    plt.ylabel('Moyenne')
    plt.show()

    """
    print(f"Pour M = {M}, K = {K} et une probabilité de retour de 0.2 : ")
    print(f"Nombre moyen de déplacement : {tries.mean()}")
    print(f"Nombre médian de déplacement : {np.median(tries)}")
    print(f"Écart type : {tries.std()}")
    print(f"Nombre minimal de déplacement : {np.min(tries)}")
    print(f"Nombre maximal de déplacement : {np.max(tries)}")
    print(f"Mode : {stats.mode(tries, keepdims=True).mode[0]}")
    print(f"Fréquence du mode : {stats.mode(tries, keepdims=True).count[0]}")
    """


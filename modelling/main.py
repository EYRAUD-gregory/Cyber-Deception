from modelling import Modelling
from attacker import Attacker
import numpy as np
from scipy import stats


if __name__ == '__main__':

    M = 4
    K = 30

    attacker = Attacker(M=M, K=K, know_M=False, is_uniform=True)

    # attacker.model.plot_graph()

    # Pour voir l'animation d'une attaque
    # attacker.animate_attack(interval=100)


    # On lance nb_tries attaques
    nb_tries = 10000
    tries = np.zeros(nb_tries)
    for i in range(0, nb_tries):
        if ((i / nb_tries) * 100) % 10 == 0:
            print(f"Progression : {(i / nb_tries) * 100}%")
        tries[i] = attacker.attack()

    print(f"Pour M = {M}, K = {K} et une probabilité de retour de 0.6 : ")
    print(f"Nombre moyen de déplacement : {tries.mean()}")
    print(f"Nombre médian de déplacement : {np.median(tries)}")
    print(f"Écart type : {tries.std()}")
    print(f"Nombre minimal de déplacement : {np.min(tries)}")
    print(f"Nombre maximal de déplacement : {np.max(tries)}")
    print(f"Mode : {stats.mode(tries, keepdims=True).mode[0]}")
    print(f"Fréquence du mode : {stats.mode(tries, keepdims=True).count[0]}")

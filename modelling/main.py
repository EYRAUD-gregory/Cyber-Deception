from attacker import Attacker
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from statistics import calculate_stats, calculate_confidence_interval

if __name__ == '__main__':

    M = 4
    K = M-1

    attacker = Attacker(M=M, K=K, know_M=False, is_uniform=True)

    attacker.model.plot_graph()

    all_p = [0.1, 0.2, 0.3]
    all_K = [i for i in range(K, M * 4, M)] # On rajoute un honeypot par service à chaque itération
    #all_K = [3]

    # Pour voir l'animation d'une attaque
    # attacker.animate_attack(interval=100)

    means, stds = calculate_stats(attacker, all_K=all_K)
    #means, stds = calculate_stats(attacker, all_p=all_p)
    calculate_confidence_interval(means, stds)

    # Création du graphique
    #plt.plot(all_p, means[len(all_p):])  # Exclure les zéros initiaux ajoutés à means
    #plt.title(f'Moyenne des déplacement d\'un attaquant en fonction de K (M = {M}, p = {attacker.p})')
    #plt.xlabel('K')
   # plt.ylabel('Moyenne')
    #plt.show()



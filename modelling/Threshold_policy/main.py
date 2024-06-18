from attacker import Attacker
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
# from stats import calculate_stats, calculate_confidence_interval

gamma = 0.995
W = 1e5


def mean_reward():
    return

if __name__ == '__main__':

    M = 4
    K = 3

    attacker = Attacker(M=M, K=K, know_M=False, is_uniform=False, is_animated=False)

    attacker.attack()

    attacker.model.plot_graph()

    #calculate_stats(attacker)

    #all_p = [0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
    #all_K = [i for i in range(K, M * 5, K)] # On rajoute un honeypot par service à chaque itération

    """
    all_alpha = [i/100 for i in range(1, 31)]

    means, stds = calculate_stats(attacker, all_alpha=all_alpha)

    # Tracer le graphique
    plt.plot(all_alpha, means)
    plt.xlabel('all_alpha')
    plt.ylabel('Moyenne')
    plt.title(f'Moyenne des déplacements d\'un attaquant en fonction de alpha (M={M}, K={K})')
    plt.grid(True)
    plt.show()
    """
    # Pour voir l'animation d'une attaque
    #attacker.animate_attack(interval=100)


    #means, stds = calculate_stats(attacker, all_K=all_K)
    """
    #means, stds = calculate_stats(attacker, all_p=all_p)
    all_confidence_intervals = calculate_confidence_interval(means, stds)

    print(means)
    print(all_confidence_intervals)

    #for i in range(len(all_confidence_intervals)):
        #print("Moyenne des déplacements de l'attaquant:", means[i])
        #print("Intervalle de confiance (95%):", all_confidence_intervals[i])
    # Tracé des bornes de l'intervalle de confiance

    # Tracé des bornes de l'intervalle de confiance
    for i, (mean, std, confidence_interval) in enumerate(
            zip(means, stds, all_confidence_intervals)):
        lower_bound, upper_bound = all_confidence_intervals[i]
        plt.plot([all_K[i], all_K[i]], [lower_bound, upper_bound], color='blue', linewidth=2)
        plt.scatter(all_K[i], mean, color='red')  # Marquer la moyenne
    
    # Création du graphique
    plt.plot(all_K, means, label='Moyenne')  # Courbe de la moyenne
    plt.title(f'Moyenne des déplacements d\'un attaquant en fonction de K (M = {M}, p = {attacker.p})')
    plt.xlabel('K')
    plt.ylabel('Moyenne')
    plt.legend()
    plt.show()
"""

from modelling import Modelling
from attacker import Attacker
import numpy as np


if __name__ == '__main__':

    attacker = Attacker(M=50, K=49, know_M=False, is_uniform=True)

    attacker.model.plot_graph()

    # Pour voir l'animation d'une attaque
    # attacker.animate_attack(interval=100)

    """
    # On lance nb_tries attaque
    nb_tries = 100
    tries = np.zeros(nb_tries)
    for i in range(0, nb_tries):
        tries[i] = attacker.attack()

    print(tries.mean())  # Nombre moyen de déplacement avant d'arriver aux données sensibles
    """

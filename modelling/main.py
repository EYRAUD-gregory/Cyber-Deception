from modelling import Modelling
from attacker import Attacker
import numpy as np


if __name__ == '__main__':

    attacker = Attacker(M=5, K=8, know_M=False, is_uniform=True)

    attacker.animate_attack(interval=1)

    #print(test)

    """
    nb_tries = 100
    tries = np.zeros(nb_tries)
    for i in range(0, nb_tries):
        tries[i] = attacker.attack()

    # print(tries)
    print(tries.mean())

    """
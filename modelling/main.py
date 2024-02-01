from modelling import Modelling
from attacker import Attacker
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__ == '__main__':

    #model = Modelling(M=5, K=16)

    #G = model.create_graph()
    #model.plot_graph()

    attacker = Attacker(M=5, K=8, know_M=False, is_uniform=False)

    attacker.animate_attack(interval=1)

    #print(test)

    nb_tries = 100
    tries = np.zeros(nb_tries)
    for i in range(0, nb_tries):
        tries[i] = attacker.attack()

    # print(tries)
    print(tries.mean())

    """
    nb_tries = 100
    tries = np.zeros(nb_tries)
    for i in range(0, nb_tries):
        tries[i] = attacker.attack()

    #print(tries)
    print(tries.mean())
    """
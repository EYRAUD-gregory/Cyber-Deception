from modelling import Modelling
from attacker import Attacker
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__ == '__main__':

    model = Modelling(M=5, K=18)

    G = model.create_graph()
    model.plot_graph()

    attacker = Attacker(G, M=5, is_uniform=True)


    nb_tries = 10
    tries = np.zeros(nb_tries)
    for i in range(0, nb_tries):
        tries[i] = attacker.attack()

    #print(tries)
    print(tries.mean())

from modelling import Modelling
from attacker import Attacker

if __name__ == '__main__':

    model = Modelling(M=8, n=5)

    G = model.create_graph()
    model.plot_graph()

    attacker = Attacker(G)

    attacker.attack()

    #model.plot_graph()
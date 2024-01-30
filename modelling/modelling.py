import networkx as nx
import matplotlib.pyplot as plt
"""
import sys
sys.path.append('../Outils de Simulation')
from animation_attack import queue, animated
"""


class Modelling:
    def __init__(self, M, n):
        self.G = nx.DiGraph()
        self.M = M
        self.n = n

    def create_graph(self):

        node_id = 0

        for i in range(0, self.M):
            node_id += 1
            self.G.add_node(str(node_id))
            if 1 < node_id <= self.M:
                self.G.add_edge(str(node_id-1), str(node_id))
            for j in range(1, self.n+1):
                self.G.add_node(str(node_id) + "," + str(j))
                self.G.add_edge(str(node_id), str(node_id) + "," + str(j))

        return self.G

    def plot_graph(self):
        # Spécification des positions des nœuds
        """"
        pos = {'1': (0, 0), '2': (1, 0), '3': (2, 0), '4': (3, 0),
               '1,1': (-0.3, -1), '1,2': (0, -1), '1,3': (0.3, -1),
               '2,1': (0.6, -1), '2,2': (1, -1), '2,3': (1.3, -1),
               '3,1': (1.6, -1), '3,2': (2, -1), '3,3': (2.3, -1),
               '4,1': (2.6, -1), '4,2': (3, -1), '4,3': (3.3, -1)}
        """

        pos = {}

        x, y = 0, 0
        x_sub = -0.333
        y_sub = -1
        node_id = 0

        for i in range(0, self.M):
            node_id += 1
            pos[str(node_id)] = (x, y)
            for j in range(1, self.n+1):
                pos[str(node_id) + "," + str(j % 3 + 1)] = (x_sub, y_sub)
                x_sub += 0.333
            x += 1

        nx.draw(self.G, pos=pos, with_labels=True, font_weight='bold', node_size=700)

        plt.show()
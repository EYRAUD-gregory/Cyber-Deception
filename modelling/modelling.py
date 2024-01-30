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
        proba = 1/self.n

        for i in range(0, self.M):
            node_id += 1
            self.G.add_node(str(node_id))
            if 1 < node_id <= self.M:
                self.G.add_edge(str(node_id-1), str(node_id), weight=proba)

            for j in range(1, self.n+1):
                self.G.add_node(str(node_id) + "," + str(j))
                self.G.add_edge(str(node_id), str(node_id) + "," + str(j), weight=proba)

        return self.G

    def plot_graph(self):
        # Spécification des positions des nœuds
        pos = {}

        x, y = 0, 0
        x_sub = -0.333
        y_sub = -1
        node_id = 0

        for i in range(0, self.M):
            node_id += 1
            pos[str(node_id)] = (x, y)
            for j in range(1, self.n+1):
                pos[str(node_id) + "," + str(j)] = (x_sub, y_sub)
                x_sub += 0.333
            x += 1

        # Couleur initiale des nœuds
        node_colors = ['blue' for _ in self.G.nodes]

        # Modification de la couleur d'un nœud spécifique (par exemple, le nœud '1')
        node_colors[list(self.G.nodes).index('1')] = 'red'

        nx.draw(self.G, pos=pos, with_labels=True, font_weight='bold', node_size=700, node_color=node_colors)

        plt.show()
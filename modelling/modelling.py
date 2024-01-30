import networkx as nx
import matplotlib.pyplot as plt


class Modelling:
    def __init__(self, K, M):
        self.G = nx.DiGraph()
        self.M = M
        self.K = K

    def create_graph(self):

        self.G.add_nodes_from(['1', '2', '3', '4', '1,1', '1,2', '1,3', '2,1', '2,2', '2,3', '3,1', '3,2', '3,3', '4,1', '4,2', '4,3'])
        self.G.add_edge('1', '2')
        self.G.add_edge('2', '3')
        self.G.add_edge('3', '4')

        self.G.add_edge('1', '1,1')
        self.G.add_edge('1', '1,2')
        self.G.add_edge('1', '1,3')

        self.G.add_edge('2', '2,1')
        self.G.add_edge('2', '2,2')
        self.G.add_edge('2', '2,3')

        self.G.add_edge('3', '3,1')
        self.G.add_edge('3', '3,2')
        self.G.add_edge('3', '3,3')

        self.G.add_edge('4', '4,1')
        self.G.add_edge('4', '4,2')
        self.G.add_edge('4', '4,3')

        return self.G

    def plot_graph(self):
        # Spécification des positions des nœuds
        # Spécification des positions des nœuds
        pos = {'1': (0, 0), '2': (1, 0), '3': (2, 0), '4': (3, 0),
               '1,1': (-0.3, -1), '1,2': (0, -1), '1,3': (0.3, -1),
               '2,1': (0.6, -1), '2,2': (1, -1), '2,3': (1.3, -1),
               '3,1': (1.6, -1), '3,2': (2, -1), '3,3': (2.3, -1),
               '4,1': (2.6, -1), '4,2': (3, -1), '4,3': (3.3, -1)}

        nx.draw(self.G, pos=pos, with_labels=True, font_weight='bold')

        plt.show()
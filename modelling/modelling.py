import networkx as nx
import matplotlib.pyplot as plt


class Modelling:
    def __init__(self, M, K):
        self.G = nx.DiGraph()  # Le graphe dirigé
        self.M = M  # Le nombre de services à atteindre avant l'accès aux données sensibles
        # self.n = n  # Le nombre de leurres pour chaque service
        self.n = int(K / (M-1))  # Le nombre de leurres pour chaque service

    def create_graph(self): # TODO mettre un type à chaque noeud pour savoir ce que c'est (et surtout pouvoir définir l'état final)

        # Id unique du noeud
        node_id = 0
        # Probabilité de transition pour chaque arrête
        proba = 1/(self.n+1)

        for i in range(0, self.M):
            node_id += 1
            if node_id == self.M:
                self.G.add_node(str(node_id), type='Goal')
            elif node_id == 1:
                self.G.add_node(str(node_id), type='Start')
            else:
                self.G.add_node(str(node_id), type='Service')
            if 1 < node_id:  # Si ce noeud n'est pas le premier, ça veut dire qu'on peut le relier avec le précédent
                self.G.add_edge(str(node_id-1), str(node_id), weight=proba)
            if node_id < self.M:
                for j in range(1, self.n+1):  # Pour chaque noeud, on créé n leurres tous reliés au service
                    self.G.add_node(str(node_id) + "," + str(j), type="Honeypot")
                    self.G.add_edge(str(node_id), str(node_id) + "," + str(j), weight=proba)

        return self.G

    def plot_graph(self):
        # Spécification des positions des nœuds
        pos = {}

        # Position des noeuds "parents"
        x, y = 0, 0

        # Positions des leurres
        x_sub = -0.333
        y_sub = -1

        # Id unique des noeuds
        node_id = 0

        # Positionnement de chaque noeud dans le graph
        for i in range(0, self.M):
            node_id += 1
            pos[str(node_id)] = (x, y)
            for j in range(1, self.n+1):
                pos[str(node_id) + "," + str(j)] = (x_sub, y_sub)
                x_sub += 1/self.n
            x += 1

        # Couleur initiale des nœuds
        #node_colors = ['green' for _ in self.G.nodes]
        node_colors = []

        for type in nx.get_node_attributes(self.G, 'type').values():
            if type == 'Start':
                node_colors.append('red')
            elif type == 'Service':
                node_colors.append('cyan')
            elif type == 'Goal':
                node_colors.append('green')
            else:
                node_colors.append('grey')

        # Modification de la couleur du noeud de départ en rouge
        #node_colors[list(self.G.nodes).index('1')] = 'red'

        nx.draw(self.G, pos=pos, with_labels=True, font_weight='bold', node_size=700, node_color=node_colors)

        plt.show()

import networkx as nx
import matplotlib.pyplot as plt


class Modelling:
    def __init__(self, M=5, K=16):
        self.G = nx.DiGraph()  # Le graphe dirigé
        self.M = M  # Le nombre de services à atteindre avant l'accès aux données sensibles
        self.n = int(K / (M-1))  # Le nombre de leurres pour chaque service
        self.pos = {}  # La position de chaque noeud dans le graphe
        self.node_colors = []  # La couleur de chaque noeud dans le graphe

    def create_graph(self):
        # Id unique du noeud
        node_id = 0
        # Probabilité de transition pour chaque arrête
        proba = 1/(self.n+1)

        # Pour chaque services
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
            if node_id < self.M: # Création de n Honeypots pour tous les services i sauf celui des données sensibles
                for j in range(1, self.n+1):
                    self.G.add_node(str(node_id) + "," + str(j), type="Honeypot")
                    self.G.add_edge(str(node_id), str(node_id) + "," + str(j), weight=proba)

        # Spécification des positions des nœuds
        self.init_pos()

        # Couleurs initiales des nœuds
        self.init_nodes_color()

        return self.G

    def init_nodes_color(self):
        for type in nx.get_node_attributes(self.G, 'type').values():
            if type == 'Start':
                self.node_colors.append('purple')  # Noeud de départ en violet
            elif type == 'Service':
                self.node_colors.append('cyan')  # Service en cyan
            elif type == 'Goal':
                self.node_colors.append('green')  # Données sensibles en vert
            else:
                self.node_colors.append('grey')  # Honeypots en gris

    def init_pos(self):
        # Position des services
        x, y = 0, 0
        # Positions des leurres
        x_sub = -0.333
        y_sub = -1
        # Id unique des noeuds
        node_id = 0
        # Positionnement de chaque noeud dans le graph
        for i in range(0, self.M):
            node_id += 1
            self.pos[str(node_id)] = (x, y)
            for j in range(1, self.n + 1):
                self.pos[str(node_id) + "," + str(j)] = (x_sub, y_sub)
                x_sub += 1 / self.n
            x += 1

    def plot_graph(self):
        nx.draw(self.G, pos=self.pos, with_labels=True, font_weight='bold', node_size=700, node_color=self.node_colors)

        plt.show()
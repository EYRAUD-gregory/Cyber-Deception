import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from modelling import Modelling

class Attacker:
    def __init__(self, M, K, know_M=False, is_uniform=False):
        self.model = Modelling(M, K)
        self.G = self.model.create_graph()
        if know_M:
            self.M = M
        else:
            self.M = None
        self.is_uniform = is_uniform
        self.position = '1'
        self.nb_movement = 0
        self.fig, self.ax = plt.subplots()
        self.ani = None

    def proba_return(self, X):
        if self.is_uniform:
            return 0.1  # probabilité uniforme de retour au point de départ
        if self.M is None:
            return 1 / (X + 1)
        return 1 / (self.M - X + 1)

    def move(self):
        if self.G.nodes[self.position]['type'] != 'Honeypot':
            services = list(self.G[self.position].keys())
            probabilities = [sous_dict['weight'] for sous_dict in self.G[self.position].values()]

            next_service = random.choices(services, probabilities)[0]
            return next_service

        return self.position # TODO peut-être devoir changer ça

    def go_back(self):
        return random.uniform(0, 1) < self.proba_return(int(self.position.split(',')[0]))

    def update_animation(self, frame):

        self.ax.clear()

        # Dessiner le graphe avec les arêtes en gris
        nx.draw(self.G, pos=self.model.pos, with_labels=True, node_color=self.model.node_colors, node_size=700, ax=self.ax)

        # Dessiner l'attaquant en rouge sur le nœud actuel
        nx.draw_networkx_nodes(self.G, pos=self.model.pos, nodelist=[self.position], node_color="red",
                                   node_size=500,
                                   ax=self.ax)

        # Vérifier si le nœud actuel est absorbant
        if self.G.nodes[self.position]['type'] == 'Goal':
            # Arrêter l'animation
            self.ani.event_source.stop()
            return

        # Faire avancer l'attaquant
        return_to_start = self.go_back()
        if return_to_start:
            self.position = '1'
        else:
            self.position = self.move()
            self.nb_movement += 1

        plt.tight_layout()

    def animate_attack(self, frames=10, interval=300):
        # Créer l'animation
        self.ani = FuncAnimation(self.fig, self.update_animation, frames=frames, interval=interval)

        # Afficher l'animation
        plt.show()

        return self.nb_movement

    def attack(self, with_graph=False):
        cpt = 1
        self.position = '1'

        while (self.M is not None and self.position != str(self.M)) or (
                self.M is None and self.G.nodes[self.position]['type'] != 'Goal'):
            return_to_start = self.go_back()
            if return_to_start:
                cpt += 1
                self.position = '1'
                continue
            self.position = self.move()
        return cpt

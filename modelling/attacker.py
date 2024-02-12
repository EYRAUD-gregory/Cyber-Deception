import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from modelling import Modelling
from math import exp

from IPython.display import HTML


class Attacker:
    def __init__(self, M, K, know_M=False, is_uniform=False):
        self.model = Modelling(M, K)
        self.G = self.model.create_graph()  # Le graphe
        if know_M: # Si l'attaquant connaît M
            self.M = M
        else:
            self.M = None
        self.is_uniform = is_uniform  # Probabilité de retour uniforme
        self.position = '1'  # Position de l'attaquant
        self.nb_movement = 0  # Nombre de déplacement de l'attaquant
        self.nb_movement_try = 0  # Nombre de mouvement pour une tentative
        self.fig, self.ax = plt.subplots()  # Utilisé pour afficher l'animation
        self.ani = None  # L'animation
        self.p = 1/(M-1)  # probabilité de retour fixe

    def proba_return(self):
        if self.M is None:
            if self.is_uniform:
                return self.p  # probabilité uniforme de retour au point de départ
            #X = -self.nb_movement_try
            X = -0.1 * self.nb_movement_try
            return 1 - exp(X)
            #return exp(X)
        #print("Length = ", self.nb_movement%self.M)
        #return (self.nb_movement % self.M) == 0

    def move(self):
        if self.G.nodes[self.position]['type'] != 'Honeypot':  # Si le noeud actuel n'est pas un Honeypot
            services = list(self.G[self.position].keys())  # Liste des noeuds où on peut aller depuis la position actuelle
            probabilities = [sous_dict['weight'] for sous_dict in self.G[self.position].values()]  # Probabilité de déplacement

            next_service = random.choices(services, probabilities)[0] # Choix aléatoire du prochain service visité
            return next_service

        return self.position

    def go_back(self):
        # Pour définir si l'attaquant doit revenir au point de départ ou non
        return random.uniform(0, 1) < self.proba_return()

    def update_animation(self, frame):
        self.ax.clear()

        # Dessiner le graphe
        nx.draw(self.G, pos=self.model.pos, with_labels=True, node_color=self.model.node_colors, node_size=700, ax=self.ax)

        # Dessiner l'attaquant en rouge sur le noeud actuel
        nx.draw_networkx_nodes(self.G, pos=self.model.pos, nodelist=[self.position], node_color="red", node_size=500, ax=self.ax)

        if self.ani is not None and self.ani.event_source is not None:  # Si l'animation est toujours en cours
            if self.G.nodes[self.position]['type'] == 'Goal':  # Si le noeud actuel est le noeud absorbant
                self.ani.event_source.stop()  # On arrête l'animation
                return

                # Faire avancer l'attaquant
        if self.position != '1':
            return_to_start = self.go_back()  # Est-ce qu'on retourne en arrière?
            if return_to_start:  # Si oui
                self.position = '1'  # Retour au point de départ
                self.nb_movement += 1
                self.nb_movement_try = 0
            else:
                    self.position = self.move()  # On avance
                    self.nb_movement += 1  # Incrément
                    self.nb_movement_try += 1  # Incrément
        else:
            self.position = self.move()  # On avance
            self.nb_movement += 1  # Incrément
            self.nb_movement_try += 1  # Incrément

        plt.tight_layout()  # Pour s'assurer que tout rentre dans l'affichage

    def animate_attack(self, frames=10, interval=300):
        # Créer l'animation
        self.ani = FuncAnimation(self.fig, self.update_animation, frames=frames, interval=interval)

        plt.show()  # Nettoyer la figure
        #plt.close()

        # Afficher l'animation dans le notebook
        #return HTML(self.ani.to_jshtml())

        # Afficher l'animation sous forme de vidéo HTML
        return HTML(self.ani.to_html5_video())

    def attack(self, with_graph=False):
        self.position = '1'  # Postion de départ

        self.nb_movement = 0  # Nombre total de mouvement réalisé par l'attaquant
        self.nb_movement_try = 0  # Nombre total de mouvement réalisé par l'attaquant

        # Tant qu'on est pas arrivé à la destination
        while ((self.M is not None and self.position != str(self.M)) or (
                self.M is None and self.G.nodes[self.position]['type'] != 'Goal')):
            #self.nb_movement += 1  # On le compte comme un mouvement

            if self.position != '1':
                return_to_start = self.go_back()  # Est-ce qu'on retourne en arrière?
                if return_to_start:  # Si oui
                    self.position = '1'  # Retour au point de départ
                    self.nb_movement += 1
                    self.nb_movement_try = 0
                    continue
            self.position = self.move()  # On avance
            self.nb_movement += 1  # Incrément
            self.nb_movement_try += 1  # Incrément
        return self.nb_movement

import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Attacker:
    def __init__(self, G, M=None, is_uniform=False):
        self.G = G
        self.M = M
        self.is_uniform = is_uniform
        self.position = '1'

    def proba_return(self, X):
        if self.is_uniform:
            return 0.1  # probabilité uniforme de retour au point de départ
        if self.M is None:
            return 1 / (X + 1)
        return 1 / (self.M - X+1)

    def move(self):
        if ',' not in self.position:
            services = list(self.G[self.position].keys())
            probabilities = [sous_dict['weight'] for sous_dict in self.G[self.position].values()]
            # print(probabilities)
            next_service = random.choices(services, probabilities)[0]
            return next_service
        return self.position  # TODO peut-être devoir changer ça

    def go_back(self):
        return True if random.uniform(0, 1) < self.proba_return(int(self.position.split(',')[0])) else False

    def attack(self, with_graph=False):
        cpt = 1
        self.position = '1'
        # Créer la figure et l'axe
        fig, ax = plt.subplots()

        while (self.M is not None and self.position != str(self.M)) or (self.M is None and self.G.nodes[self.position]['type'] != 'Goal'):
            return_to_start = self.go_back()
            if return_to_start:
                cpt += 1
                self.position = '1'
                continue
            self.position = self.move()
        return cpt



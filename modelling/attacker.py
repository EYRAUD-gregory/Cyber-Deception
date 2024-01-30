import random

class Attacker:
    def __init__(self, G, M=None):
        self.G = G
        self.M = M
        self.position = '1'

    def proba_return(self, X):
        if self.M is None:
            return 1 / (X + 1)
        else:
            return 1 / (self.M - X)

    def move(self):
        print(self.G[self.position])
        return str(int(self.position) + 1)

    def go_back(self):
        return True if random.uniform(0, 1) < self.proba_return(int(self.position)) else False

    def attack(self):
        while self.position != self.M and not self.go_back():
            self.position = self.move()



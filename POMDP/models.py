import random
import pomdp_py
from scipy.stats import truncnorm
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

from domain import *

M = 2
L = 4


class ObservationModel(pomdp_py.ObservationModel):
    def probability(self, observation, next_state, action): #Pr(o|s',a)
        if action == "go_back":
            return 0.0
        if observation.T == 0:
            if next_state.chi == 0 and next_state.T == 0:
                return a_priori_distribution(next_state.k)
        return 1.0

    def sample(self, next_state, action):
        return Observation(next_state.k, next_state.T)

    def get_all_observations(self):
        all_observations = []
        for i in range (0, M-2):
            all_observations.append(Observation(i, 0))
        all_observations.append(Observation(M-1, 0))
        all_observations.append(Observation(M-1, 1))
        return all_observations


class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        if action.name == "go_back":
            if state.k == 0:
                return 1.0
            if state.T == 1:
                return 1.0
            return a_priori_distribution(state.k)

        else:
            if state.T == 1:
                return 1.0
            if next_state.chi == 0 and state.chi == 0:
                return 1/L
            elif next_state.chi == 1 and state.chi == 0:
                return 1 - 1/L
            elif next_state.chi == 1 and state.chi == 1:
                return 1
            elif next_state.chi == 0 and state.chi == 1:
                return 0

    def sample(self, state, action):
        if action.name == "go_back":
            return State(0,0,0)
        else:
            if state.chi == 0:
                if state.k == M-2:
                    #print(random.choices([State(state.k+1, 0, 1), State(state.k+1, 1, 0)], weights=[1 / L, 1 - (1 / L)])[0])
                    return random.choices([State(state.k+1, 0, 1), State(state.k+1, 1, 0)], weights=[1 / L, 1 - (1 / L)])[0]
                else:
                    return random.choices([State(state.k+1, 0, 0), State(state.k+1, 1, 0)], weights=[1 / L, 1 - (1 / L)])[0]
            else:
                return State(state.k+1, 1, 0)

    def get_all_states(self):
        all_states = [State(0, 0, 0)]
        for i in range (1, M-1):
            all_states.append(State(i, 0, 0))
            all_states.append(State(i, 1, 0))
        all_states.append(State(M-1, 0, 1))
        all_states.append(State(M-1, 1, 0))
        return all_states


# pomdp_py doc: The job of a PolicyModel is to:
# (1) determine the set of actions that the robot can take at given state (and/or history);
# (2) sample an action from this set according to some probability distribution
class PolicyModel(pomdp_py.RolloutPolicy):
    def probability(self, action, state):
        if state.k == 0:
            if action.name == "go_back":
                return 0
            return 1

        if action.name == "go_forward":
            return is_backing_off(state.k)
        return 1 - is_backing_off(state.k)

    def sample(self, state):
        if state.k == 0:
            return Action("go_forward")
        return random.choices([Action("go_back"), Action("go_forward")], weights=[is_backing_off(state.k), 1 - is_backing_off(state.k)])[0]

    def rollout(self, state, tuple_history=None):
        return self.sample(state)

    def get_all_actions(self, *args):
        return {Action(s) for s in {"go_back", "go_forward"}}


class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action, next_state):
        if action == "go_back":
            if state.T == 1:
                return 1.0
            return -1.0
        if next_state.T == 1:
            return 1.0
        return -1.0

    def sample(self, state, action, next_state):
        return self._reward_func(state, action, next_state)


def a_priori_distribution(k):
    M_conject = 4
    mean = M_conject/2
    std = 1

    # Définir les paramètres de la distribution tronquée
    a = (0 - mean) / std  # limite inférieure standardisée
    b = (M_conject - mean) / std  # limite supérieure standardisée

    # Calcul de la probabilité que M dépasse k
    probability_M_gt_k = 1 - truncnorm.cdf(k, a, b, loc=mean, scale=std)

    #print(f"For k : {k}, Pr = {probability_M_gt_k}")
    """
    # Générer des données à partir de la distribution tronquée
    data = truncnorm.rvs(a, b, loc=mean, scale=std, size=10000)

    # Tracer l'histogramme des données générées
    plt.hist(data, bins=50, density=True, alpha=0.6, color='g')

    # Tracer la densité de probabilité de la distribution tronquée
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 10)
    p = truncnorm.pdf(x, a, b, loc=mean, scale=std)
    plt.plot(x, p, 'k', linewidth=2)

    plt.title("Distribution tronquée")
    plt.xlabel("Valeurs")
    plt.ylabel("Densité de probabilité")
    plt.show()
    """
    return probability_M_gt_k

    #print(f"La probabilité que M dépasse {k} est : {probability_M_gt_k}")


def is_backing_off(k):
    if k is None:
        return 1/M
    return 0.8

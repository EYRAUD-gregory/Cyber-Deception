import random

import pomdp_py

from domain import *
from scipy.stats import truncnorm

M = 7
L = 3

class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self):
        super().__init__()

    def probability(self, observation, next_state, action): #Pr(o|s',a)
        if observation.Y > 0 and observation.T == 0:
            if next_state.chi == 0 and next_state.T == 0:
                return a_priori_distribution(next_state.k)
            else:
                return 1
        return 0

    def sample(self, next_state, action):
        return Observation(next_state.k, next_state.T)

    def get_all_observations(self):
        return ObservationModel.get_all_observations(self)


class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        if action.name == "go_back":
            return 1
        else:
            if next_state.chi == 0 and state.chi == 0:
                return 1/L
            elif next_state.chi == 1 and state.chi == 0:
                return 1 - 1/L
            elif next_state.chi == 1 and state.chi == 1:
                return 1

    def sample(self, state, action):
        if action.name == "go_back":
            return State(0,0,0)
        else:
            if state.chi == 0:
                if state.k == M-2:
                    return random.choices([State(state.k+1, 0, 1), State(state.k+1, 1, 0)], weights=[1 / L, 1 - 1 / L])
                else:
                    return random.choices([State(state.k+1, 0, 0), State(state.k+1, 1, 0)], weights=[1 / L, 1 - 1 / L])
            else:
                return State(state.k+1, 1, 0)

    def get_all_states(self):
        return TransitionModel.get_all_states(self)


# pomdp_py doc: The job of a PolicyModel is to:
# (1) determine the set of actions that the robot can take at given state (and/or history);
# (2) sample an action from this set according to some probability distribution
class PolicyModel(pomdp_py.RolloutPolicy):
    def probability(self, action, state):
        if state.k == 0:
            if action.name == "go_back":
                return 0
            return 1

        if action.name == "go_back":
            return is_backing_off(None)
        return 1 - is_backing_off(None)

    def sample(self, state):
        if state.k == 0:
            return Action("go_forward")
        return random.choices([Action("go_back"), Action("go_forward")], weights=[is_backing_off(None), 1 - is_backing_off(None)])

    def rollout(self, state, tuple_history=None):
        return self.sample(state)

    def get_all_actions(self, *args):
        return {Action(s) for s in {"go_back", "go_forward"}}


class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        if action.name == "go_back":
            return -1
        else:
            if state.k == M-1:
                return 1
            else:
                return -1

    def sample(self, state, action, next_state):
        return self._reward_func(state, action)


def a_priori_distribution(k):
    mean = 50
    std = 20

    # Définir les paramètres de la distribution tronquée
    a = (0 - mean) / std  # limite inférieure standardisée
    b = (100 - mean) / std  # limite supérieure standardisée

    # Calcul de la probabilité que M dépasse k
    probability_M_gt_k = 1 - truncnorm.cdf(k, a, b, loc=mean, scale=std)

    #print(f"La probabilité que M dépasse {k} est : {probability_M_gt_k}")


def is_backing_off(k):
    if k is None:
        return 1/M
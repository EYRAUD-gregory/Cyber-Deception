import pomdp_py

from models import *


class AttackerProblem(pomdp_py.POMDP):

    def __init__(self, init_state, init_belief):
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(),
                               ObservationModel(),
                               RewardModel())

        env = pomdp_py.Environment(init_state,
                                   TransitionModel(),
                                   RewardModel())

        super().__init__(agent, env, name="AttackerProblem")

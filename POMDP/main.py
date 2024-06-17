import pomdp_py
import matplotlib.pyplot as plt
import numpy as np
from domain import *
from models import a_priori_distribution
from POMDP import AttackerProblem

init_state = State(0,0,0)
init_belief = pomdp_py.Histogram({State(0,0,0): 1, State(0,1,0): 0})

attacker_problem = AttackerProblem(init_state, init_belief)

# Step 1; in main()
# creating planners
vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
pouct = pomdp_py.POUCT(max_depth=3, discount_factor=0.95,
                       planning_time=.5, exploration_const=110,
                       rollout_policy=attacker_problem.agent.policy_model)
pomcp = pomdp_py.POMCP(max_depth=3, discount_factor=0.95,
                       planning_time=.5, exploration_const=110,
                       rollout_policy=attacker_problem.agent.policy_model)
 # call test_planner() for steps 2-6.

# Steps 2-6; called in main()
def test_planner(attacker_problem, planner, nsteps=3):
    for i in range(nsteps):  # Step 6
        # Step 2
        print(attacker_problem.agent)
        action = planner.plan(attacker_problem.agent)

        print("==== Step %d ====" % (i+1))
        print("True state:", attacker_problem.env.state)
        print("Belief:", attacker_problem.agent.cur_belief)
        beliefs.append(attacker_problem.agent.cur_belief)
        print("Action:", action)
        # Step 3; There is no state transition for the tiger domain.
        # In general, the ennvironment state can be transitioned
        # using
        #
        #print(attacker_problem.env.state_transition(action))
        reward = attacker_problem.env.state_transition(action, execute=True)
        #
        # Or, it is possible that you don't have control
        # over the environment change (e.g. robot acting
        # in real world); In that case, you could skip
        # the state transition and re-estimate the state
        # (e.g. through the perception stack on the robot).
        #reward = attacker_problem.env.reward_model.sample(attacker_problem.env.state, action, None)
        print("Reward:", reward)

        # Step 4
        # Let's create some simulated real observation;
        # Here, we use observation based on true state for sanity
        # checking solver behavior. In general, this observation
        # should be sampled from agent's observation model, as
        #
        #    real_observation = tiger_problem.agent.observation_model.sample(tiger_problem.env.state, action)
        #
        # or coming from an external source (e.g. robot sensor
        # reading). Note that tiger_problem.env.state should store
        # the environment state after transition.
        real_observation = Observation(attacker_problem.env.state.k, attacker_problem.env.state.T)
        print(">> Observation: %s" % real_observation)

        # Step 5
        # Update the belief. If the planner is POMCP, planner.update
        # also automatically updates agent belief.
        attacker_problem.agent.update_history(action, real_observation)
        planner.update(attacker_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims: %d" % planner.last_num_sims)
        if isinstance(attacker_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(attacker_problem.agent.cur_belief,
                                                          action, real_observation,
                                                          attacker_problem.agent.observation_model,
                                                          attacker_problem.agent.transition_model,
                                                          normalize=True)

            attacker_problem.agent.set_belief(new_belief)


beliefs = []

belief1 = []
belief2 = []


test_planner(attacker_problem, vi, nsteps=5)

i = 0
for belief in beliefs:
    # Récupérer le dictionnaire d'éléments de la croyance
    histogram_items = belief.get_histogram().items()
    # Itérer sur les éléments du dictionnaire
    for state, value in histogram_items:
        if i == 0:
            belief1.append(value)
            i = 1
        else:
            belief2.append(value)
            i = 0
        #print(i, ":", value)  # Imprimer la valeur de chaque état dans la croyance

print(belief1)
print(belief2)


# Création du graphique
plt.figure(figsize=(8, 6))  # Définition de la taille du graphique

# Tracer les données des deux tableaux
plt.plot(belief1, label="Belief to be in the right path")
plt.plot(belief2, label="Belief to be in a honeypot")

# Ajout des titres et des étiquettes d'axe
plt.title("Beliefs' progression for L = 4")
plt.xlabel('Iteration')
plt.ylabel('Probability')

plt.xticks(range(len(belief1)))

# Ajout de la légende
plt.legend()

# Affichage du graphique
plt.show()


#a_priori_distribution(3)
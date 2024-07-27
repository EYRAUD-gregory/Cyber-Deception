import random

import numpy as np
import math

import numpy.random
from scipy.stats import truncnorm, poisson
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt

L = 2
gamma = 0.995

# Classe représentant un état (i.e un microservice)
class State:
    def __init__(self, Y, chi, T):
        self.Y = Y  # L'indice du microservice
        self.chi = chi  # 1 si honeypot 0 sinon
        self.T = T  # 1 si arrivé dans la cible 0 sinon


# Fonction pour simuler le déplacement d'un attaquant
def go_forward(s, M):
    if s.chi == 0:  # Si dans le bon chemin
        is_going_in_honeypot = random.choices([True, False], weights=[(1 - 1/L), 1/L])[0]  # Vers un honeypot avec proba 1 - 1/L, et 1/l pour continuer sur le bon chemin
        if is_going_in_honeypot:  # Si on se dirige vers un honeypot
            return State(s.Y +1, 1, 0)  # On incrémente Y et chi passe à 1
        # Sinon on continue vers la cible
        if s.Y == M - 1:  # Si le microservice actuel est juste avant la cible
            return State(s.Y +1, 0,1) #  On arrive à la cible
        return State(s.Y +1, 0, 0)  # Sinon on incrémente Y et on reste sur le bon chemin
    return State(s.Y +1, 1, 0) # Si on était déjà dans un honeypot on incrément juste Y


# Fonction pour retourner au point de départ
def go_back():
    return State(0,0,0)


# Fonction pour générer un M suivant une loi de Poisson positive
def sample_truncated_poisson(lam):
    sample = np.random.poisson(lam=lam)
    while sample < 1:
        sample = np.random.poisson(lam=lam)
    return sample


# Fonction pour simuler les déplacements d'un attaquant utilisant un a priori sur M
def simulate(episodes, sigma, lambda_, W, distrib="Poisson"):
    print("Simulation : sigma = ", sigma)
    rewards = np.zeros(episodes)
    for i in range(0, episodes):
        reward = 0
        nb_step = 0  # Le nombre de pas pendant le cycle
        nb_step_total = 0  # Le nombre de pas total
        if distrib == "Poisson":
            M = sample_truncated_poisson(lambda_)  # Pour distribution Poisson
        else:
            M = random.choices([1, 2, 3], weights=[1 / 3, 1 / 3, 1 / 3])[0]  # Pour distribution, uniforme
        if M > sigma:  # Si M est plus grand que sigma, on n'atteindra jamais la cible
            rewards[i] = -1 / (1 - gamma)  # La reward devient automatiquement -1 / (1 - gamma)
            # print("Reward = ", rewards[i])
            continue
        #print("M = ", M)
        state = State(0, 0, 0)
        while state.T != 1:  # Tant que la cible n'est pas atteinte
            if nb_step < sigma:  # On avance si on a fait moins de sigma déplacement
                state = go_forward(state, M)
                nb_step += 1
                nb_step_total += 1
            else:  # Sinon on revient au point de départ
                state = go_back()
                nb_step = 0
                nb_step_total += 1

        # print("nb_step_total = ", nb_step_total)
        for j in range(0, nb_step_total - 1):  # Calcul de la reward
            reward += -gamma ** j
        reward += W * gamma ** (nb_step_total - 1)

        rewards[i] = reward
        # print("Reward = ", rewards[i])
    return rewards


# Fonction pour simuler episodes fois et faire le moyenne des rewards
def simu_for_threshold(episodes, sigmas, lambda_, W, distrib="Poisson"):
    all_rewards = np.zeros(len(sigmas))
    for sigma in sigmas:
        rewards = simulate(episodes, sigma, lambda_, W, distrib)  # Appele de la fonction pour simuler episodes fois une attaque
        all_rewards[sigma - 1] = np.mean(rewards)  # On prend la moyenne
    return all_rewards


# Fonction pour récupérer le sigma optimal pour un lambda donné
def optimal_sigma_for_threshold(episodes, sigmas, lambda_, W, distrib="Poisson"):
    all_rewards = {}
    for sigma in sigmas:
        rewards = simulate(episodes, sigma, lambda_, W, distrib)
        all_rewards[sigma] = np.mean(rewards)
    print(all_rewards)
    return max(all_rewards, key=all_rewards.get)


# Fonction pour créer le graphique de l'évolution de la reward de la simulation et de la formule
def plot_rewards(all_rewards, lambda_):
    id = 0
    for reward in all_rewards:
        if id == 0:
            label = "Simulation"
        else:
            label = "Formule"
        plt.plot(sigmas, reward, marker='o', label=label)
        id +=1
    plt.title(f"Evolution des rewards selon sigma (lambda = {lambda_})")
    plt.xlabel("Sigma")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend()
    plt.show()


# Fonction pour calculer la reward moyenne suivant la formule de l'article
def mean_reward(sigma, lambda_, W, distrib='Poisson'):
    somme = 0
    proba = [0, 1/3, 1/3, 1/3, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Distribution uniforme de support 3
    m = 1

    while m <= sigma:
        if distrib == 'Poisson':
            beta_m = (np.exp(-lambda_) * (lambda_ ** m) / math.factorial(m))
        else:
            beta_m = proba[m]
        somme += ((L**(-m) * gamma**m) / (1 - (1 - L**(-m))* gamma**(sigma+1))) * beta_m
        m += 1

    print("sigma = ", sigma, "somme = ", somme)
    reward = -(1 / (1 - gamma)) + (W + (1 / (1 - gamma))) * somme

    #print("Reward formule = ", reward)

    return reward


# Fonction permettant d'appeler les fonctions pour simuler et calculer la reward moyenne avec la formule pour ensuite les comparer
def compare_simulation_formula(episodes, sigmas, lambda_, W):
    all_rewards_simu = simu_for_threshold(episodes, sigmas, lambda_, W)
    all_rewards_formula = np.zeros(len(sigmas))
    for sigma in sigmas:
        all_rewards_formula[sigma - 1] = mean_reward(sigma, lambda_, W)
    all_rewards = [all_rewards_simu, all_rewards_formula]
    # for i in range(0, len(all_rewards_simu)):
    #   print("ratio = ", all_rewards_simu[i]/all_rewards_formula[i])
    plot_rewards(all_rewards, lambda_)


#  Fonction pour calculer et afficher les sigmas optimaux pour chaque lambda
def find_optimal_sigmas(episodes, sigmas, all_lambdas, W):
    all_optimal_sigma = []
    for lambda_ in all_lambdas:
        print('lambda = ', lambda_)
        all_optimal_sigma.append(optimal_sigma_for_threshold(episodes, sigmas, lambda_, W))

    print(all_optimal_sigma)

    plt.plot(all_lambdas, all_optimal_sigma, marker='o')
    plt.title(f"Evolution du sigma optimal selon lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Sigma")
    plt.grid()
    plt.show()


# Fonction pour calculer P(M>k) suivant une loi de Poisson
def probability_M_greater_k(M, k, lambda_, distrib="Poisson"):
    proba = 0
    if distrib == "Poisson":
        for i in range(0, k):
            proba += (np.exp(-lambda_) * (lambda_ ** i) / math.factorial(i))

    return 1 - proba


# Fonction pour simuler une politique donnée se servant d'une distribution a priori donné
def simulate_policy(episodes, all_lambdas, W, distrib, policy):
    all_rewards = np.zeros(len(all_lambdas))
    probabilities_to_return = [0.38, 0.3, 0.2, 0.2, 0.17, 0.17, 0.15, 0.13, 0.12, 0.1]  # Calculer avec les fonctions calculate_optimal_p et find_optimal_p (10 000 episodes)
    for lambda_ in all_lambdas:
        print("Simulation : lambda = ", lambda_)
        rewards = np.zeros(episodes)
        for i in range(0, episodes):
            # print("episode : ", i)
            reward = 0
            nb_step = 0  # Nombre de pas pendant ce cycle
            nb_step_total = 0  # Nombre de pas total
            if distrib == "Poisson":
                M = sample_truncated_poisson(lambda_)  # Pour distribution Poisson
            # print("M = ", M)
            state = State(0, 0, 0)
            is_going_back = False
            while state.T != 1 and nb_step_total < 1e5:  # à voir par rapport à la limitation
                if state.Y > 0:
                    if policy == "mixte":
                        # print("proba de retour = ", 1 - probability_M_greater_k(M, nb_step, lambda_))
                        is_going_back = random.choices([True, False],
                                                       weights=[1 - probability_M_greater_k(M, nb_step, lambda_),
                                                                probability_M_greater_k(M, nb_step, lambda_)])[0]
                    elif policy == "uniforme":
                        p= probabilities_to_return[lambda_]
                        is_going_back = random.choices([True, False],
                                                       weights=[p,
                                                               1 - p])[0]
                if is_going_back and nb_step > 0:
                    state = go_back()
                    nb_step = 0
                    nb_step_total += 1
                    is_going_back = False
                else:
                    state = go_forward(state, M)
                    nb_step += 1
                    nb_step_total += 1

            # print("nb_step_total = ", nb_step_total)
            for j in range(0, nb_step_total - 1):
                reward += -gamma ** j
            if state.T == 1:
                reward += W * gamma ** (nb_step_total - 1)
            else:
                reward += -gamma ** (nb_step_total - 1)

            rewards[i] = reward
            # print("pas total = ", nb_step_total)
            # print("Reward = ", rewards[i])
        all_rewards[lambda_ - 1] = np.mean(rewards)

    return all_rewards


# Fonction pour calculer l'ensemble des rewards pour chaques probabilités de retour pour chaque lambda
def calculate_optimal_p(episodes, all_lambdas, W, distrib, policy):
    #all_rewards = np.zeros(len(all_lambdas))
    rewards_for_p = {}
    for lambda_ in all_lambdas:
        all_p = np.arange(0.01, 1 / lambda_ + 0.01, 0.01)
        print("Simulation : lambda = ", lambda_)
        rewards = np.zeros(episodes)
        for p in all_p:
            print(p)
            for i in range(0, episodes):
                # print("episode : ", i)
                reward = 0
                nb_step = 0  # Nombre de pas pendant le cycle
                nb_step_total = 0  # Nombre de pas total
                if distrib == "Poisson":
                    M = sample_truncated_poisson(lambda_)  # Pour distribution Poisson
                # print("M = ", M)
                state = State(0, 0, 0)
                is_going_back = False
                while state.T != 1 and nb_step_total < 1e5:  # à voir par rapport à la limitation
                    if state.Y > 0:
                        if policy == "mixte":
                            # print("proba de retour = ", 1 - probability_M_greater_k(M, nb_step, lambda_))
                            is_going_back = random.choices([True, False],
                                                           weights=[1 - probability_M_greater_k(M, nb_step, lambda_),
                                                                    probability_M_greater_k(M, nb_step, lambda_)])[0]
                        elif policy == "uniforme":
                            #probability_to_return = 1 / lambda_
                            is_going_back = random.choices([True, False],
                                                           weights=[p,
                                                                   1 - p])[0]
                    if is_going_back and nb_step > 0:
                        state = go_back()
                        nb_step = 0
                        nb_step_total += 1
                        is_going_back = False
                    else:
                        state = go_forward(state, M)
                        nb_step += 1
                        nb_step_total += 1


                # print("nb_step_total = ", nb_step_total)
                for j in range(0, nb_step_total - 1):
                    reward += -gamma ** j
                if state.T == 1:
                    reward += W * gamma ** (nb_step_total - 1)
                else:
                    reward += -gamma ** (nb_step_total - 1)

                rewards[i] = reward
                # print("pas total = ", nb_step_total)
                # print("Reward = ", rewards[i])
            #all_rewards[lambda_ - 1] = np.mean(rewards)
            rewards_for_p[(lambda_, p )] = np.mean(rewards)
    return rewards_for_p


#  Fonction pour trouver les probabilités de retour optimales
def find_optimal_p(data):
    # Dictionnaire pour stocker les résultats
    max_rewards = {}

    # Parcourir le dictionnaire
    for (lambda_, p), reward in data.items():
        if lambda_ not in max_rewards:
            max_rewards[lambda_] = (p, reward)
        else:
            current_max_p, current_max_reward = max_rewards[lambda_]
            if reward > current_max_reward:
                max_rewards[lambda_] = (p, reward)

    # Afficher les résultats
    for lambda_, (p, reward) in max_rewards.items():
        print(f"Pour lambda = {lambda_}, la valeur maximale de reward est {reward} avec p = {p}")

# Fonction pour générer un graphique avec plusieurs politiques
def compare_policies(policies, rewards):
    for i in range(len(policies)):
        plt.plot(all_lambdas, rewards[i], marker='o', label=policies[i])
    plt.title(f"Evolution de la reward optimal selon lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':

    episodes = 10000
    sigmas = [1, 2, 3, 4, 5, 6, 7]
    #lambda_ = 8
    all_lambdas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sigmas_optimal = [4, 5, 5, 6, 7, 7, 7, 7, 7, 8]
    W = 1e3

    i = 0
    rewards_threshold = np.zeros(len(all_lambdas))
    for lambda_ in all_lambdas:
        print("Simulation: lambda=", lambda_)
        rewards_threshold[i] = np.mean(simulate(episodes, sigmas_optimal[i], lambda_, W, distrib="Poisson"))
        i += 1

    #compare_simulation_formula(episodes, sigmas, lambda_, W)

    #find_optimal_sigmas(episodes, sigmas, all_lambdas, W)

    distrib = "Poisson"
    policies = ["mixte", 'uniforme']
    all_rewards = []
    for policy in policies:
        all_rewards.append(simulate_policy(episodes, all_lambdas, W, distrib, policy))

    policies.append("seuil")
    all_rewards.append(rewards_threshold)

    compare_policies(policies=policies, rewards=all_rewards)


    """
    
    
    reward_for_p = calculate_optimal_p(episodes, all_lambdas, W,"Poisson", 'uniforme')

    print(reward_for_p)

    print(find_optimal_p(reward_for_p))

    
    Pour lambda = 1, la valeur maximale de reward est 909.0988826002067 avec p = 0.38
    Pour lambda = 2, la valeur maximale de reward est 778.9427510508276 avec p = 0.3
    Pour lambda = 3, la valeur maximale de reward est 613.8007214506345 avec p = 0.2
    Pour lambda = 4, la valeur maximale de reward est 434.1678062737107 avec p = 0.2
    Pour lambda = 5, la valeur maximale de reward est 269.33658064395246 avec p = 0.17
    Pour lambda = 6, la valeur maximale de reward est 139.2502551808424 avec p = 0.17
    Pour lambda = 7, la valeur maximale de reward est 32.58830612730737 avec p = 0.15000000000000002
    Pour lambda = 8, la valeur maximale de reward est -48.42620413972533 avec p = 0.13
    Pour lambda = 9, la valeur maximale de reward est -104.43532486130091 avec p = 0.12
    Pour lambda = 10, la valeur maximale de reward est -140.8995224765586 avec p = 0.09999999999999999
    """


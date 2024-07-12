import random

import numpy as np
import math

import numpy.random
from scipy.stats import truncnorm, poisson
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt

L = 2
k_max = 3
gamma = 0.995


class State:
    def __init__(self, Y, chi, T):
        self.Y = Y
        self.chi = chi
        self.T = T


def go_forward(s, M):
    if s.chi == 0:
        is_going_in_honeypot = random.choices([True, False], weights=[(1 - 1/L), 1/L])[0]
        if is_going_in_honeypot:
            return State(s.Y +1, 1, 0)
        #if s.Y == M-2:
        if s.Y == M - 1:
            return State(s.Y +1, 0,1)
        return State(s.Y +1, 0, 0)
    return State(s.Y +1, 1, 0)


def go_back():
    return State(0,0,0)


def poisson_truncated_pmf(x, lamb):
    if x == 0:
        return 0
    return (lamb ** x) / ((math.exp(lamb) - 1) * math.factorial(x))


def plot_graph(all_evolution_reward, type_distrib='Poisson'):
    plt.figure(figsize=(12, 6))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'brown', 'olive', 'cyan', 'gray', 'lime']

    if type_distrib == 'Poisson':
        param = 'lambda'
    else:
        param = 'std'
    for idx, evolution_reward in enumerate(all_evolution_reward):
        plt.plot(range(1, 15), evolution_reward, label=param + f'= {idx + 2}', marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)])
    plt.xlabel('Sigma')
    plt.ylabel('Reward')
    plt.title('Evolution de la reward pour différentes valeurs du paramètre ' + param + ', ' + type_distrib)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_sub_graphics(all_evolution_reward):
    global evolution_reward
    # Affichage des sous-graphiques
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
    for idx, (ax, evolution_reward) in enumerate(zip(axes.flatten(), all_evolution_reward)):
        ax.plot(range(1, 10), evolution_reward, label=f'lambda = {idx + 2}', marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)])
        ax.set_title(f'lambda = {idx + 2}')
        ax.set_xlabel('Sigma')
        ax.grid(True)
    axes[0, 0].set_ylabel('Reward')
    axes[1, 0].set_ylabel('Reward')
    fig.suptitle('Evolution de la reward pour différentes valeurs de lambda')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def sample_truncated_poisson(lam):
    sample = np.random.poisson(lam=lam)
    while sample < 1:
        sample = np.random.poisson(lam=lam)
    return sample


def simulate(episodes, sigma, lambda_, W, distrib="Poisson"):
    print("Simulation : sigma = ", sigma)
    rewards = np.zeros(episodes)
    for i in range(0, episodes):
        reward = 0
        nb_step = 0
        nb_step_total = 0
        if distrib == "Poisson":
            M = sample_truncated_poisson(lambda_)  # Pour distribution Poisson
        else:
            M = random.choices([1, 2, 3], weights=[1 / 3, 1 / 3, 1 / 3])[0]  # Pour distribution, uniforme
        if M > sigma:
            rewards[i] = -1 / (1 - gamma)
            # print("Reward = ", rewards[i])
            continue
        print("M = ", M)
        state = State(0, 0, 0)
        while state.T != 1:
            if nb_step < sigma:
                state = go_forward(state, M)
                nb_step += 1
                nb_step_total += 1
            else:
                state = go_back()
                nb_step = 0
                nb_step_total += 1

        # print("nb_step_total = ", nb_step_total)
        for j in range(0, nb_step_total - 1):
            reward += -gamma ** j
        reward += W * gamma ** (nb_step_total - 1)

        rewards[i] = reward
        # print("Reward = ", rewards[i])
    return rewards


def simu_for_threshold(episodes, sigmas, lambda_, W, distrib="Poisson"):
    all_rewards = np.zeros(len(sigmas))
    for sigma in sigmas:
        rewards = simulate(episodes, sigma, lambda_, W, distrib)
        all_rewards[sigma - 1] = np.mean(rewards)
    return all_rewards


def optimal_sigma_for_threshold(episodes, sigmas, lambda_, W, distrib="Poisson"):
    all_rewards = {}
    for sigma in sigmas:
        rewards = simulate(episodes, sigma, lambda_, W, distrib)
        all_rewards[sigma] = np.mean(rewards)
    print(all_rewards)
    return max(all_rewards, key=all_rewards.get)


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
    #plt.title(f"Evolution des rewards selon sigma (distribution uniforme)")
    plt.xlabel("Sigma")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend()
    plt.show()


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


def compare_simulation_formula(episodes, sigmas, lambda_, W):
    all_rewards_simu = simu_for_threshold(episodes, sigmas, lambda_, W)
    all_rewards_formula = np.zeros(len(sigmas))
    for sigma in sigmas:
        all_rewards_formula[sigma - 1] = mean_reward(sigma, lambda_, W)
    all_rewards = [all_rewards_simu, all_rewards_formula]
    # for i in range(0, len(all_rewards_simu)):
    #   print("ratio = ", all_rewards_simu[i]/all_rewards_formula[i])
    plot_rewards(all_rewards, lambda_)


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


def probability_M_greater_k(M, k, lambda_, distrib="Poisson"):
    proba = 0
    if distrib == "Poisson":
        for i in range(0, k):
            proba += (np.exp(-lambda_) * (lambda_ ** i) / math.factorial(i))

    return 1 - proba


if __name__ == '__main__':

    episodes = 10000
    sigmas = [1, 2, 3, 4, 5, 6, 7]
    #lambda_ = 8
    all_lambdas = [1, 2, 3, 4, 5, 6]
    W = 1e3

    #compare_simulation_formula(episodes, sigmas, lambda_, W)

    #find_optimal_sigmas(episodes, sigmas, all_lambdas, W)

    distrib = "Poisson"
    all_rewards = np.zeros(len(all_lambdas))

    for lambda_ in all_lambdas:
        print("Simulation : lambda = ", lambda_)
        rewards = np.zeros(episodes)
        for i in range(0, episodes):
            #print("episode : ", i)
            reward = 0
            nb_step = 0
            nb_step_total = 0
            if distrib == "Poisson":
                M = sample_truncated_poisson(lambda_)  # Pour distribution Poisson
            else:
                M = random.choices([1, 2, 3], weights=[1 / 3, 1 / 3, 1 / 3])[0]  # Pour distribution, uniforme
            #print("M = ", M)
            state = State(0, 0, 0)
            is_going_back = False
            while state.T != 1 and nb_step_total < 1e5:
                if state.Y > 0:
                    if distrib == "Poisson":
                        #print("proba de retour = ", 1 - probability_M_greater_k(M, nb_step, lambda_))
                        is_going_back = random.choices([True, False],
                                                       weights=[1 - probability_M_greater_k(M, nb_step, lambda_),
                                                                probability_M_greater_k(M, nb_step, lambda_)])[0]
                if not is_going_back:
                    state = go_forward(state, M)
                    nb_step += 1
                    nb_step_total += 1
                else:
                    state = go_back()
                    nb_step = 0
                    nb_step_total += 1
                    is_going_back = False

            # print("nb_step_total = ", nb_step_total)
            for j in range(0, nb_step_total - 1):
                reward += -gamma ** j
            if state.T == 1:
                reward += W * gamma ** (nb_step_total - 1)
            else:
                reward += -gamma ** (nb_step_total - 1)

            rewards[i] = reward
            # print("Reward = ", rewards[i])
        all_rewards[lambda_ - 1] = np.mean(rewards)

    plt.plot(all_lambdas, all_rewards, marker='o')
    plt.title(f"Evolution de la reward optimal selon lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()
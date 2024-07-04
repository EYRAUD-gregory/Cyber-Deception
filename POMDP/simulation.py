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


def belief(s):

    if s.chi == 0 and s.T == 0:
        return N1(s.Y) / (N1(s.Y) + N2(s.Y))
    elif s.chi == 1 and s.T == 0:
        return N2(s.Y) / (N1(s.Y) + N2(s.Y))

    #return [(N1(s.Y) / (N1(s.Y) + N2(s.Y))), (N2(s.Y) / (N1(s.Y) + N2(s.Y)))]


def N1(k):
    result = a_priori_distribution(0)
    for i in range(0, k+1):
        result *= a_priori_distribution(i)
    return result


def N2(k):
    result = 0
    result2 = 1
    for i in range(0, k):
        result1 = (L-1)*L**i
        for j in range(0, k-i):
            result2 *= a_priori_distribution(j)
        result += result1*result2
    return result


def value_function(s):
    gamma = 0.9995
    if s.chi == 0:
        s2 = State(s.Y, 1, 0)
    else:
        s2 = State(s.Y, 0, 0)

    #value = (belief(s)*-1 + belief(s2)*-1) + gamma*(a_priori_distribution(s.Y)*111+1256)


def a_priori_distribution(M_conject, mean, std, k):
    #M_conject = 10
    #mean = M_conject/2
    #std = 1

    # Définir les paramètres de la distribution tronquée
    a = (0 - mean) / std  # limite inférieure standardisée
    b = (M_conject - mean) / std  # limite supérieure standardisée

    # Calcul de la probabilité que M dépasse k
    probability_M_gt_k = 1 - truncnorm.cdf(k, a, b, loc=mean, scale=std)


    #print(f"For k : {k}, Pr = {probability_M_gt_k}")

    return probability_M_gt_k


def poisson_survie(lambda_, valeur):
    return np.exp(-lambda_) * sum((lambda_ ** k) / math.factorial(k) for k in range(valeur + 1))


def poisson_positive_survie(lambda_, valeur):
    #return sum((lambda_**k) / ((math.exp(lambda_) -1) * math.factorial(k)) for k in range(valeur + 1) )
    proba_zero = poisson.pmf(0, lambda_)
    return (poisson.cdf(valeur, lambda_)) / (1 - proba_zero)


def poisson_truncated_pmf(x, lamb):
    if x == 0:
        return 0
    return (lamb ** x) / ((math.exp(lamb) - 1) * math.factorial(x))

def poisson_truncated_cdf(k, lamb):
    cdf = 0
    for x in range(1, k + 1):
        cdf += poisson_truncated_pmf(x, lamb)
    return cdf

def a(sigma):
    return -gamma * (1 - gamma ** (sigma + 1)) / (1 - gamma)


def plot_graph(all_evolution_reward, type_distrib):
    # Affichage du graphique avec des améliorations
    plt.figure(figsize=(12, 6))
    #markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']
    #colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
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
    sample = poisson.rvs(mu=lam)
    while sample < 2:
        sample = poisson.rvs(mu=lam)
    return sample


def simu_for_threshold(episodes, sigmas, lambda_, W):
    all_rewards = np.zeros(len(sigmas))
    for sigma in sigmas:
        print("Simulation : sigma = ", sigma)
        rewards = np.zeros(episodes)
        for i in range(0, episodes):
            reward = 0
            nb_step = 0
            nb_step_total = 0
            M = np.random.poisson(lam=lambda_)
            while M < 1:
                M = np.random.poisson(lam=lambda_)
            #M = sample_truncated_poisson(lambda_)
            #M = random.choices([1, 2, 3], weights=[1/3, 1/3, 1/3])[0]
            #M = 4
            if M > sigma:
                rewards[i] = -1 / (1 - gamma)
                #print("Reward = ", rewards[i])
                continue
            #print("M = ", M)
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
                #if(M == 2):
                    #print("Y: ", state.Y, "chi: ", state.chi, "T: ", state.T)
            # print("yes it works : nb_step_total = ", nb_step_total, ", M = ", M)
            #print("nb_step_total = ", nb_step_total)
            #print("nb_step_total = ", nb_step_total)
            for j in range(0, nb_step_total-1):
                reward += -gamma ** j
            reward += W * gamma ** (nb_step_total-1)


            rewards[i] = reward
            #print("Reward = ", rewards[i])
        all_rewards[sigma - 1] = np.mean(rewards)
    return all_rewards


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


def mean_reward(sigma, lambda_, W):
    somme = 0
    #proba = [0, 1/3, 1/3, 1/3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    m = 1

    #for m in range(0, sigma):
    while m <= sigma:
        somme += ((L**(-m) * gamma**m) / (1 - (1 - L**(-m))* gamma**(sigma+1))) *(np.exp(-lambda_) * (lambda_ ** m) / math.factorial(m))
        #somme += ((L ** (-m) * gamma ** m) / (1 - (1 - L ** (-m)) * gamma ** (sigma+1))) * proba[m]
        #print("rho = ", L**(-m))
        m += 1
    #reward = -(1 / (1 - gamma)) * (1 - poisson_truncated_cdf(sigma, lambda_)) + (W + (1 / (1 - gamma))) * somme
    print("sigma = ", sigma, "somme = ", somme)
    reward = -(1 / (1 - gamma)) + (W + (1 / (1 - gamma))) * somme

    #print("Reward formule = ", reward)

    return reward


if __name__ == '__main__':

    # M = 4
    # p_success = 1 / L ** (M - 1)

    episodes = 10000
    sigmas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lambda_ = 8
    #all_lambdas = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    W = 1e2


    all_rewards_simu = simu_for_threshold(episodes, sigmas, lambda_, W)
    #all_rewards = [all_rewards_simu]
    #plot_rewards(all_rewards, lambda_)

    all_rewards_formula = np.zeros(len(sigmas))
    for sigma in sigmas:
        all_rewards_formula[sigma-1] = mean_reward(sigma, lambda_, W)

    all_rewards = [all_rewards_simu, all_rewards_formula]

    for i in range(0, len(all_rewards_simu)):
        print("ratio = ", all_rewards_simu[i]/all_rewards_formula[i])

    plot_rewards(all_rewards, lambda_)













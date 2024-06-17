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
        if s.Y == M-1:
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


def get_M_From_Poisson(lambda_):
    # Générer une seule valeur suivant la loi de Poisson
    M = np.random.poisson(lambda_)
    #print('M =', M)
    # Calculer la probabilité P(X>=valeur)
    #probabilite = 1 - poisson_survie(lambda_, M)
    #print(probabilite)

    return M


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


if __name__ == '__main__':

    """
    rewards_total = np.zeros(10 - k_max)
    lower_bound = 1
    upper_bound = 10
    lambda_param = 0.5

    x = np.linspace(lower_bound, upper_bound, 1000)
    pdf_exponential = pdf_normalized(x, lambda_param)

    for i in range (k_max, 10):
        rewards_k = np.zeros(10)
        print("K_max = ", i)

        for j in range(0, 10):
            M = get_M_from_exponential_distribution(x, pdf_exponential)
            print("itération n°", j, "M = ", M)

            episodes = 1000
            all_values = np.zeros(episodes)
            epsilon = 0.09

            for episode in range(0, episodes):
                nb_step = 0
                nb_step_total = 0
                s = State(0, 0, 0)
                is_going_forward = True
                s = go_forward(s,M)
                nb_step += 1
                nb_step_total += 1
                while s.T != 1 and nb_step_total < 1e5:
                    #print(s.Y, s.chi, s.T)
                    if nb_step < i:
                        weights = [1-epsilon, epsilon]
                    else:
                        weights = [epsilon, 1-epsilon]
                    is_going_forward = random.choices([True, False], weights=weights)[0]
                    if is_going_forward:
                        s = go_forward(s,M)
                        nb_step += 1
                    else:
                        s = go_back()
                        nb_step = 0


                #print('step total = ', nb_step_total)
                reward = -1 * (nb_step_total - 1)
                if s.T == 1:
                    reward += 1e4
                all_values[episode] = reward
                #print("itération ", i, "terminée")

            rewards_k[j] = np.mean(all_values)
            #print(np.mean(all_values))
            #print(belief(State(2,0,0)))

        rewards_total[i-k_max] = np.mean(rewards_k)

    # Indices pour les valeurs de k
    indices = np.arange(k_max, 10)

    # Création du graphique
    plt.plot(indices, rewards_total)
    plt.xlabel('Valeur de k')
    plt.ylabel('Récompenses moyennes')
    plt.title('Récompenses moyennes en fonction de la valeur de k')
    plt.grid(True)
    plt.show()
    """

    """
    episodes = 10000

    epsilon = 0.01

    reward_comparison = []

    for t in range(0, 2):
        print(f"t = {t}")
        reward_for_all_sigma = np.zeros(12)
        for sigma in range(3, 15):
            print(f"sigma = {sigma}")
            all_values = np.zeros(episodes)
            M = 4

            all_reward = np.zeros(episodes)

            nb_pas_total = np.zeros(episodes)

            for episode in range(0, episodes):
                nb_pas = 0
                nb_step = 0
                nb_step_total = 0
                s = State(0, 0, 0)
                s = go_forward(s, M)
                nb_step += 1
                nb_step_total += 1
                while s.T != 1 and nb_step_total < 1e5:
                    if t == 0: # determinist
                    # print(s.Y, s.chi, s.T)
                        if nb_step < sigma:
                            s = go_forward(s, M)
                            nb_step += 1
                            nb_step_total +=1
                        else:
                            s = go_back()
                            nb_step = 0
                            nb_pas += 1
                            nb_step_total += 1
                    else: # epsilon-greedy
                        if nb_step < sigma:
                            weights = [1 - epsilon, epsilon]
                        else:
                            weights = [epsilon, 1 - epsilon]

                        is_going_forward = random.choices([True, False], weights=weights)[0]
                        if is_going_forward:
                            s = go_forward(s, M)
                            nb_step += 1
                            nb_step_total += 1
                        else:
                            s = go_back()
                            nb_step = 0
                            nb_pas += 1
                            nb_step_total += 1


                nb_pas_total[episode] = nb_pas

                for i in range(0, nb_step_total-1):
                   all_reward[episode] += -1 * gamma**i

                all_reward[episode] += 1e5 * gamma**nb_step_total

            reward_for_all_sigma[sigma-3] = np.mean(all_reward)

        reward_comparison.append(reward_for_all_sigma)

        #print(all_reward)

        #print(reward)

    print(reward_comparison)

    # Tracer le graphique
    sigmas = np.arange(3, 15)
    plt.plot(sigmas, reward_comparison[0], marker='o', label='Politique déterministe')
    plt.plot(sigmas, reward_comparison[1], marker='o', label='Politique Epsilon-Greedy')
    plt.xlabel('Sigma')
    plt.ylabel('Coût moyen')
    plt.title('Comparaison du coût: Politique déterministe vs Epsilon-Greedy')
    plt.legend()
    plt.grid(True)
    plt.show()

    #print(all_reward)
    #print(np.mean(nb_pas_total))
    #print(np.mean(all_reward))

        # print(np.mean(all_values))
        # print(belief(State(2,0,0)))
    #reward_for_all_sigma[sigma - 2] = np.mean(all_values)
    """

    test = 0


    #k_barre = 80

    """
    #----------Pour loi de Poisson--------------------
    all_evolution_reward = []
    for lambda_ in range(2, 10):
        evolution_reward = []
        #lambda_ = 3
        #print(k_barre*(sigma+1)+M-1)
        for sigma in range(1, 10):
            test = 0
            for k in range(1, (L**(lambda_))*(sigma+1)+lambda_-1):
                #test += a(sigma) * gamma ** ((k-1) * (sigma + 1))
                test -= gamma**k
            #print('étape 1 : ', test)
            #print('poisson : ', poisson_survie(lambda_, sigma))
            test = (test + gamma**(k+1)) * (poisson_survie(lambda_, sigma))
            #print('étape 2 : ', test)
            test = test + (-gamma / (1- gamma))* (1 - poisson_survie(lambda_, sigma))
            #print("étape 3 :", test)
        #test = test * (poisson_survie(4, sigma))
        #test += (-gamma ** (k_barre * (sigma + 1) + 1) * (1 - gamma ** (M - 1)) / (1 - gamma))
            evolution_reward.append(test)
            #print('sigma = ', sigma, 'reward = ', test)
        all_evolution_reward.append(evolution_reward)

    plot_graph(all_evolution_reward, 'Poisson')
    """


    # ----------Pour loi de Poisson v2--------------------
    #M = 4
    #p_success = 1 / L ** (M - 1)


    all_evolution_reward = []
    for lambda_ in range(2, 15):
        print('lambda_ = ', lambda_)
        g = 0
        g1 = 0
        g2 = 0
        #M = 4
        evolution_reward = []
        # lambda_ = 3
        # print(k_barre*(sigma+1)+M-1)
        for sigma in range(1, 15):
            #g1_temp = 0
            #on s'en fiche de M
            for m in range(2, sigma+1):
                p_success = 1 / L ** (m - 1)
                g1_temp = 0
                for i in range(1, 10000):
                    g1_temp += p_success * (1 - p_success) ** (i - 1) * (
                                -(1 - gamma ** ((i - 1) * (sigma + 1) + m)) / (1 - gamma))
                    # print(g1)
                #g1_temp *= lambda_ ** m / math.factorial(m) * math.exp(-lambda_)
                print('sigma = ', sigma, 'm = ', m, 'g1_temp = ', g1_temp)
                #g1_temp *= (lambda_ ** m) / ((math.exp(lambda_) - 1) * math.factorial(m))
                g1_temp *= poisson_truncated_pmf(m, lambda_)
                print('sigma = ', sigma, 'm = ', m, 'g1_temp_v2 = ', g1_temp)
                g1 += g1_temp

                print('sigma = ', sigma, 'm = ', m, 'g1 = ', g1)

            #g2 = -1 / (1 - gamma) * (1 - (poisson_survie(lambda_, sigma)))
            #g2 = -1 / (1 - gamma) * (1 - (poisson_positive_survie(lambda_, sigma)))
            g2 = -1 / (1 - gamma) * (1 - poisson_truncated_cdf(sigma, lambda_))

            print('sigma = ', sigma, 'proba = ', 1 - poisson_truncated_cdf(sigma, lambda_))
            #print('sigma = ', sigma, 'proba = ', 1 - poisson_survie(lambda_, sigma))

            print('sigma = ', sigma, 'g2 = ', g2)

            g = g1 + g2

            print('sigma = ', sigma, 'g = ', g)
            # print("étape 3 :", test)
            # test = test * (poisson_survie(4, sigma))
            # test += (-gamma ** (k_barre * (sigma + 1) + 1) * (1 - gamma ** (M - 1)) / (1 - gamma))
            evolution_reward.append(g)
            # print('sigma = ', sigma, 'reward = ', test)
        all_evolution_reward.append(evolution_reward)

    plot_graph(all_evolution_reward, 'Poisson')
    #plot_sub_graphics(all_evolution_reward)

    for i in range(len(all_evolution_reward)):
        print('i = ', i)
        print(max(all_evolution_reward[i]))


    """
    # ----------Pour loi de Poisson Positive--------------------
    #M = 4
    #p_success = 1 / L ** (M - 1)
    all_evolution_reward = []
    for lambda_ in range(2, 10):
        evolution_reward = []
        #lambda_ = 7
        # print(k_barre*(sigma+1)+M-1)
        for sigma in range(1, 10):
            g1_temp = 0
            for m in range(1, sigma):
                p_success = 1 / L ** (m - 1)
                for i in range(1, 1000):
                    g1_temp += p_success * (1 - p_success) ** (i - 1) * (
                            -(1 - gamma ** ((i - 1) * (sigma + 1) + m)) / (1 - gamma))
                    # print(g1)
                g1_temp *= (lambda_**m) / ((math.exp(lambda_) -1) * math.factorial(m))
                g1 += g1_temp
    
            g2 = -1 / (1 - gamma) * (1 - (poisson_positive_survie(lambda_, sigma)))
    
            g = g1 + g2
            # print("étape 3 :", test)
            # test = test * (poisson_survie(4, sigma))
            # test += (-gamma ** (k_barre * (sigma + 1) + 1) * (1 - gamma ** (M - 1)) / (1 - gamma))
            evolution_reward.append(g)
            # print('sigma = ', sigma, 'reward = ', test)
        all_evolution_reward.append(evolution_reward)

    

    plot_graph(all_evolution_reward, 'Poisson')
    #plot_sub_graphics(all_evolution_reward)
    """


    """
    #----------Pour loi Gaussienne--------------------
    # Paramètres de la distribution
    M_conject = 10
    std = 1


    all_evolution_reward = []
    for mean in range(M_conject-1, 1, -1):


        evolution_reward = []
        # lambda_ = 3
        # print(k_barre*(sigma+1)+M-1)
        for sigma in range(1, 10):
            test = 0
            for k in range(1, (L ** (mean)) * (sigma + 1) + mean - 1):
                # test += a(sigma) * gamma ** ((k-1) * (sigma + 1))
                test -= gamma ** k
            # print('étape 1 : ', test)
            # print('poisson : ', poisson_survie(lambda_, sigma))
            test = (test + gamma**(k+1)) * (1 - a_priori_distribution(M_conject, mean, std, sigma))
            # print('étape 2 : ', test)
            test = test + (-gamma / (1 - gamma)) * a_priori_distribution(M_conject, mean, std, sigma)
            # print("étape 3 :", test)
            # test = test * (poisson_survie(4, sigma))
            # test += (-gamma ** (k_barre * (sigma + 1) + 1) * (1 - gamma ** (M - 1)) / (1 - gamma))
            evolution_reward.append(test)
            # print('sigma = ', sigma, 'reward = ', test)
        all_evolution_reward.append(evolution_reward)
    

    plot_graph(all_evolution_reward, 'Gaussienne')
    """

    """

    # ----------Pour loi Gaussienne v2--------------------
    # Paramètres de la distribution
    M_conject = 10
    mean = round(M_conject/2)
    #std = 1

    all_evolution_reward = []
    #for mean in range(M_conject - 1, 1, -1):
    for std in range(1, 10):

        lim1 = (0 - mean) / std  # limite inférieure standardisée
        lim2 = (10 - mean) / std  # limite supérieure standardisée

        # Créer la distribution normale tronquée
        truncated_normal = truncnorm(lim1, lim2, loc=mean, scale=std)

        evolution_reward = []
        # lambda_ = 3
        # print(k_barre*(sigma+1)+M-1)
        for sigma in range(1, 10):
            g1_temp = 0
            for m in range(1, sigma):
                for i in range(1, 1000):
                    g1_temp += p_success * (1 - p_success) ** (i - 1) * (
                            -(1 - gamma ** ((i - 1) * (sigma + 1) + M)) / (1 - gamma))
                    # print(g1)
                g1_temp *= truncated_normal.pdf(sigma)
                g1 += g1_temp

            g2 = -1 / (1 - gamma) * (a_priori_distribution(M_conject, mean, std, sigma))

            g = g1 + g2
            evolution_reward.append(g)
            # print('sigma = ', sigma, 'reward = ', test)
        all_evolution_reward.append(evolution_reward)

    plot_graph(all_evolution_reward, 'Gaussienne')

    """



    """
    #----------Pour loi Uniforme--------------------
    all_evolution_reward = []
    for M_conject in range(4, 18, 2):
        mean = round(M_conject / 2)
        probability = 1 / M_conject
        evolution_reward = []
        # lambda_ = 3
        # print(k_barre*(sigma+1)+M-1)
        for sigma in range(1, 10):
            test = 0
            for k in range(1, (L ** (mean)) * (sigma + 1) + mean - 1):
                # test += a(sigma) * gamma ** ((k-1) * (sigma + 1))
                test -= gamma ** k
            # print('étape 1 : ', test)
            # print('poisson : ', poisson_survie(lambda_, sigma))
            test = (test + gamma**(k+1)) * probability
            # print('étape 2 : ', test)
            test = test + (-gamma / (1 - gamma)) * (1 - probability)
            # print("étape 3 :", test)
            # test = test * (poisson_survie(4, sigma))
            # test += (-gamma ** (k_barre * (sigma + 1) + 1) * (1 - gamma ** (M - 1)) / (1 - gamma))
            evolution_reward.append(test)
            # print('sigma = ', sigma, 'reward = ', test)
        all_evolution_reward.append(evolution_reward)

    plot_graph(all_evolution_reward, 'Uniforme')
    """















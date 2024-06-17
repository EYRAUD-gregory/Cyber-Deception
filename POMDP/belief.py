from scipy.stats import truncnorm
import matplotlib.pyplot as plt

L = 3

class State():
    def __init__(self, Y, chi, T):
        self.Y = Y
        self.chi = chi
        self.T = T


def belief(s):
    if s.chi == 0 and s.T == 0:
        return N1(s.Y) / (N1(s.Y) + N2(s.Y))
    elif s.chi == 1 and s.T == 0:
        return N2(s.Y) / (N1(s.Y) + N2(s.Y))


def N1(k):
    result = a_priori_distribution(0)
    for i in range(0, k+1):
        result *= a_priori_distribution(i)
    return result


def N2(k):
    result = 0
    result2 = 1
    for i in range(0, k-1):
        result1 = (L-1)*L**i
        for j in range(0, k-i):
            result2 *= a_priori_distribution(j)
        result += result1*result2
    return result


def a_priori_distribution(k):
    M_conject = 100
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


"""

s1 = State(1,0,0)
s2 = State(2,0,0)
s3 = State(3,0,0)
s4 = State(4,0,0)

print("right path")
print(belief(s1))
print(belief(s2))
print(belief(s3))
print(belief(s4))

s1 = State(1,1,0)
s2 = State(2,1,0)
s3 = State(3,1,0)
s4 = State(4,1,0)


print("wrong path")
print(belief(s1))
print(belief(s2))
print(belief(s3))
print(belief(s4))
"""

Dk = []

for i in range (0, 10):
    Dk.append(1 / (N1(i) + N2(i)))

plt.plot(Dk)
plt.title("Evolution de Dk")
plt.show()
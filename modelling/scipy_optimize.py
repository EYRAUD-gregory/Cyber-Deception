from scipy.optimize import minimize
import numpy as np


# Définition de la fonction à minimiser
def function(p):
    M = 4
    L = 3
    return M-1 + (2*p[0] + 3*p[1]*(1 - p[0]) + 4*p[2] * ((1 - p[0])*(1-p[1]))) * ((L**(M-1) / ((1-p[0])*(1-p[1]))) - 1)


# Pour exprimer la contrainte p2 >= 0
def ineq_constraint_p2_0(p):
    return p[0]


# Pour exprimer la contrainte p3 >= 0
def ineq_constraint_p3_0(p):
    return p[1]


# Pour exprimer la contrainte p4 >= 0
def ineq_constraint_p4_0(p):
    return p[2]


# Pour exprimer la contrainte p2 <= 1
def ineq_constraint_p2_1(p):
    return 1 - p[0]


# Pour exprimer la contrainte p3 <= 1
def ineq_constraint_p3_1(p):
    return 1 - p[1]


# Pour exprimer la contrainte p4 <= 1
def ineq_constraint_p4_1(p):
    return 1 - p[2]


# Valeurs initiales des variables
initial_guess = np.array([0.1, 0.1, 0.1])

# Définition des contraintes
constraints = [{'type': 'ineq', 'fun': ineq_constraint_p2_0},
               {'type': 'ineq', 'fun': ineq_constraint_p3_0},
               {'type': 'ineq', 'fun': ineq_constraint_p4_0},
               {'type': 'ineq', 'fun': ineq_constraint_p2_1},
               {'type': 'ineq', 'fun': ineq_constraint_p3_1},
               {'type': 'ineq', 'fun': ineq_constraint_p4_1}]

# Minimisation de la fonction avec les contraintes
result = minimize(function, initial_guess, constraints=constraints)

# Affichage du résultat
print("Minimum trouvé aux coordonnées:", result.x)
print("Valeur de la fonction objectif au minimum trouvé:", result.fun)

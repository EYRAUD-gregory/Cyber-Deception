import gurobipy as gp

# Create a new model
m = gp.Model("qcp")

M = 4
L = 3

# Create variables
p2 = m.addVar(vtype='C', name="p2")
p3 = m.addVar(vtype='C', name="p3")
p4 = m.addVar(vtype='C', name="p4")
f = m.addVar(vtype='C', name="f")
g = m.addVar(vtype='C', name="g")

# Create auxiliary variables
y = m.addVar(vtype='C', name="y")
frac = m.addVar(vtype='C', name="frac")
frac_inv = m.addVar(vtype='C', name="frac_inv")

# Add linearization constraint
m.addConstr(y == (1 - p2) * (1 - p3))

m.addConstr(f == 2*p2 + 3*p3 * (1 - p2) + 4*p4 * y)

m.addConstr(frac == y / L**3)
#  Pour avoir l'inverse de y / L^3, gurobi ne le supporte pas directement sans passer par une contrainte
m.addConstr(frac * frac_inv == 1)

m.addConstr(g == frac_inv - 1)

# Set objective function : 3 + 2 * p2 + 3 * p3(1-p2) + 4 * p4( ( 1-p2 )( 1-p3 ) ) * [( L^3 / ( ( 1-p2 )( 1-p3 ) ) ) -1 ]
m.setObjective(3 + f * g, gp.GRB.MINIMIZE)

# Add variable bounds
m.addConstr(p2 >= 0)
m.addConstr(p2 <= 1)
m.addConstr(p3 >= 0)
m.addConstr(p3 <= 1)
m.addConstr(p4 >= 0)
m.addConstr(p4 <= 1)

# Solve the model
m.optimize()

# Print optimal objective value and solution values
print(f"Optimal objective value: {m.objVal}")
print(f"Solution values: p2={p2.X}, p3={p3.X}, p4={p4.X}")
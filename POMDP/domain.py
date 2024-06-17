import pomdp_py


class State(pomdp_py.State):
    def __init__(self, k, chi, T):
        if k < 0 or (chi < 0 or chi > 1) or (T < 0 or T > 1):
            raise ValueError(f"Invalid state: ({k}, {chi}, {T})")
        self.k = k
        self.chi = chi
        self.T = T

    def __hash__(self):
        # Le hachage est calculé en utilisant les valeurs des attributs k, chi et T
        return hash((self.k, self.chi, self.T))

    def __eq__(self, state):
        # Vérifie si les valeurs des attributs k, chi et T sont égales
        if not isinstance(state, State):
            return False
        return self.k == state.k and self.chi == state.chi and self.T == state.T

    def __str__(self):
        return f"({self.k}, {self.chi}, {self.T})"


class Action(pomdp_py.Action):
    def __init__(self, name):
        if name != "go_back" and name != "go_forward":
            raise ValueError(f"Invalid action: ({name})")
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, action):
        if not isinstance(action, Action):
            return False
        return self.name == action.name

    def __str__(self):
        return self.name


class Observation(pomdp_py.Observation):
    def __init__(self, Y, T):
        if Y < 0  or (T < 0 or T > 1):
            raise ValueError(f"Invalid observation: ({Y}, {T})")
        self.Y = Y
        self.T = T

    def __hash__(self):
        return hash((self.Y, self.T))

    def __eq__(self, observation):
        if not isinstance(observation, Observation):
            return False
        return self.Y == observation.Y and self.T == observation.T

    def __str__(self):
        return f"({self.Y}, {self.T})"
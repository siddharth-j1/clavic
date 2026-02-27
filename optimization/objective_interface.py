# optimization/objective_interface.py

class ObjectiveInterface:

    def __init__(self, certified_policy, objective_function):
        self.certified_policy = certified_policy
        self.objective_function = objective_function

    def evaluate(self, xi):
        trace = self.certified_policy.rollout(xi)
        return self.objective_function(trace)
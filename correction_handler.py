import numpy as np

# Constant definitions
METHOD_SATURATION = 0
METHOD_MIDPOINT_TARGET = 1
METHOD_MIDPOINT_BEST = 2
METHOD_UNIF = 3
METHOD_MIRROR = 5
METHOD_TOROIDAL = 6
METHOD_EXPC_TARGET = 8
METHOD_EXPC_BEST = 9
METHOD_VECTOR_TARGET = 11
METHOD_VECTOR_BEST = 12

NONE = -1

corr_names={
METHOD_SATURATION : 'sat',
METHOD_MIDPOINT_TARGET : 'midT',
METHOD_MIDPOINT_BEST : 'midB',
METHOD_UNIF : 'unif',
METHOD_MIRROR : 'mir',
METHOD_TOROIDAL : 'tor',
METHOD_EXPC_TARGET : 'expC_T',
METHOD_EXPC_BEST : 'expC_B',
METHOD_VECTOR_TARGET : 'vectT',
METHOD_VECTOR_BEST : 'vectB'
}


class CorrectionHandler:
    def __init__(self, problem):
        self.problem = problem
        self.lb = self.problem.f.lower_bounds
        self.ub = self.problem.f.upper_bounds
        self.gamma = -1

    def correction_component(self, method, lower, upper, component, target, best, population_mean):
        """Applies correction to a component based on the specified method if it is out of bounds.
            Args:
                method (str): The correction method.
                lower (float): The lower bound.
                upper (float): The upper bound.
                component (float): The component from element.
                target (float): The component from target.
                best (float): The component from the best.
            Returns:
                tuple: Corrected component value and a boolean indicating if it was repaired.
                """
        #print(f"Method: {method}, Component: {component}, Lower: {lower}, Upper: {upper}")

        if component < lower:
            return self.correct_lower(method, lower, upper, component, target, best, population_mean)
        elif component > upper:
            return self.correct_upper(method, lower, upper, component, target, best, population_mean)
        else:
            return component, False

    def correct_lower(self, method, lower, upper,component, target, best, population_mean):
        """Applies correction to a component which exceed its lower bound."""
        repaired = True
        component_correction_methods = {
            METHOD_SATURATION: lambda: lower,
            METHOD_MIDPOINT_TARGET: lambda: (lower + target) / 2,
            METHOD_MIDPOINT_BEST: lambda: (lower + best)/2,
            METHOD_EXPC_TARGET: lambda: lower - np.log(1 + np.random.uniform(0, 1) * (np.exp(lower - target)-1)),
            METHOD_EXPC_BEST: lambda: lower - np.log(1 + np.random.uniform(0, 1) * (np.exp(lower - best)-1)),
            METHOD_UNIF: lambda: np.random.uniform(lower,upper),
            METHOD_MIRROR: lambda: 2*lower-component,
            METHOD_TOROIDAL: lambda: upper-lower+component,
        }
        component = component_correction_methods.get(method, lambda: component)()
        return component, repaired

    def correct_upper(self, method, lower, upper, component, target, best, population_mean):
        """Applies correction to a component which exceed its upper bound."""
        repaired = True
        component_correction_methods = {
            METHOD_SATURATION: lambda: upper,
            METHOD_MIDPOINT_TARGET: lambda: (upper + target) / 2,
            METHOD_MIDPOINT_BEST: lambda: (upper + best) / 2,
            METHOD_EXPC_TARGET: lambda: upper + np.log(1 + (1 - np.random.uniform(0, 1)) * (np.exp(target - upper) - 1)),
            METHOD_EXPC_BEST: lambda: upper + np.log(1 + (1 - np.random.uniform(0, 1)) * (np.exp(best - upper) - 1)),
            METHOD_UNIF: lambda: np.random.uniform(lower, upper),
            METHOD_MIRROR: lambda: 2*upper-component,
            METHOD_TOROIDAL: lambda: lower-upper+component,
        }

        if lower >= upper:
            raise ValueError(
                f"Limita inferioară trebuie să fie mai mică decât limita superioară. Lower: {lower}, Upper: {upper}")
        component = component_correction_methods.get(method, lambda: component)()
        return component, repaired

    def correction_vectorial(self, lower, upper, trial_vector, R):
        """
        Applies correction on the entire vector, including the feasible components.
        Args:
            lower (float): The lower bound.
            upper (float): The upper bound.
            trial_vector (ndarray): The trial vector to be corrected.
            R (ndarray): The reference vector.
        Returns:
            tuple: A tuple containing the corrected vector, a boolean array indicating the repaired indices,
                   and the gamma value.
        """
        alpha = np.zeros(self.problem.dim)
        repaired = [False] * self.problem.dim
        eps0 = 10 ** (-8)
        for j in range(self.problem.dim):
            if trial_vector[j] < lower[j]:# -eps0:
                if R[j] == trial_vector[j]:
                    alpha[j] = 1
                else:
                    alpha[j] = round((R[j] - lower[j]) / (R[j] - trial_vector[j]), 10)
                repaired[j] = True
            elif trial_vector[j] > upper[j]:  # +eps0:
                if R[j] == trial_vector[j]:
                    alpha[j] = 1
                else:
                    alpha[j] = round((upper[j] - R[j]) / (trial_vector[j] - R[j]), 10)
                repaired[j] = True
            else:
                alpha[j] = 1
        self.gamma = alpha.min()

        if (self.gamma < 0) or (self.gamma > 1):
            print("!!! gamma=", self.gamma, " alpha=", alpha)
            if (self.gamma < 0):
                self.gamma = 0
            else:
                self.gamma = 1

        repaired_vector = self.gamma * trial_vector + (1 - self.gamma) * R  # repaired vector

        for j in range(self.problem.dim):
            if repaired_vector[j] < lower[j]:
                # print("projected on lower")
                repaired_vector[j] = lower[j]
            if repaired_vector[j] > upper[j]:
                # print("projected on upper")
                repaired_vector[j] = upper[j]

        return repaired_vector, repaired, self.gamma


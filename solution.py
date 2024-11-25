import numpy as np
from typing import Tuple


class Solution:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = y


class HNOProblem:
    def __init__(self, dimension=5, instance=1):
        """Initialize the HNO problem.

        Args:
            dimension (int): Problem dimension (default is 5)
            instance (int): Problem instance (default is 1)
        """
        self.dimension = dimension
        self.instance = instance

        # Define bounds
        self.lower_bounds = np.array([78, 33, 27, 27, 27])
        self.upper_bounds = np.array([102, 45, 45, 45, 45])

        # Known optimum
        optimum_x = [78, 33, 29.9953, 45, 36.7758]
        optimum_y = -30665.5386717834
        self.optimum = Solution(optimum_x, optimum_y)

        # Problem properties
        self.is_minimization = True
        self.constraint_violation = 0

    def __call__(self, x):
        """Evaluate the objective function and constraints.

        Args:
            x (numpy.ndarray): Solution vector to evaluate

        Returns:
            tuple: (objective_value, constraint_violation)
        """
        return self.evaluate(x)

    def _evaluate_objective(self, x):
        """Compute the objective function value.

        Args:
            x (numpy.ndarray): Solution vector

        Returns:
            float: Objective function value
        """
        x1, x2, x3, x4, x5 = x
        return 5.3578547 * x3 ** 2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141

    def _evaluate_constraints(self, x):
        """Evaluate all constraints and compute violation.

        Args:
            x (numpy.ndarray): Solution vector

        Returns:
            float: Sum of constraint violations
        """
        x1, x2, x3, x4, x5 = x

        # Define constraints

        g1 = 85.334407 + 0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022053 * x3 * x5 - 92
        g2 = -85.334407 - 0.0056858 * x2 * x5 - 0.0006262 * x1 * x4 + 0.0022053 * x3 * x5
        g3 = 80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3 ** 2 - 110
        g4 = -80.51249 - 0.0071317 * x2 * x5 - 0.0029955 * x1 * x2 - 0.0021813 * x3 ** 2 + 90
        g5 = 9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4 - 25
        g6 = -9.300961 - 0.0047026 * x3 * x5 - 0.0012547 * x1 * x3 - 0.0019085 * x3 * x4 + 20

        # Calculate total violation
        violations = [
            max(0, g1),
            max(0, g2),
            max(0, g3),
            max(0, g4),
            max(0, g5),
            max(0, g6)
        ]
        return sum(violations)

    def evaluate(self, x):
        """Main evaluation method.

        Args:
            x (numpy.ndarray): Solution vector to evaluate

        Returns:
            tuple: (objective_value, constraint_violation)
        """

        # Evaluate objective and constraints
        objective_value = self._evaluate_objective(x)
        self.constraint_violation = self._evaluate_constraints(x)

        return objective_value, self.constraint_violation


class PVDProblem:
    """Pressure Vessel Design Problem"""

    def __init__(self, dimension: int = 4, instance: int = 1):
        self.dimension = dimension
        self.instance = instance

        # Define bounds
        self.lower_bounds = np.array([1, 1, 10, 10])
        self.upper_bounds = np.array([100, 100, 200, 200])

        # Known optimum
        optimum_x = [0.8125, 0.4375, 42.0984, 176.6366]
        optimum_y = 6059.7143350385
        self.optimum = Solution(optimum_x, optimum_y)

        # Problem properties
        self.is_minimization = True
        self.constraint_violation = 0
        self.discrete_values = np.arange(1, 100)

    def discretize(self, x: float) -> float:
        """Discretize x to nearest integer between 1 and 99, then multiply by 0.0625"""
        nearest_int = self.discrete_values[np.argmin(np.abs(self.discrete_values - x))]
        return nearest_int * 0.0625

    def _evaluate_objective(self, x: np.ndarray) -> float:
        x1, x2, x3, x4 = x
        x1, x2 = self.discretize(x1), self.discretize(x2)
        return 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 ** 2 + 3.1661 * x1 ** 2 * x4 + 19.84 * x1 ** 2 * x3

    def _evaluate_constraints(self, x: np.ndarray) -> float:
        x1, x2, x3, x4 = x
        x1, x2 = self.discretize(x1), self.discretize(x2)

        g1 = -x1 + 0.0193 * x3
        g2 = -x2 + 0.00954 * x3
        g3 = -np.pi * x3 ** 2 * x4 - (4 / 3) * np.pi * x3 ** 3 + 1296000
        g4 = x4 - 240

        violations = [max(0, g1), max(0, g2), max(0, g3), max(0, g4)]
        return sum(violations)

    def __call__(self, x: np.ndarray) -> Tuple[float, float]:
        return self.evaluate(x)

    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        objective_value = self._evaluate_objective(x)
        self.constraint_violation = self._evaluate_constraints(x)

        return objective_value, self.constraint_violation


class TCSDProblem:
    """Tension/Compression Spring Design Problem"""

    def __init__(self, dimension: int = 3, instance: int = 1):
        self.dimension = dimension
        self.instance = instance

        # Define bounds
        self.lower_bounds = np.array([0.05, 0.25, 2.00])
        self.upper_bounds = np.array([2.00, 1.30, 15.0])

        # Known optimum
        optimum_x = [0.051587, 0.354268, 11.434058]
        optimum_y = 0.012665
        self.optimum = Solution(optimum_x, optimum_y)

        # Problem properties
        self.is_minimization = True
        self.constraint_violation = 0

    def _evaluate_objective(self, x: np.ndarray) -> float:
        x1, x2, x3 = x
        return x1 ** 2 * x2 * (x3 + 2)

    def _evaluate_constraints(self, x: np.ndarray) -> float:
        x1, x2, x3 = x

        g1 = 1 - (x2 ** 3 * x3) / (71785 * x1 ** 4)
        g2 = (4 * x2 ** 2 - x1 * x2) / (12566 * (x2 * x1 ** 3 - x1 ** 4)) + 1 / (5108 * x1 ** 2) - 1
        g3 = 1 - 140.45 * x1 / (x2 ** 2 * x3)
        g4 = (x1 + x2) / 1.5 - 1

        violations = [max(0, g1), max(0, g2), max(0, g3), max(0, g4)]
        return sum(violations)

    def __call__(self, x: np.ndarray) -> Tuple[float, float]:
        return self.evaluate(x)

    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        objective_value = self._evaluate_objective(x)
        self.constraint_violation = self._evaluate_constraints(x)
        return objective_value, self.constraint_violation


class SRDProblem:
    """Speed Reducer Design Problem"""

    def __init__(self, dimension: int = 7, instance: int = 1):
        self.dimension = dimension
        self.instance = instance

        # Define bounds
        self.lower_bounds = np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0])
        self.upper_bounds = np.array([3.6, 0.8, 29, 8.3, 8.3, 3.9, 5.5])

        # Known optimum
        optimum_x = [3.5, 0.7, 17, 7.3, 7.715, 3.35, 5.287]
        optimum_y =  2994.47107
        self.optimum = Solution(optimum_x, optimum_y)

        # Problem properties
        self.is_minimization = True
        self.constraint_violation = 0
        self.discrete_values_x3 = np.arange(17, 29)

    def discretize_x3(self, x3: float) -> int:
        """Discretize x3 to nearest integer in range [17, 28]"""
        return self.discrete_values_x3[np.argmin(np.abs(self.discrete_values_x3 - x3))]

    def _evaluate_objective(self, x: np.ndarray) -> float:
        x1, x2, x3, x4, x5, x6, x7 = x
        x3 = self.discretize_x3(x3)
        return (0.7854 * x1 * x2 ** 2 * (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934) -
                1.508 * x1 * (x6 ** 2 + x7 ** 2) +
                7.4777 * (x6 ** 3 + x7 ** 3) +
                0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2))

    def _evaluate_constraints(self, x: np.ndarray) -> float:
        x1, x2, x3, x4, x5, x6, x7 = x
        x3 = self.discretize_x3(x3)

        constraints = [
            27 / (x1 * x2 ** 2 * x3) - 1,
            397.5 / (x1 * x2 ** 2 * x3 ** 2) - 1,
            1.93 * x4 ** 3 / (x2 * x3 * x6 ** 4) - 1,
            1.93 * x5 ** 3 / (x2 * x3 * x7 ** 4) - 1,
            np.sqrt((745 * x4 / (x2 * x3)) ** 2 + 16.9e6) / (110.0 * x6 ** 3) - 1,
            np.sqrt((745 * x5 / (x2 * x3)) ** 2 + 157.5e6) / (85.0 * x7 ** 3) - 1,
            x2 * x3 / 40 - 1,
            5 * x2 / x1 - 1,
            x1 / (12 * x2) - 1,
            (1.5 * x6 + 1.9) / x4 - 1,
            (1.1 * x7 + 1.9) / x5 - 1
        ]

        return sum(max(0, g) for g in constraints)

    def __call__(self, x: np.ndarray) -> Tuple[float, float]:
        return self.evaluate(x)

    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        objective_value = self._evaluate_objective(x)
        self.constraint_violation = self._evaluate_constraints(x)
        return objective_value, self.constraint_violation


class TBTDProblem:
    """Three Bar Truss Design Problem"""

    def __init__(self, dimension: int = 2, instance: int = 1):
        self.dimension = dimension
        self.instance = instance

        # Define bounds
        self.lower_bounds = np.array([0, 0])
        self.upper_bounds = np.array([1, 1])

        # Known optimum
        optimum_x = [0.78868473, 0.4082211]
        optimum_y = 263.8958434
        self.optimum = Solution(optimum_x, optimum_y)

        # Problem properties
        self.is_minimization = True
        self.constraint_violation = 0

    def _evaluate_objective(self, x: np.ndarray) -> float:
        x1, x2 = x
        l = 100  # cm
        return (2 * np.sqrt(2) * x1 + x2) * l

    def _evaluate_constraints(self, x: np.ndarray) -> float:
        x1, x2 = x
        P = 2  # kN/cm^2
        sigma = 2  # kN/cm^2

        g1 = (np.sqrt(2) * x1 + x2) / (np.sqrt(2) * x1 ** 2 + 2 * x1 * x2) * P - sigma
        g2 = x2 / (np.sqrt(2) * x1 ** 2 + 2 * x1 * x2) * P - sigma
        g3 = 1 / (x1 + np.sqrt(2) * x2) * P - sigma

        violations = [max(0, g1), max(0, g2), max(0, g3)]
        return sum(violations)

    def __call__(self, x: np.ndarray) -> Tuple[float, float]:
        return self.evaluate(x)

    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        objective_value = self._evaluate_objective(x)
        self.constraint_violation = self._evaluate_constraints(x)
        return objective_value, self.constraint_violation


class GTDProblem:
    """Gear Train Design Problem"""

    def __init__(self, dimension: int = 4, instance: int = 1):
        self.dimension = dimension
        self.instance = instance

        # Define bounds
        self.lower_bounds = np.array([12, 12, 12, 12])
        self.upper_bounds = np.array([61, 61, 61, 61])

        # Known optimum
        optimum_x = [43, 19, 16, 49]
        optimum_y = 2.7e-12
        self.optimum = Solution(optimum_x, optimum_y)

        # Problem properties
        self.is_minimization = True
        self.constraint_violation = 0
        self.discrete_values = np.arange(12, 61)

    def discretize(self, x: float) -> int:
        """Discretize to nearest integer in range [12, 60]"""
        return self.discrete_values[np.argmin(np.abs(self.discrete_values - x))]

    def _evaluate_objective(self, x: np.ndarray) -> float:
        x_discrete = np.array([self.discretize(xi) for xi in x])
        x1, x2, x3, x4 = x_discrete
        return (1 / 6.931 - x3 * x2 / (x1 * x4)) ** 2

    def _evaluate_constraints(self, x: np.ndarray) -> float:
        # No explicit constraints
        return 0

    def __call__(self, x: np.ndarray) -> Tuple[float, float]:
        return self.evaluate(x)

    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        objective_value = self._evaluate_objective(x)
        self.constraint_violation = self._evaluate_constraints(x)
        return objective_value, self.constraint_violation


class CBHDProblem:
    """Corrugated Bulkhead Design"""

    def __init__(self, dimension: int = 4, instance: int = 1):
        self.dimension = dimension
        self.instance = instance

        # Define bounds
        self.lower_bounds = np.array([0, 0, 0, 0])
        self.upper_bounds = np.array([100, 100, 100, 5])

        # Known optimum
        optimum_x = [ 57.6923073,  34.1476202,  57.6923072, 1.05]
        optimum_y =  6.84295801
        self.optimum = Solution(optimum_x, optimum_y)

        # Problem properties
        self.is_minimization = True
        self.constraint_violation = 0

    def _evaluate_objective(self, x: np.ndarray) -> float:
        x1, x2, x3, x4 = x
        return 5.885 * (x1 + x3) * x4 / (x1 + np.sqrt(abs(x3 ** 2 - x2 ** 2)))

    def _evaluate_constraints(self, x: np.ndarray) -> float:
        x1, x2, x3, x4 = x

        g1 = -x2 * x4 * (2 * x1 / 5 + x3 / 6) + 8.94 * (x1 + np.sqrt(abs(x3 ** 2 - x2 ** 2)))
        g2 = -x2 ** 2 * x4 * (x1 / 5 + x3 / 12) + 2.2 * (8.94 * (x1 + np.sqrt(abs(x3 ** 2 - x2 ** 2)))) ** (4 / 3)
        g3 = 0.0156 * x1 - x4 + 0.15
        g4 = 0.0156 * x3 - x4 + 0.15
        g5 = -x4 + 1.05
        g6 = x2 - x3

        violations = [max(0, g1), max(0, g2), max(0, g3), max(0, g4), max(0, g5), max(0, g6)]
        return sum(violations)

    def __call__(self, x: np.ndarray) -> Tuple[float, float]:
        return self.evaluate(x)

    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        objective_value = self._evaluate_objective(x)
        self.constraint_violation = self._evaluate_constraints(x)
        return objective_value, self.constraint_violation


class TCDProblem:
    """Tubular Column Design Problem"""

    def __init__(self, dimension: int = 2, instance: int = 1):
        self.dimension = dimension
        self.instance = instance

        # Define bounds
        self.lower_bounds = np.array([2, 0.2])
        self.upper_bounds = np.array([14, 0.8])

        # Known optimum
        optimum_x = [5.45115623, 0.29196547]
        optimum_y = 26.4994969
        self.optimum = Solution(optimum_x, optimum_y)

        # Problem properties
        self.is_minimization = True
        self.constraint_violation = 0

    def _evaluate_objective(self, x: np.ndarray) -> float:
        x1, x2 = x
        return 9.8 * x1 * x2 + 2 * x1

    def _evaluate_constraints(self, x: np.ndarray) -> float:
        x1, x2 = x
        P = 2500  # kgf
        sigma_y = 500  # kgf/cm^2
        L = 250  # cm
        E = 8.5e5  # kgf/cm^2

        g1 = P / (np.pi * x1 * x2 * sigma_y) - 1
        g2 = 8 * P * L ** 2 / (np.pi ** 3 * E * x1 * x2 * (x1 ** 2 + x2 ** 2)) - 1
        g3 = 2 / x1 - 1
        g4 = x1 / 14 - 1
        g5 = 0.2 / x2 - 1
        g6 = x2 / 0.8 - 1

        violations = [max(0, g1), max(0, g2), max(0, g3), max(0, g4), max(0, g5), max(0, g6)]
        return sum(violations)

    def __call__(self, x: np.ndarray) -> Tuple[float, float]:
        return self.evaluate(x)

    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        objective_value = self._evaluate_objective(x)
        self.constraint_violation = self._evaluate_constraints(x)
        return objective_value, self.constraint_violation


class IBDProblem:
    """I-Beam Design Problem"""

    def __init__(self, dimension: int = 4, instance: int = 1):
        self.dimension = dimension
        self.instance = instance

        # Define bounds
        self.lower_bounds = np.array([10, 10, 0.9, 0.9])
        self.upper_bounds = np.array([50, 80, 5, 5])

        # Known optimum
        optimum_x = [80, 50, 0.9, 2.3217]
        optimum_y = 0.0130741
        self.optimum = Solution(optimum_x, optimum_y)

        # Problem properties
        self.is_minimization = True
        self.constraint_violation = 0

    def _evaluate_objective(self, x: np.ndarray) -> float:
        x1, x2, x3, x4 = x
        return 5000 / (x3 * (x2 - 2 * x4) ** 3 / 12 + x1 * x4 ** 3 / 6 + 2 * x1 * x4 * ((x2 - x4) / 2) ** 2)

    def _evaluate_constraints(self, x: np.ndarray) -> float:
        x1, x2, x3, x4 = x

        g1 = 2 * x1 * x4 + x3 * (x2 - 2 * x4) - 300
        g2 = (180000 * x2) / (x3 * (x2 - 2 * x4) ** 3 + 2 * x1 * x4 * (4 * x4 ** 2 + 3 * x2 * (x2 - 2 * x4))) + \
             (15000 * x1) / (x3 ** 3 * (x2 - 2 * x4) + 2 * x3 * x4 ** 3) - 16

        violations = [max(0, g1), max(0, g2)]
        return sum(violations)

    def __call__(self, x: np.ndarray) -> Tuple[float, float]:
        return self.evaluate(x)

    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        objective_value = self._evaluate_objective(x)
        self.constraint_violation = self._evaluate_constraints(x)
        return objective_value, self.constraint_violation


class CBDProblem:
    """Cantilever Beam Design Problem"""

    def __init__(self, dimension: int = 5, instance: int = 1):
        self.dimension = dimension
        self.instance = instance

        # Define bounds
        self.lower_bounds = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        self.upper_bounds = np.array([100, 100, 100, 100, 100])

        # Known optimum
        optimum_x = [6.01545, 5.31066, 4.4880, 3.50528, 2.15428]
        optimum_y = 1.33995
        self.optimum = Solution(optimum_x, optimum_y)

        # Problem properties
        self.is_minimization = True
        self.constraint_violation = 0

    def _evaluate_objective(self, x: np.ndarray) -> float:
        return 0.0624 * sum(x)

    def _evaluate_constraints(self, x: np.ndarray) -> float:
        g1 = 61 / x[0] ** 3 + 37 / x[1] ** 3 + 19 / x[2] ** 3 + 7 / x[3] ** 3 + 1 / x[4] ** 3 - 1

        return max(0, g1)

    def __call__(self, x: np.ndarray) -> Tuple[float, float]:
        return self.evaluate(x)

    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        objective_value = self._evaluate_objective(x)
        self.constraint_violation = self._evaluate_constraints(x)
        return objective_value, self.constraint_violation


def create_ibd_problem(dim: int = 4, instance: int = 1) -> IBDProblem:
    return IBDProblem(dimension=dim, instance=instance)


def create_cbd_problem(dim: int = 5, instance: int = 1) -> CBDProblem:
    return CBDProblem(dimension=dim, instance=instance)


def create_tbtd_problem(dim: int = 2, instance: int = 1) -> TBTDProblem:
    return TBTDProblem(dimension=dim, instance=instance)


def create_gtd_problem(dim: int = 4, instance: int = 1) -> GTDProblem:
    return GTDProblem(dimension=dim, instance=instance)


def create_cbhd_problem(dim: int = 4, instance: int = 1) -> CBHDProblem:
    return CBHDProblem(dimension=dim, instance=instance)


def create_tcd_problem(dim: int = 2, instance: int = 1) -> TCDProblem:
    return TCDProblem(dimension=dim, instance=instance)


def create_hno_problem(dim=5, instance=1):
    return HNOProblem(dimension=dim, instance=instance)


def create_pvd_problem(dim: int = 4, instance: int = 1) -> PVDProblem:
    return PVDProblem(dimension=dim, instance=instance)


def create_tcsd_problem(dim: int = 3, instance: int = 1) -> TCSDProblem:
    return TCSDProblem(dimension=dim, instance=instance)


def create_srd_problem(dim: int = 7, instance: int = 1) -> SRDProblem:
    return SRDProblem(dimension=dim, instance=instance)
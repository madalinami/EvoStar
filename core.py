from time import time
import copy
from monitoring_population import *
from correction_handler import *
from solution import *

def time_checker(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        ms = round(te - ts)
        m, s = divmod(ms, 60)
        h, m = divmod(m, 60)
        print(f'Execution time: {h} h {m} m {s} s')
        return result

    return timed

class HashTable:
    def __init__(self, precision=25):
        self.table = {}
        self.precision = precision

    def get_key(self, individual):
        rounded_coords = [round(x, self.precision) for x in individual.x]
        return tuple(rounded_coords)

    def insert(self, individual):
        key = self.get_key(individual)
        if key in self.table:
            return True
        else:
            self.table[key] = individual
            return False

    def contains(self, individual):
        key = self.get_key(individual)
        return key in self.table

class ContinuousOptimization:
    def __init__(self, dim, fct_name, inst): #, near_bounds, index_params):
        """Constructor for the ContinuousOptimization class.
            Args:
                dim (int): The dimensionality of the optimization problem.
                fct_name (str): The name of the evaluation function to be used.
        """
        self.dim = dim
        self.eval_fct = fct_name
        self.inst = inst
        if fct_name == "HNO":
            self.f = create_hno_problem(dim=dim, instance=inst)
            self.lower_bounds = self.f.lower_bounds
            self.upper_bounds = self.f.upper_bounds

        elif fct_name == "PVD":
            self.f = create_pvd_problem(dim=dim, instance=inst)
            self.lower_bounds = self.f.lower_bounds
            self.upper_bounds = self.f.upper_bounds

        elif fct_name == "TCSD":
            self.f = create_tcsd_problem(dim=dim, instance=inst)
            self.lower_bounds = self.f.lower_bounds
            self.upper_bounds = self.f.upper_bounds

        elif fct_name == "SRD":
            self.f = create_srd_problem(dim=dim, instance=inst)
            self.lower_bounds = self.f.lower_bounds
            self.upper_bounds = self.f.upper_bounds

        if fct_name == "HNO":
            self.f = create_hno_problem(dim=dim, instance=inst)
            self.lower_bounds = self.f.lower_bounds
            self.upper_bounds = self.f.upper_bounds

        elif fct_name == "PVD":
            self.f = create_pvd_problem(dim=dim, instance=inst)
            self.lower_bounds = self.f.lower_bounds
            self.upper_bounds = self.f.upper_bounds

        elif fct_name == "TCSD":
            self.f = create_tcsd_problem(dim=dim, instance=inst)
            self.lower_bounds = self.f.lower_bounds
            self.upper_bounds = self.f.upper_bounds

        elif fct_name == "SRD":
            self.f = create_srd_problem(dim=dim, instance=inst)
            self.lower_bounds = self.f.lower_bounds
            self.upper_bounds = self.f.upper_bounds

        elif fct_name == "TBTD":
            self.f = create_tbtd_problem(dim=dim, instance=inst)
            self.lower_bounds = self.f.lower_bounds
            self.upper_bounds = self.f.upper_bounds

        elif fct_name == "GTD":
            self.f = create_gtd_problem(dim=dim, instance=inst)
            self.lower_bounds = self.f.lower_bounds
            self.upper_bounds = self.f.upper_bounds

        elif fct_name == "CBHD":
            self.f = create_cbhd_problem(dim=dim, instance=inst)
            self.lower_bounds = self.f.lower_bounds
            self.upper_bounds = self.f.upper_bounds

        elif fct_name == "TCD":
            self.f = create_tcd_problem(dim=dim, instance=inst)
            self.lower_bounds = self.f.lower_bounds
            self.upper_bounds = self.f.upper_bounds

        elif fct_name == "IBD":
            self.f = create_ibd_problem(dim=dim, instance=inst)
            self.lower_bounds = self.f.lower_bounds
            self.upper_bounds = self.f.upper_bounds

        elif fct_name == "CBD":
            self.f = create_cbd_problem(dim=dim, instance=inst)
            self.lower_bounds = self.f.lower_bounds
            self.upper_bounds = self.f.upper_bounds

        else:
            supported_problems = ["HNO", "PVD", "TCSD", "SRD", "TBTD", "GTD", "CBHD", "TCD", "IBD", "CBD"]
            raise ValueError(f"Unknown function name: {fct_name}. Supported problems are: {supported_problems}")


    def generate_random_elem(self):
        """Generates a random element within the bounds of the optimization domain.
           Returns:
               numpy.ndarray: A random element within the domain.
        """
        return self.f.lower_bounds + np.random.random(self.dim) * (self.f.upper_bounds - self.f.lower_bounds)

    def eval(self, element):
        """Evaluates the given element using the specified evaluation function.
            Args:
                element (numpy.ndarray): The element from the population.
            Returns:
                float: The fitness value.
        """
        fitness, violation = self.f(element)
        return fitness, violation

class Element:
    def __init__(self, x):
        """Constructor for the Element class.
            Args:
                x (numpy.ndarray): This element represents an individual in the population.
        """
        self.x = x
        self.cost = 0
        self.violation = 0
        self.repaired = [False] * len(x)  # Correction indicator for each component


    def copy(self):
        """Creates a copy of this element.
            Returns:
                Element: A new Element object with the same values.
        """
        new_el = Element(self.x.copy())
        new_el.cost = self.cost
        new_el.violation = self.violation
        new_el.repaired = self.repaired.copy()
        return new_el

    def update(self, el, cost, violation, repaired):
        """Updates the values of this element with the values from another element.
            Args:
                el (numpy.ndarray): The new values for x.
                cost (float): The new cost value.
                repaired (list): The new repaired value.
        """
        self.x = el.copy()
        self.cost = cost
        self.violation = violation
        self.repaired = repaired.copy()

    def __repr__(self):
        """Returns:
            str: A string representation of this element.
        """
        return "cost: {} element: {}".format(self.cost, self.x, self.violation)

class OptimizationAlgorithm:
    def __init__(self, problem, corr_type, corr_method, constraint_strategy):
        """Initializes the OptimizationAlgorithm instance.
            Args:
            problem (ContinuousOptimization): The optimization problem to be solved.
            probabilities_update_strategy (int): Strategy for updating the distribution probability for
            adaptive correction operator
            learning_period (int): Update the distribution probability every LP generations
        """
        self.problem = problem
        self.population = []
        self.nr_infeasible = 0  # Number of infeasible components
        self.nr_mutated = 0  # Number of mutated elements

        self.generation_counter = 0  # Generation counter
        self.epsilon = 1e-10  # Prevent zero division for violation probability
        self.epsVar = 10 ** (-15)  # Variance error
        self.gamma = -1  # No vectorial correction used

        # Correction initialization
        self.correction_handler = CorrectionHandler(problem)
        self.corr_type = corr_type
        self.corr_method = corr_method
        self.constraint_strategy = constraint_strategy

        # Store bounds from the problem
        self.lb = self.problem.lower_bounds
        self.ub = self.problem.upper_bounds


    def initialize_population(self, pop_size):
        """Initializes the population with random elements and proper evaluation.
        Args:
            pop_size (int): The size of the population.
        """
        self.population = []
        for _ in range(pop_size):
            x = self.problem.generate_random_elem()
            element = Element(x)

            fitness, violation = self.problem.eval(x)
            element.cost = fitness
            if self.constraint_strategy == "PENALTY":
                element.cost += 1e6 * violation
            element.violation = violation

            self.population.append(element)

    def evaluate_population(self):
        """Evaluates the population using the evaluation function defined in the ContinuousOptimization class."""
        for element in self.population:
            cost, violation = self.problem.eval(element.x)
            if self.constraint_strategy == "PENALTY":
                element.cost = cost + 1e6 * violation
            else:
                element.cost = cost
            element.violation = violation


    def correct_component(self, method, component, target, best, lower, upper, population_mean):
        """Returns: component, repaired"""
        return self.correction_handler.correction_component(
            method, lower, upper,
            component, target, best, population_mean)

    def correct_vector(self, vector, R, lower, upper):
        """Returns: repaired_vector, repaired, gamma"""
        return self.correction_handler.correction_vectorial(lower, upper, vector, R)

    def bounding_box(self):
        min_vals = np.full(self.problem.dim, self.ub)
        max_vals = np.full(self.problem.dim, self.lb)

        for el in self.population:
            min_vals = np.minimum(min_vals, el.x)
            max_vals = np.maximum(max_vals, el.x)

        self.BBlower = min_vals
        self.BBupper = max_vals

    def bounding_box_ratios(self):
        self.bounding_box()
        BBmin, BBmax = self.BBlower, self.BBupper
        ratios_min = np.zeros(self.problem.dim)
        ratios_max = np.zeros(self.problem.dim)
        for i in range(self.problem.dim):
            if BBmin[i] != self.lb[i]:
                ratios_min[i] = (BBmax[i] - BBmin[i]) / (BBmin[i] - self.lb[i])
            else:
                ratios_min[i] = np.inf

            if BBmax[i] != self.ub[i]:
                ratios_max[i] = (BBmax[i] - BBmin[i]) / (self.ub[i] - BBmax[i])
            else:
                ratios_max[i] = np.inf

        return ratios_min, ratios_max

    def bounding_box_ratios_closeness(self):
        self.bounding_box()
        BBmin, BBmax = self.BBlower, self.BBupper
        ratios_min = np.zeros(self.problem.dim)
        ratios_max = np.zeros(self.problem.dim)
        out_lower = []  # Lista pentru a stoca care latura a 'cutiei fezabile' este mai apropiata pentru fiecare dimensiune
        out_upper = []

        for i in range(self.problem.dim):
            if BBmin[i] != self.lb[i]:
                ratios_min[i] = (BBmax[i] - BBmin[i]) / (BBmin[i] - self.lb[i])
            else:
                ratios_min[i] = np.inf

            if BBmax[i] != self.ub[i]:
                ratios_max[i] = (BBmax[i] - BBmin[i]) / (self.ub[i] - BBmax[i])
            else:
                ratios_max[i] = np.inf

            # Calculeaza distantele de la BBlower si BBupper la limitele problemelor lb si ub
            distance_to_lower_bound = BBmin[i] - self.lb[i]
            distance_to_upper_bound = self.ub[i] - BBmax[i]

            # Determina care latura este mai apropiata
            if distance_to_lower_bound < distance_to_upper_bound:
                out_lower.append((distance_to_lower_bound, 'Lower'))
            else:
                out_upper.append((distance_to_upper_bound, 'Upper'))

        return (out_lower, out_upper)

    def bounding_box_proportion(self):
        self.bounding_box()
        BBmin, BBmax = self.BBlower, self.BBupper
        ratio = np.zeros(self.problem.dim)
        for i in range(self.problem.dim):
            ratio[i] = (BBmax[i] - BBmin[i]) / (self.ub[i] - self.lb[i])
        return ratio

    def compute_extension(self):  # use sqrt
        return (np.prod((self.BBupper - self.BBlower) / (self.ub - self.lb)))**(1/self.problem.dim)

    def compute_density(self):  # move if before division; extract sqrt from volume
        volume = np.prod(self.BBupper - self.BBlower)
        return len(self.population) / np.sqrt(volume) if volume != 0 else 0

    def compute_shape(self):
        extension = self.BBupper - self.BBlower
        return np.min(extension) / np.max(extension)

    def compute_eccentricity(self):  # imparte la diagonala
        # Compute the centroid of the population
        centroid = np.mean([el.x for el in self.population], axis=0)

        # Compute the middle point of the search space
        middle = (self.BBupper + self.BBlower) / 2

        # Compute the distance from the centroid to the middle
        distance = np.linalg.norm(centroid - middle)

        # Compute the length of the diagonal of the search space
        diagonal_length = np.linalg.norm(self.BBupper - self.BBlower)

        # Normalize the distance by the length of the diagonal
        eccentricity = distance / diagonal_length

        return eccentricity

    def compute_uniformity(self, m):  # m=popSize
        uniformity = np.zeros(self.problem.dim)
        for j in range(self.problem.dim):
            segment_length = (self.BBupper[j] - self.BBlower[j]) / m
            segment_counts = np.zeros(m)
            for i in range(m):
                lower_bound = self.BBlower[j] + i * segment_length
                upper_bound = lower_bound + segment_length
                segment_counts[i] = sum(lower_bound <= el.x[j] < upper_bound for el in self.population)
            p_ij = segment_counts / len(self.population)
            # Avoid division by zero and log(0)
            p_ij = p_ij[p_ij > 0]
            uniformity[j] = -np.sum(p_ij * np.log(p_ij))  # if p=0 do not compute log(0) set 0
        return uniformity

    def compute_distance_to_optimum(self):
        distances = []
        for i in range(self.problem.dim):
            if self.problem.f.optimum.x[i] < self.BBlower[i]:
                distance = self.BBlower[i] - self.problem.f.optimum.x[i]
            elif self.problem.f.optimum.x[i] > self.BBupper[i]:
                distance = self.problem.f.optimum.x[i] - self.BBupper[i]
            else:
                distance = 0  # Optimum is inside the bounding box
            distances.append(distance)
        return np.linalg.norm(distances)

    def optimize(self):
        """Placeholder method for optimization. Should be implemented by subclasses."""
        raise NotImplementedError("The subclasses must implement optimize method!")

class AdaptationOfParameters(OptimizationAlgorithm):
    def __init__(self, problem, pop_size, sizeH, NFE_max, corr_type, corr_method, constraint_strategy, N_min):
        super().__init__(problem, corr_type, corr_method, constraint_strategy)
        self.dim = problem.dim
        self.initialize_population(pop_size)
        self.evaluate_population()
        self.measures = MonitoringPopulation(problem, self.population)
        self.global_best = min(self.population, key=lambda element: element.cost)
        self.crossover_points = np.zeros(self.dim, dtype=bool)
        self.repaired = [False] * self.dim

        # Archives for success factors initialization
        self.sizeH = sizeH
        self.NFE_max = NFE_max
        self.memF = [0.5] * sizeH  # initial archive for F
        self.memCR = [0.5] * sizeH  # initial archive for CR
        self.k = 1  # index for success control parameters values actualization

        # Adaptive population size initializations
        self.N_min = N_min  # smallest population size
        # self.Ninit = 18 * self.dim  # initial population size
        self.Ninit = 180
        self.pop_size = self.Ninit
        self.pmin = 2 / self.pop_size

        # Memory initialization
        self.memA = copy.deepcopy(self.population)
        self.pop_size = pop_size
        self.avgF = 0
        self.stdF = 0
        self.avgCR = 0
        self.stdCR = 0

    def generate_F_CR_p_values(self, sizeH, memF, memCR, pmin):
        # sizeH  - size of the archive used for F and CR
        # memF - archive for F
        # memCR - archive for CR
        # pmin - minimum proportion of population
        idxH = np.random.randint(0, sizeH)  # random index in the archives
        muF = memF[idxH]  # mean of the Cauchy distribution for F
        muCR = memCR[idxH]  # mean of the Gaussian distribution for CR
        sd = 0.1  # standard deviations of the distributions used to generate F and CR

        # Fi = np.random.normal(muF, sd)
        Fi = np.random.standard_cauchy() * sd + muF
        while Fi <= 0:
            # Fi = np.random.normal(muF, sd)
            Fi = np.random.standard_cauchy() * sd + muF
        if Fi > 1:
            Fi = 1

        CRi = np.random.normal(muCR, sd)
        CRi = np.clip(CRi, 0, 1)
        pi = np.random.rand() * (0.2 - pmin) + pmin
        return Fi, CRi, pi

    def update_mem_F_CR(self, SF, SCR, improvements):
        total = np.sum(improvements)
        # Total might be 0 when selection accepts elements with the same fitness value

        if (total > 0):
            weights = improvements / total
        else:
            weights = np.array([1 / len(SF)] * len(SF))
        Fnew = np.sum(weights * SF * SF) / np.sum(weights * SF)  # Lehmer mean
        Fnew = np.clip(Fnew, 0, 1)
        CRnew = np.sum(weights * SCR)  # weighted mean
        CRnew = np.clip(CRnew, 0, 1)
        return Fnew, CRnew

    def limit_memory(self, memory, memorySize):
        """
        Limit the memory to  the memorySize by removing randomly selected elements
        """
        if len(memory) > memorySize:
            indexes = np.random.permutation(len(memory))[:memorySize]
            return [memory[index] for index in indexes]
        else:
            return memory

    def update_F_CR_archives(self, SCR, SF, improvements):
        # Update MemF and MemCR
        if len(SCR) > 0 and len(SF) > 0:  # at least one successful trial vector
            Fnew, CRnew = self.update_mem_F_CR(SF, SCR, improvements)
            self.memF[self.k] = Fnew
            self.memCR[self.k] = CRnew
            self.k = (self.k + 1) % self.sizeH  # limit the memory - old values are overwritten
            self.avgF = np.mean(self.memF)
            self.avgCR = np.mean(self.memCR)
            self.stdF = np.std(self.memF)
            self.stdCR = np.std(self.memCR)

    def mutation(self, el):
        Fi, CRi, p = self.generate_F_CR_p_values(self.sizeH, self.memF, self.memCR,
                                                 self.pmin)  # for each trial vector a new F, CR and p value is generate
        maxbest = int(p * len(self.population))
        idx_pBest = np.random.randint(low=0, high=maxbest + 1)
        pbest = self.population[idx_pBest]
        r1, r2 = -1, -1  # Initialize to invalid values

        while True:
            r1 = np.random.randint(low=0, high=len(self.population))
            r2 = np.random.randint(low=0, high=len(self.memA))

            # Break out of loop if both r1 and r2 are different and
            # the x values of the selected elements are also different
            if r1 != r2 and not np.all(self.population[r1].x == self.memA[r2].x):
                break

        diff1= pbest.x - el.x
        diff2= self.population[r1].x - self.memA[r2].x
        mutantVector = el.x + Fi * diff1 + Fi * diff2

        return mutantVector, Fi, CRi, diff1, diff2

    def crossover(self, mutantVector, el, CR):
        self.crossover_points = np.random.rand(self.dim) < CR
        # Test if there is the possibility of performing crossover, if not, randomly create one
        if not np.any(self.crossover_points):
            self.crossover_points[np.random.randint(0, self.dim)] = True
        # If crossover_points=true, return the element from the mutant, otherwise,
        # return the element from the current population
        trialVector = np.where(self.crossover_points, mutantVector, el.x)
        self.nr_mutated += np.sum(self.crossover_points)
        return trialVector

    def generate_trial_vector(self, el, cov, population_mean):
        self.nr_infeasible = 0  # Number of infeasible components
        self.nr_mutated = 0  # Number of mutated elements
        mutantVector, F, CR, diff1, diff2 = self.mutation(el)
        trialVector = self.crossover(mutantVector, el, CR)
        bchm = self.corr_method


        if bchm == METHOD_VECTOR_BEST:
            trialVector, self.repaired, gamma = self.correct_vector(
                vector=trialVector,
                R=self.global_best.x,
                lower=self.lb,
                upper=self.ub
            )
            for i in range(self.dim):
                if self.repaired[i] is True:
                    self.nr_infeasible += 1

        elif bchm == METHOD_VECTOR_TARGET:
            trialVector, self.repaired, gamma = self.correct_vector(
                vector=trialVector,
                R=el.x,
                lower=self.lb,
                upper=self.ub
            )
            for i in range(self.dim):
                if self.repaired[i] is True:
                    self.nr_infeasible += 1

        else:
            for i in range(self.dim):
                trialVector[i], self.repaired[i] = self.correct_component(
                    # method=self.corr_method,
                    method=bchm,
                    component=trialVector[i],
                    target=el.x[i],
                    best=self.global_best.x[i],
                    lower=self.lb[i],
                    upper=self.ub[i],
                    population_mean=population_mean
                )
                if self.repaired[i] is True:
                    self.nr_infeasible += 1

        return trialVector, bchm,F, CR, diff1, diff2
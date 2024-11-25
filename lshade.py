from core import AdaptationOfParameters, Element
from core import time_checker
from core import HashTable
import pandas as pd
from correction_handler import *

class LSHADE(AdaptationOfParameters):
    def __init__(self, problem, pop_size, sizeH, NFE_max, N_min, corr_type, corr_method, run_number, path,
                 constraint_strategy):
        super().__init__(
        problem = problem,
        pop_size = pop_size,
        sizeH = sizeH,
        NFE_max = NFE_max,
        corr_type = corr_type,
        corr_method = corr_method,
        constraint_strategy = constraint_strategy,
        N_min = N_min
        )
        self.run_number = run_number
        self.path = path
        self.constraint_strategy = constraint_strategy
        self.feasible_best=None

    def generate_csv_filename(self):
        alg_name = "LSHADE"
        func_name = f"f{self.problem.eval_fct}"
        dim = f"D{self.problem.dim}"
        strategy = f"{self.constraint_strategy}"
        correction = corr_names[self.corr_method]

        filename_parts = [alg_name, correction, func_name, dim, strategy,
                          f"run{self.run_number}"]
        filename = "_".join(part for part in filename_parts if part) + "_gen.csv"# Exclude componentele goale
        filepath = self.path / filename
        return filepath

    @time_checker
    def optimize(self):
        correction_probabilities = []  # Collecting probabilities for BCHM adaptive over generations
        NFE = 0  # Number of Function Evaluation
        NFS = 0  # Number of successful trial vectors/generation
        totalInfeasibleComponent = 0
        totalMutatedComponent = 0
        totalInfeasibleElement = 0

        max_generations = self.NFE_max
        rez = [None] * max_generations
        BB_metrics = [None] * max_generations
        diff_strs = [None] * max_generations
        X_vect = [None] * max_generations

        meanPopVector, varPopVector, varPop = self.measures.variance(self.population)
        # mean and variance after linear transformation [lower, upper] -> [0,1]
        meanPopVector = (meanPopVector - self.lb) / (self.ub - self.lb)
        varPopVector = varPopVector / (
                (self.ub - self.lb) * (self.ub - self.lb))
        covPop = self.measures.covariance(self.population)

        hash_table = HashTable()
        total_collisions = 0

        while (NFE < self.NFE_max) and ((abs(self.global_best.cost - self.problem.f.optimum.y) >= 10 ** (-8)) or self.global_best.violation>0) and varPop >= 10**(-10):
            difference1 = []
            difference2 = []
            self.generation_counter += 1
            self.bounding_box()
            bb_ratios = self.bounding_box_ratios()
            closeness = self.bounding_box_ratios_closeness()
            density = self.compute_density()
            dist_to_opt = self.compute_distance_to_optimum()
            shape = self.compute_shape()
            uniformity = self.compute_uniformity(m=self.pop_size)
            extension = self.compute_extension()
            eccentricity = self.compute_eccentricity()

            gammaList = []
            genInfeasibleElement = 0  # number of infeasible elements
            genInfeasibleComponent = 0  # number of infeasible components
            genMutatedComponent = 0  # number of mutated components
            genSuccessMutants = 0
            meanImprovements = 0
            meanImprovementsMut = 0
            SF = []
            SCR = []
            improvements = []
            new_population = []
            collision_counter = 0
            feasible_population=[]

            for el_idx, el in enumerate(self.population):
                # Generate a trial vector
                trial_vector, selected_bchm, F, CR, diff1, diff2 = self.generate_trial_vector(el, covPop,
                                                                                                        np.mean(
                                                                                                            meanPopVector))
                if hash_table.insert(Element(trial_vector)):
                    collision_counter += 1
                    # print('Collision detected:', trial_vector)
                difference1.append([round(x, 2) for x in diff1])
                difference2.append([round(x, 2) for x in diff2])
                if self.gamma != -1:
                    gammaList.append(self.gamma)
                genMutatedComponent = genMutatedComponent + self.nr_mutated
                if self.nr_infeasible > 0:
                    genInfeasibleComponent = genInfeasibleComponent + self.nr_infeasible
                    genInfeasibleElement = genInfeasibleElement + 1

                # Evaluate the fitness of the trial vector
                trial_fitness, constraint_violations = self.problem.eval(trial_vector)
                NFE += 1  # Increment the number of function evaluations

                if (self.constraint_strategy == "PENALTY"):
                    trial_fitness = trial_fitness + 1e6 * constraint_violations
                    is_success = trial_fitness < el.cost

                else:
                    if constraint_violations == el.violation == 0:
                        is_success = trial_fitness < el.cost
                        # print(f"Both feasible: trial={trial_fitness}, el={el.cost}, success={is_success}")

                    elif constraint_violations == 0 and el.violation > 0:
                        is_success = 1
                        # print(f"Trial feasible, el not: trial_viol={constraint_violations}, el_viol={el.violation}")

                    elif constraint_violations > 0 and el.violation == 0:
                        is_success = 0
                        # print(f"El feasible, trial not: trial_viol={constraint_violations}, el_viol={el.violation}")

                    else:
                        is_success = constraint_violations < el.violation
                        # print(f"Neither feasible: trial_viol={constraint_violations}, el_viol={el.violation}")

                if is_success:
                    # el.update(trial_vector, trial_fitness, self.repaired)
                    new_population.append(Element(trial_vector))
                    new_population[el_idx].cost = trial_fitness
                    new_population[el_idx].violation = constraint_violations
                    new_population[el_idx].repaired = self.repaired
                    NFS += 1  # number of successful mutations
                    genSuccessMutants += 1
                    SF.append(F)
                    SCR.append(CR)
                    improvements.append(el.cost - trial_fitness)
                    self.memA.append(el)
                else:
                    new_population.append(el)

            if improvements:
                meanImprovements = sum(improvements) / len(improvements)
                meanImprovementsMut = sum(improvements) / genSuccessMutants
            else:
                meanImprovements = 0
                meanImprovementsMut = 0

            if NFS > 1:
                # Update the global best solution
                # print(f"Before sort - best violation: {self.global_best.violation}")
                new_population = sorted(new_population, key=lambda element: element.cost)

                for el_idx, el in enumerate(self.population):
                    self.population[el_idx] = new_population[el_idx].copy()

                for el_idx, el in enumerate(self.population):
                    # el.update(new_population[el_idx].x, new_population[el_idx].cost, new_population[el_idx].repaired)
                    # print(f"Before: {self.population[el_idx].x}, Cost: {self.population[el_idx].cost}")
                    self.population[el_idx] = new_population[el_idx].copy()
                    # print(f"After: {self.population[el_idx].x}, Cost: {self.population[el_idx].cost}")

                self.feasible_best=min(self.population,
                      key=lambda element: (element.violation if element.violation > 0 else -1, element.cost))
                self.global_best = min(self.population, key=lambda element: element.cost)
                # print(f"After sort - best violation: {self.global_best.violation}")
                is_best_feasible = self.global_best.violation == 0
                meanPopVector, varPopVector, varPop = self.measures.variance(self.population)
                meanPopVector = (meanPopVector - self.lb) / (self.ub - self.lb)
                varPopVector = varPopVector / ((self.ub - self.lb) * (self.ub - self.lb))
                covPop = self.measures.covariance(self.population)

                for i in range(self.problem.dim):
                    if varPopVector[i] < self.epsVar:
                        varPopVector[i] = self.epsVar

            totalInfeasibleComponent = totalInfeasibleComponent + genInfeasibleComponent
            totalMutatedComponent = totalMutatedComponent + genMutatedComponent
            totalInfeasibleElement = totalInfeasibleElement + genInfeasibleElement
            total_collisions += collision_counter

            # print(f"New pop:{new_population} , \n , Self:{self.population}")

            self.memA = self.limit_memory(self.memA, len(self.population))
            self.update_F_CR_archives(SCR, SF, improvements)
            newPopSize = round(self.Ninit - NFE / self.NFE_max * (self.Ninit - self.N_min))
            self.population = self.population[0:newPopSize]
            self.pop_size = newPopSize
            feasible_population = [x for x in self.population if x.violation == 0]
            num_feasible = len(feasible_population)
            # print(f"Gen {self.generation_counter}:")
            # print(f"Best solution - cost: {self.global_best.cost}, violations: {self.global_best.violation}")
            # print(f"Feasible best - cost: {self.feasible_best.cost}, violations: {self.feasible_best.violation}")

            try:
                # estimation of ViolationProbability*MutationProbability (is counted only for components selected by crossover)
                prob_infeas = genInfeasibleComponent / genMutatedComponent
                # print(f"STEP 1 - Values at calculation:")
                # print(f"prob_infeas type: {type(prob_infeas)}")
                # print(f"prob_infeas value: {prob_infeas}")

            except ZeroDivisionError:
                prob_infeas = genInfeasibleComponent / (genMutatedComponent + self.epsilon)

            if prob_infeas>1:
                print(f"!!!!!!!!!Debug - genInfeasibleComponent: {genInfeasibleComponent}, "
                      f"genMutatedComponent: {genMutatedComponent}, "
                      f"ratio: {genInfeasibleComponent / genMutatedComponent},"
                      f"prob_infeas: {prob_infeas}")
            #
            # print(f"STEP 2 - Value going into rez:")
            # print(f"prob_infeas value: {prob_infeas}")

            rez[self.generation_counter - 1] = [
                self.generation_counter,
                self.pop_size,
                num_feasible,
                num_feasible/self.pop_size,
                self.global_best.cost,
                self.global_best.cost - self.problem.f.optimum.y,
                self.feasible_best.cost - self.problem.f.optimum.y,
                is_best_feasible,
                self.feasible_best.cost,
                self.global_best.violation,
                prob_infeas,
                genInfeasibleElement,
                genInfeasibleComponent,
                genMutatedComponent,
                genSuccessMutants / self.pop_size,
                meanImprovements,
                meanImprovementsMut,
                varPop,
                self.avgF,
                self.stdF,
                self.avgCR,
                self.stdCR
            ]

            BB_metrics[self.generation_counter - 1] = {
                'bb_ratios': bb_ratios,
                'closeness': closeness,
                'extension': extension,
                'density': density,
                'shape': shape,
                'eccentricity': eccentricity,
                'uniformity': uniformity,
                'dist_to_opt': dist_to_opt,
                'collision_counter': collision_counter,
                'total_collisions': total_collisions
            }

            X_vect[self.generation_counter - 1] = {'X': self.global_best.x}
            diff_strs[self.generation_counter - 1] = {
                'diff1': difference1,
                'diff2': difference2
            }

        final_rez = [x for x in rez if x is not None]
        final_BB = [x for x in BB_metrics if x is not None]
        final_X = [x for x in X_vect if x is not None]
        final_diff = [x for x in diff_strs if x is not None]



        columns_rez = ['it', 'pop_size', 'num_feas','feas_ratio','best_cost', 'error', "feasible_error", "is_best_feasible","feasible_best_cost", "best violation", "prob_infeas",
                       'gen_infeasible_element','genInfeasibleComponent', 'gen_mutated', 'success_ratio', 'mean_improvements',
                       'mean_improvements_mut', 'var_pop', 'avg_F', 'std_F', 'avg_CR', 'std_CR']

        # print("DEBUG before DataFrame creation:")
        # print(f"final_rez prob_infeas values: {[x[4] for x in final_rez]}")

        df_rez = pd.DataFrame(final_rez, columns=columns_rez)
        #
        # print("DEBUG after DataFrame creation:")
        # print(f"DataFrame prob_infeas values: {df_rez['prob_infeas'].values}")
        # print(f"DataFrame prob_infeas dtypes: {df_rez['prob_infeas'].dtype}")


        # print(f"STEP 3 - Value in DataFrame:")
        # print(f"DataFrame value type: {type(df_rez['prob_infeas'].iloc[0])}")
        # print(f"DataFrame value: {df_rez['prob_infeas'].iloc[0]}")
        df_BB = pd.DataFrame(final_BB)
        df_X = pd.DataFrame(final_X)
        df_diff = pd.DataFrame(final_diff)

        # if abs(df_rez['prob_infeas'].iloc[0] - (genInfeasibleComponent / genMutatedComponent)) > 1e-10:
        #     print(f"!!!!!!!!!!1Warning: Division mismatch detected!")
        #     print(f"prob_infeas: {prob_infeas}")
        #     print(f"direct division: {genInfeasibleComponent / genMutatedComponent}")

        # Concatenate with proper alignment
        df_final = pd.concat([df_rez, df_BB, df_diff, df_X], axis=1)
        csv_filename = self.generate_csv_filename()
        df_final.to_csv(csv_filename, index=False, float_format='%.8f')

        return self.global_best
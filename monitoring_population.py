import numpy as np


class MonitoringPopulation:
    def __init__(self, problem, population):
        self.problem = problem
        self.dim = problem.dim
        self.population = population

    def variance(self, population):
        # Return: mean per component, var per component, averaged population variance
        s = np.sum([el.x for el in population], axis=0)
        s2 = np.sum([el.x * el.x for el in population], axis=0)
        var = s2 / len(population) - (s / len(population)) ** 2
        return s / len(population), var, var.mean()

    def covariance(self, population):
        mean = np.zeros(self.dim)
        for ind in range(self.dim):
            for el in population:
                mean[ind] += el.x[ind]
            mean[ind] /= len(population)
        cov = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(i, self.dim):
                for el in population:
                    cov[i, j] += (el.x[i] - mean[i]) * (el.x[j] - mean[j])
                cov[i, j] /= len(population)
                cov[j, i] = cov[i, j]  # use symmetry to fill the lower triangle
        return cov













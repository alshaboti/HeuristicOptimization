
import numpy as np
from scipy.spatial.distance import euclidean, cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import random
from simanneal import Annealer
import sys, math

# GA
from deap import base
from deap import creator
from deap import tools

import  itertools

class HillClimbing:
    """"Hill climbing class"""

    def __init__(self, get_neighbor, get_score):
        self.get_neighbor = get_neighbor
        self.get_score = get_score

    def climb(self, init_node):
        current_node = init_node

        current_score = self.get_score(current_node)[0]
        next_score = current_score
        next_node = current_node

        while True:
            neighbors_list = self.get_neighbor(next_node)
            # go through all nieghbors to get the max score (next_score)
            for neighbor_point in neighbors_list:
                neigh_score = self.get_score(neighbor_point)[0]
                # print("neighbor score:", neigh_score)
                if (neigh_score > next_score):
                    next_node = neighbor_point
                    next_score = neigh_score

            #print("HC: next score", next_score, self.get_score(next_node)[1])

            # if all neighbor score less than the current then end
            if next_score <= current_score:
                return current_node, current_score
            # otherwise jump to the next best neighbor point.
            else:
                current_score = next_score
                current_node = next_node

#https://github.com/perrygeo/simanneal
class TasktoIoTmapingProblem(Annealer):
    """" Mapping tasks functions to a best combination of devices preferred by user"""

    # rand.seed(58479)
    # Tmax = 10000
    # steps = 10000
    # updates = 1000

    def __init__(self, init_state, problem_model,get_score ):
        Annealer.__init__(self, init_state)
        self.problem_model = problem_model
        self.get_score = get_score
        # if you don't want to see any update in the output
        self.updates=0

    def move(self):
        """"select random neighbor"""
        neighbors = self.problem_model.get_all_neighbors(self.state)
        # print(neighbors)
        self.state = neighbors[random.randint(0, len(neighbors) - 1)]
        # print(self.state)

    def energy(self):
        """" calculate the spanning tree distance """
        e = self.get_score(self.state)[0]
        # print(self.state, e)
        return 1 - e


class BruteForceSearch:
    """" Brute Force Search class"""

    def __init__(self, subt_dev_dict,task, get_score):
        self.task = task
        self.subt_dev_dict = subt_dev_dict
        self.get_score = get_score

    def run(self):
        max_score = -1
        best_cand = []
        alts_dev = [self.subt_dev_dict[f] for f in self.task]
        cand_prod = itertools.product(*alts_dev)
        
        for fun_dev_cand in cand_prod:
            tmp_cand = list(fun_dev_cand)
            tmp_score = self.get_score(tmp_cand)[0]
            #print("best_cand, max_score, :",  best_cand, max_score)
            #print("tmp, score, :", tmp_cand, tmp_score)
            if tmp_score > max_score:
                max_score = tmp_score
                best_cand = tmp_cand
                if max_score == 1.0:
                    return max_score, best_cand
        return max_score, best_cand

class GA:
    """ GA algorithm """

    def __init__(self, prob_domain, get_score):
        self.prob_domain = prob_domain
        self.get_score = get_score

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Attribute generator
        #                      define 'attr_bool' to be an attribute ('gene')
        #                      which corresponds to integers sampled uniformly
        #                      from the range [0,1] (i.e. 0 or 1 with equal
        #                      probability)
        # self.toolbox.register("attr_bool", random.randint, 0, 1)

        # Structure initializers
        #                         define 'individual' to be an individual
        #                         consisting of 100 'attr_bool' elements ('genes')
        self.toolbox.register("individual", tools.initIterate, creator.Individual,
                              prob_domain.get_rand_solution)

        # define the population to be a list of individuals
        self.toolbox.register("population", tools.initRepeat, list, \
            self.toolbox.individual)

        # ----------
        # Operator registration
        # ----------
        # register the goal / fitness function
        self.toolbox.register("evaluate", self.get_score)

        # register the crossover operator
        self.toolbox.register("mate", tools.cxTwoPoint)

        # register a mutation operator with a probability to
        # flip each attribute/gene of 0.05
        self.toolbox.register("mutate", self.flipFunDev, indpb=0.05)

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def flipFunDev(self, indv, indpb):
        if random.random() > indpb:
            return indv

        # select random func to flip its device
        idx = random.randint(0, len(indv) - 1)
        # get the available devices for that func
        
        f_idx  = self.prob_domain.task[idx]
        f_devs_list = self.prob_domain.func_alter_devices[f_idx]
        
        
        if len(f_devs_list) > 1:
            # select dev_id other than the existing one in indv
            d_idx = random.choice(f_devs_list)
            
            while d_idx == indv[idx]:            
                d_idx = random.choice(f_devs_list)

            indv[idx] = d_idx
        
        return indv

    def run(self, n=100, max_iteration=100):
        # random.seed(64)

        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = self.toolbox.population(n=n)

        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        CXPB, MUTPB = 0.5, 0.2

        # print("Start of evolution")

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, pop))

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # print("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0

        # Begin the evolution
        while max(fits) < 10 and g < max_iteration:
            # A new generation
            g = g + 1
            # print("-- Generation %i --" % g)

            # Select the next generation individuals
            # same invd selected more than once! check if okay.
            offspring = self.toolbox.select(pop, len(pop))

            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)

                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            # print("  Min %s" % min(fits))
            # print("  Max %s" % max(fits))
            # print("  Avg %s" % mean)
            # print("  Std %s" % std)
        #
        # print("-- End of (successful) evolution --")

        best_ind = tools.selBest(pop, 1)[0]
        # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        # print(self.prob_domain.func_alter_devices)
        return (best_ind, best_ind.fitness.values)



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
    # default values
    # Tmax = 25000
    # Tmin = 2.5
    # steps = 50000
    # updates = 100
    def __init__(self, init_state, problem_model,get_score ):
        Annealer.__init__(self, init_state)
        self.problem_model = problem_model
        self.get_score = get_score
        # if you don't want to see any update in the output
        self.updates = 0
        self.steps = 100000
        # it was 10000  
    def move(self):
        """"select random neighbor"""
        neighbors = self.problem_model.get_all_neighbors(self.state)
        # print(neighbors)
        self.state = random.choice(neighbors)
        #[random.randint(0, len(neighbors) - 1)]
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
    """ GA algorithm 
    Generation_n *
    population_size *
    crossover = x *
    else
     mutation
    elitizem = 0.1
    selection method Tournament, size 3
    stop creteria
    initialization 

    """
    
    def __init__(self, prob_domain, get_score):
        self.prob_domain = prob_domain
        self.get_score = get_score

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual,
                              prob_domain.get_rand_solution)

        # define the population to be a list of individuals
        self.toolbox.register("population", \
            tools.initRepeat, list, self.toolbox.individual)

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

    def run(self, generations =100, max_iteration=100):

        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = self.toolbox.population(n=generations)
        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        CXPB, MUTPB = 0.7, 0.2

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, pop))

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        gen = 0
        # stop if for 10 gen improvement doesn't increase 
        fit_margin = 0.01
        no_improv_gen = 0
        old_fits = -1
        # Begin the evolution
        while gen < max_iteration:
            # A new generation
            gen = gen + 1

            # (alshaboti) from here
            max_fits = max(fits) 
            # check if there is any improvement
            if max_fits < (old_fits+ fit_margin):
                no_improv_gen +=1
                # if this is the 10th generation without improvement break
                if no_improv_gen > 9:
                    break 
            else:
                no_improv_gen = 0

            # (alshaboti) to here

            old_fits = max_fits
            # Select the next generation individuals
            # elitisin (alshaboti) 10%
            n_elitisin = int(generations*0.1)
            best_childs = tools.selBest(pop, n_elitisin)

            # same invd selected more than once! check if okay.
            offspring = self.toolbox.select(pop, len(pop)-n_elitisin)

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
            # add the best from last generation
            pop.extend(best_childs)

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            # length = len(pop)
            # mean = sum(fits) / length
            # sum2 = sum(x * x for x in fits)
            # std = abs(sum2 / length - mean ** 2) ** 0.5

            # # print("  Min %s" % min(fits))
            # # print("  Max %s" % max(fits))
            # # print("  Avg %s" % mean)
            # # print("  Std %s" % std)

        best_ind = tools.selBest(pop, 1)[0]
        # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        # print(self.prob_domain.func_alter_devices)
        return (best_ind, best_ind.fitness.values)

